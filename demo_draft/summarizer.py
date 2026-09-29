"""Evidence-grounded summarizer for the retrieval UI.

Strict boundary: this module summarizes ONLY the ``EvidenceItem``s handed to it
by the Day 2 retriever. It deliberately does **not** import or call
``load_records`` / ``get_record``, does not read the CSVs, and does not touch
the image. The only input is the already-retrieved evidence, so the data path is

    retriever  →  summarizer

never

    dataset  →  LLM

It also never receives the pneumonia label or the ``outcome_*`` columns — those
are excluded from ``clinical_evidence`` upstream, and :func:`build_evidence_context`
asserts none of them can appear in the context (defense in depth).

Model access is optional and pluggable. If no LLM API key is configured the app
keeps working and reports that the AI summary is unavailable — it never
fabricates a summary and labels it AI-generated.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence

# Label/outcome fields that must never reach the summarizer.
FORBIDDEN_LABEL_FIELDS = frozenset(
    {"Pneumonia", "outcome_all_pne", "outcome_bac_pne", "outcome_viral_pne"}
)

# A provider is any callable that takes (system_prompt, user_payload_json) and
# returns the model's text. This keeps the underlying model/provider swappable.
Provider = Callable[[str, str], str]

SYSTEM_PROMPT = (
    "You are an evidence-grounded clinical information summarizer.\n"
    "Your task is to summarize ONLY the evidence provided to you.\n"
    "Do not add medical facts that are not present in the evidence.\n"
    "Do not diagnose the patient.\n"
    "Do not infer conditions from numeric or binary values.\n"
    "Do not reinterpret binary fields (a value of 0 or 1 is a recorded dataset "
    "field, not the presence or absence of a condition).\n"
    "Do not recommend treatment.\n"
    "Do not provide medical advice.\n"
    "Preserve uncertainty and attribution from radiology reports (keep words "
    "like 'may', 'could', 'suggests' exactly as attributed to the report).\n"
    "Clearly distinguish structured clinical data from statements made in the "
    "radiology report.\n"
    "Every substantive statement must be supported by the supplied evidence.\n"
    "If the evidence does not support a conclusion, say that the available "
    "retrieved evidence is insufficient.\n"
    "Return a concise 3-5 sentence summary followed by source references. Cite "
    "evidence inline using bracketed field names, e.g. [triage_heartrate] for a "
    "structured field or [radiology_report] for a report excerpt."
)


class SummarizerError(Exception):
    """Raised for invalid summarizer input (never for a missing API key)."""


def _item_attr(item: Any, name: str, default: Any = None) -> Any:
    """Read an attribute from an EvidenceItem or a key from a dict."""
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def build_evidence_context(evidence_items: Sequence[Any]) -> List[Dict[str, Any]]:
    """Build the exact, minimal context sent to the model.

    For each item only these keys are included:
      - clinical / image_metadata: source, field, value, matched_terms, reason
      - radiology: source, field, excerpt, matched_terms, reason

    Nothing else (no scores are required, no dataset rows, no labels). Raises if a
    forbidden label field is ever encountered.
    """
    context: List[Dict[str, Any]] = []
    for item in evidence_items:
        source = _item_attr(item, "source")
        field = _item_attr(item, "field")
        if field in FORBIDDEN_LABEL_FIELDS:
            raise SummarizerError(
                f"Refusing to summarize forbidden label field '{field}'."
            )
        entry: Dict[str, Any] = {
            "source": source,
            "field": field,
            "matched_terms": list(_item_attr(item, "matched_terms", []) or []),
            "reason": _item_attr(item, "reason", ""),
        }
        if source == "radiology":
            entry["excerpt"] = _item_attr(item, "value")
        else:
            entry["value"] = _item_attr(item, "value")
        context.append(entry)
    return context


def derive_sources(evidence_items: Sequence[Any]) -> List[str]:
    """Ordered, de-duplicated source references for traceability."""
    sources: List[str] = []
    for item in evidence_items:
        ref = "radiology_report" if _item_attr(item, "source") == "radiology" else _item_attr(item, "field")
        if ref and ref not in sources:
            sources.append(ref)
    return sources


def _build_user_payload(context: List[Dict[str, Any]], specialty: str, question: str) -> str:
    return json.dumps(
        {
            "specialty": specialty,
            "clinical_question": question,
            "retrieved_evidence": context,
            "instructions": (
                "Summarize ONLY the retrieved_evidence above in 3-5 sentences. "
                "Distinguish structured fields from radiology-report statements. "
                "Preserve exact numeric values and 0/1 field values without "
                "reinterpreting them. Cite sources inline with [field] or "
                "[radiology_report]."
            ),
        },
        indent=2,
    )


# --- Default provider: OpenAI-compatible chat completions via stdlib only. ---
def _openai_config() -> Optional[Dict[str, str]]:
    """Return provider config from the environment, or None if no key is set."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    return {
        "api_key": api_key,
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    }


def _make_env_provider() -> Optional[Provider]:
    cfg = _openai_config()
    if cfg is None:
        return None

    def provider(system_prompt: str, user_payload: str) -> str:
        body = json.dumps(
            {
                "model": cfg["model"],
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_payload},
                ],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{cfg['base_url']}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg['api_key']}",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()

    return provider


def active_model_name() -> Optional[str]:
    cfg = _openai_config()
    return cfg["model"] if cfg else None


def generate_summary(
    evidence_items: Sequence[Any],
    specialty: str,
    question: str,
    *,
    provider: Optional[Provider] = None,
) -> Dict[str, Any]:
    """Generate an evidence-grounded summary from retrieved evidence only.

    Returns a dict with a ``status`` describing the outcome:

    - ``"empty"``       — no evidence was retrieved; the model is NOT called.
    - ``"unavailable"`` — no LLM API key configured; retrieval still works.
    - ``"error"``       — the model call failed.
    - ``"ok"``          — ``summary`` holds the model's grounded text.

    The returned ``evidence_context`` is exactly what was (or would be) sent to
    the model, for auditing.
    """
    context = build_evidence_context(evidence_items)
    sources = derive_sources(evidence_items)

    if not evidence_items:
        return {
            "status": "empty",
            "summary": None,
            "sources": [],
            "evidence_context": [],
            "message": (
                "No relevant evidence was retrieved for this question, so an AI "
                "summary was not generated."
            ),
        }

    chosen = provider if provider is not None else _make_env_provider()
    if chosen is None:
        return {
            "status": "unavailable",
            "summary": None,
            "sources": sources,
            "evidence_context": context,
            "message": "AI summary unavailable: no LLM API key configured.",
        }

    user_payload = _build_user_payload(context, specialty, question)
    try:
        text = chosen(SYSTEM_PROMPT, user_payload)
    except urllib.error.URLError as exc:  # network/endpoint failure
        return {
            "status": "error",
            "summary": None,
            "sources": sources,
            "evidence_context": context,
            "message": f"AI summary unavailable: model request failed ({exc}).",
        }
    except Exception as exc:  # any provider/parsing failure — never fabricate
        return {
            "status": "error",
            "summary": None,
            "sources": sources,
            "evidence_context": context,
            "message": f"AI summary unavailable: {exc}",
        }

    return {
        "status": "ok",
        "summary": text,
        "sources": sources,
        "evidence_context": context,
        "model": active_model_name(),
        "message": "",
    }
