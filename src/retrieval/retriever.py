"""Transparent lexical clinical-evidence retrieval baseline.

Given a real ``study_id``, a ``specialty``, and a free-text clinical question,
this module retrieves the pieces of evidence already present in that patient's
record (structured triage/comorbidity fields, the radiology report, and
optionally image metadata) that lexically match the question.

Design goals (this is a *baseline*, intentionally simple):

* No LLM, no embeddings, no vector DB, no external API, no new ML model.
* Purely lexical, deterministic, and fully explainable: every returned
  :class:`EvidenceItem` carries the exact query terms it matched and a
  human-readable reason.
* It never diagnoses. A score means only "how strongly this evidence matched
  the retrieval query" — not clinical probability. A structured value of ``0``
  is returned verbatim and is never interpreted as "the patient does not have
  X".

Label-leakage guarantee:
    The pneumonia label and outcome columns (``Pneumonia``, ``outcome_all_pne``,
    ``outcome_bac_pne``, ``outcome_viral_pne``) must never participate in
    relevance. The loader already excludes them from
    ``ClinicalRecord.clinical_evidence``, and this module only ever reads
    ``clinical_evidence``, ``raw_report``, and image metadata. As an additional
    guard, :data:`FORBIDDEN_LABEL_FIELDS` is checked against every field the
    concept dictionary maps to (see :func:`_validate_concept_mappings`, run at
    import time), and the scorer asserts it is never handed a forbidden field.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, List, Tuple

from .data_loader import ClinicalRecord, get_record

# --- Label / outcome columns that must never influence retrieval. ------------
FORBIDDEN_LABEL_FIELDS = frozenset(
    {"Pneumonia", "outcome_all_pne", "outcome_bac_pne", "outcome_viral_pne"}
)


# --- Concept dictionary ------------------------------------------------------
# Each concept is a small, explicit retrieval vocabulary mapped to the fields
# that ACTUALLY exist in this dataset. This is only a retrieval vocabulary; a
# term matching does NOT mean the patient has the condition.
#
# For every concept:
#   query_terms  : terms that, if present in the question, trigger the concept.
#   fields       : structured clinical_evidence field(s) to return as evidence.
#   report_terms : terms to search for (as free text) inside the raw report.
#
# ``fields`` only ever references the 15 non-label clinical fields. No concept
# maps to a label/outcome column (enforced by _validate_concept_mappings()).
CONCEPTS: Dict[str, Dict[str, List[str]]] = {
    "heart rate": {
        "query_terms": ["heart rate", "hr", "pulse", "tachycardia", "bradycardia"],
        "fields": ["triage_heartrate"],
        "report_terms": ["heart rate", "pulse", "tachycardia", "bradycardia"],
    },
    "blood pressure": {
        "query_terms": [
            "blood pressure", "bp", "systolic", "diastolic",
            "hypertension", "hypotension",
        ],
        "fields": ["triage_sbp", "triage_dbp"],
        "report_terms": ["blood pressure", "hypertension", "systolic", "diastolic"],
    },
    "heart failure": {
        "query_terms": ["chf", "heart failure", "congestive"],
        "fields": ["cci_CHF"],
        "report_terms": [
            "chf", "heart failure", "cardiac", "cardiomegaly",
            "pulmonary edema",
        ],
    },
    "respiratory rate": {
        "query_terms": ["respiratory rate", "resp rate", "respiration", "breathing rate"],
        "fields": ["triage_resprate"],
        "report_terms": ["respiratory rate", "tachypnea"],
    },
    "oxygen": {
        "query_terms": [
            "oxygen", "o2", "o2sat", "oxygen saturation", "spo2",
            "hypoxia", "saturation",
        ],
        "fields": ["triage_o2sat"],
        "report_terms": ["oxygen", "hypoxia", "hypoxemia"],
    },
    "shortness of breath": {
        "query_terms": ["shortness of breath", "sob", "dyspnea", "breathless"],
        "fields": ["chiefcom_shortness_of_breath"],
        "report_terms": ["shortness of breath", "dyspnea"],
    },
    "cough": {
        "query_terms": ["cough"],
        "fields": ["chiefcom_cough"],
        "report_terms": ["cough"],
    },
    "fever": {
        "query_terms": ["fever", "chills", "febrile", "temperature", "pyrexia"],
        "fields": ["chiefcom_fever_chills", "triage_temperature"],
        "report_terms": ["fever", "febrile"],
    },
    "pulmonary": {
        "query_terms": ["pulmonary", "lung", "lungs"],
        "fields": ["cci_Pulmonary"],
        "report_terms": ["pulmonary", "lung", "lungs", "atelectasis"],
    },
    # "pneumonia" is a report-text retrieval vocabulary only. It intentionally
    # maps to NO structured field: the only structured pneumonia signal in this
    # dataset is the label column, which is off-limits. Searching the free-text
    # report for these words is fine — the report is real evidence, not a label.
    "pneumonia": {
        "query_terms": ["pneumonia", "consolidation", "infiltrate", "opacity", "infection"],
        "fields": [],
        "report_terms": ["pneumonia", "consolidation", "infiltrate", "opacity", "infection"],
    },
    "comorbidity": {
        "query_terms": ["comorbidity", "comorbidities", "cci", "charlson", "comorbidity index"],
        "fields": ["score_CCI"],
        "report_terms": [],
    },
    "acuity": {
        "query_terms": ["acuity", "triage acuity", "severity"],
        "fields": ["triage_acuity"],
        "report_terms": [],
    },
    "age": {
        "query_terms": ["age", "how old", "years old", "elderly"],
        "fields": ["age"],
        "report_terms": [],
    },
    "gender": {
        "query_terms": ["gender", "sex"],
        "fields": ["gender"],
        "report_terms": [],
    },
    # Optional image metadata (never image analysis). Only triggered by explicit
    # imaging words so it does not fire spuriously.
    "imaging": {
        "query_terms": [
            "x-ray", "xray", "radiograph", "radiography", "image", "imaging",
            "view position", "projection", "chest film",
        ],
        "fields": ["__image_metadata__"],
        "report_terms": [],
    },
}


# --- Specialties -------------------------------------------------------------
# A specialty scopes which concepts are eligible and provides "umbrella" terms.
# If an umbrella term appears in the question, every in-scope concept is
# activated (so "what cardiovascular information is available?" surfaces the
# whole cardiovascular panel). Specific concept terms in the question activate
# individual concepts regardless. "general" has no umbrella term, so it only
# returns concepts whose specific terms actually appear in the question.
_ALL_CONCEPTS = list(CONCEPTS.keys())

SPECIALTIES: Dict[str, Dict[str, List[str]]] = {
    "cardiology": {
        "umbrella_terms": ["cardiovascular", "cardiac", "cardiology", "heart", "circulatory"],
        "concepts": ["heart rate", "blood pressure", "heart failure"],
    },
    "pulmonology": {
        "umbrella_terms": [
            "respiratory", "pulmonary", "pulmonology", "breathing",
            "lung", "lungs", "chest", "pneumonia",
        ],
        "concepts": [
            "respiratory rate", "oxygen", "shortness of breath",
            "cough", "fever", "pulmonary", "pneumonia",
        ],
    },
    "general": {
        "umbrella_terms": [],
        "concepts": _ALL_CONCEPTS,
    },
}

# Specialty name aliases so callers can pass friendly names.
_SPECIALTY_ALIASES = {
    "cardiology": "cardiology",
    "cardiac": "cardiology",
    "cardio": "cardiology",
    "pulmonology": "pulmonology",
    "pulmonary": "pulmonology",
    "respiratory": "pulmonology",
    "pulm": "pulmonology",
    "general": "general",
    "": "general",
}


# --- Scoring weights (deterministic, documented). ----------------------------
# The score reflects ONLY lexical match strength between the question and the
# evidence — never clinical probability.
_W_PHRASE = 3.0   # a multi-word concept term matched verbatim (e.g. "heart rate")
_W_TOKEN = 2.0    # a single-word concept term matched (e.g. "cough")
_W_UMBRELLA = 1.5  # concept activated only via a specialty umbrella term
_W_REPEAT = 0.5   # small bonus per additional occurrence of a matched term
_REPEAT_CAP = 3   # cap the repeat bonus so it never dominates


class RetrievalError(Exception):
    """Raised for invalid retrieval requests (e.g. an unknown specialty)."""


@dataclass(frozen=True)
class EvidenceItem:
    """One retrieved piece of evidence with a full explanation.

    ``score`` means only how strongly this evidence matched the query. It is
    NOT a diagnosis/clinical confidence. ``value`` is the real value from the
    record, returned verbatim.
    """

    source: str  # "clinical", "radiology", or "image_metadata"
    field: str
    value: Any
    matched_terms: List[str] = dataclass_field(default_factory=list)
    score: float = 0.0
    reason: str = ""


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace for matching (non-destructive copy)."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def _term_occurrences(term: str, normalized_text: str) -> int:
    """Count word-boundary occurrences of ``term`` in already-normalized text."""
    pattern = r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)"
    return len(re.findall(pattern, normalized_text))


def _matched_query_terms(terms: List[str], normalized_question: str) -> List[Tuple[str, int]]:
    """Return (term, occurrences) for each vocabulary term present in the question."""
    found = []
    for term in terms:
        n = _term_occurrences(term, normalized_question)
        if n > 0:
            found.append((term, n))
    return found


def _score_from_terms(matched: List[Tuple[str, int]], umbrella_only: bool) -> float:
    """Deterministic score from the matched terms.

    - multi-word (phrase) term match: ``_W_PHRASE``
    - single-word (token) term match: ``_W_TOKEN``
    - repeated occurrences: ``_W_REPEAT`` each, capped
    - if the concept was activated only by a specialty umbrella term (no
      specific concept term in the question): ``_W_UMBRELLA``
    """
    if umbrella_only:
        return _W_UMBRELLA
    score = 0.0
    for term, count in matched:
        score += _W_PHRASE if (" " in term or "-" in term) else _W_TOKEN
        score += _W_REPEAT * min(max(count - 1, 0), _REPEAT_CAP)
    return score


def _split_sentences(report: str) -> List[str]:
    """Split a raw report into trimmed candidate sentences.

    MIMIC reports hard-wrap sentences across many lines, so newlines are NOT
    sentence boundaries. We first flatten all whitespace to single spaces, then
    split on sentence-ending punctuation (``.`` or ``:`` followed by
    whitespace). Whitespace is collapsed for readability, but no words are
    altered or removed, and the source ``raw_report`` on the record is never
    modified.
    """
    flat = re.sub(r"\s+", " ", report).strip()
    rough = re.split(r"(?<=[.:])\s+", flat)
    return [frag.strip() for frag in rough if frag.strip()]


def _search_report(
    report_terms: List[str], sentences: List[str]
) -> List[Tuple[str, List[str], float]]:
    """Find report sentences containing any ``report_terms``.

    Returns a list of (excerpt, matched_report_terms, score). Each matching
    sentence becomes at most one excerpt so the full report is not duplicated.
    """
    results = []
    for sentence in sentences:
        norm = _normalize(sentence)
        hits = _matched_query_terms(report_terms, norm)
        if hits:
            score = _score_from_terms(hits, umbrella_only=False)
            matched = [t for t, _ in hits]
            results.append((sentence, matched, score))
    return results


def _score_structured_field(field_name: str, value: Any, terms: List[str], umbrella_only: bool) -> float:
    """Score a structured field, guarding against label leakage.

    An unavailable value scores 0. This function asserts it is never handed a
    forbidden label field.
    """
    assert field_name not in FORBIDDEN_LABEL_FIELDS, (
        f"Label field '{field_name}' must never be scored by retrieval."
    )
    if value is None:
        return 0.0
    # Treat NaN as unavailable without importing numpy/pandas here.
    if isinstance(value, float) and value != value:
        return 0.0
    return _score_from_terms(terms, umbrella_only)


def _normalize_specialty(specialty: str) -> str:
    key = _normalize(specialty)
    if key in _SPECIALTY_ALIASES:
        return _SPECIALTY_ALIASES[key]
    raise RetrievalError(
        f"Unknown specialty '{specialty}'. Supported: Cardiology, Pulmonology, General."
    )


def _excerpt(sentence: str, max_len: int = 240) -> str:
    """Return a short excerpt, truncating very long sentences."""
    return sentence if len(sentence) <= max_len else sentence[: max_len - 1].rstrip() + "…"


def retrieve_evidence(
    study_id: int,
    specialty: str,
    question: str,
    top_k: int = 10,
) -> List[EvidenceItem]:
    """Retrieve the evidence in a real record most relevant to ``question``.

    Parameters
    ----------
    study_id:
        A real cohort study id.
    specialty:
        One of ``Cardiology``, ``Pulmonology``, ``General`` (case-insensitive;
        a few aliases are accepted). Scopes which concepts are eligible.
    question:
        Free-text clinical question. Only its lexical content is used.
    top_k:
        Maximum number of ranked evidence items to return.

    Returns
    -------
    list[EvidenceItem]
        Ranked (highest match first). Empty if nothing matched — the system
        never invents evidence for concepts absent from the question.
    """
    specialty_key = _normalize_specialty(specialty)
    spec = SPECIALTIES[specialty_key]
    normalized_question = _normalize(question)

    umbrella_hits = _matched_query_terms(spec["umbrella_terms"], normalized_question)
    umbrella_terms_found = [t for t, _ in umbrella_hits]
    umbrella_active = bool(umbrella_hits)

    record: ClinicalRecord = get_record(study_id)
    sentences = _split_sentences(record.raw_report)

    items: List[EvidenceItem] = []

    for concept_name in spec["concepts"]:
        concept = CONCEPTS[concept_name]

        specific_hits = _matched_query_terms(concept["query_terms"], normalized_question)
        triggered_by_specific = bool(specific_hits)

        # A concept is retrieved if a specific concept term is present, or if a
        # specialty umbrella term activates the whole in-scope panel.
        if not triggered_by_specific and not umbrella_active:
            continue

        umbrella_only = not triggered_by_specific
        if triggered_by_specific:
            matched_terms = [t for t, _ in specific_hits]
        else:
            matched_terms = list(umbrella_terms_found)

        # --- Structured clinical evidence -------------------------------------
        for field_name in concept["fields"]:
            if field_name == "__image_metadata__":
                # Optional image metadata (not image analysis).
                for meta_field, meta_value in (
                    ("view_position", record.view_position),
                    ("study_date", record.study_date),
                    ("image_path", record.image_path),
                ):
                    score = _score_from_terms(specific_hits, umbrella_only)
                    items.append(
                        EvidenceItem(
                            source="image_metadata",
                            field=meta_field,
                            value=meta_value,
                            matched_terms=matched_terms,
                            score=score,
                            reason=(
                                f"Retrieved because the question matched the "
                                f"'{concept_name}' concept; returning image metadata "
                                f"'{meta_field}' (metadata only, the image was not analyzed)."
                            ),
                        )
                    )
                continue

            value = record.clinical_evidence[field_name]
            score = _score_structured_field(field_name, value, specific_hits, umbrella_only)
            items.append(
                EvidenceItem(
                    source="clinical",
                    field=field_name,
                    value=value,
                    matched_terms=matched_terms,
                    score=score,
                    reason=(
                        f"Retrieved because the question matched the '{concept_name}' "
                        f"concept, which maps to the '{field_name}' field. The value is "
                        f"reported as-is and is not interpreted as a diagnosis."
                    ),
                )
            )

        # --- Radiology report evidence ----------------------------------------
        report_hits = _search_report(concept["report_terms"], sentences)
        for excerpt, report_matched, report_score in report_hits:
            items.append(
                EvidenceItem(
                    source="radiology",
                    field="raw_report",
                    value=_excerpt(excerpt),
                    matched_terms=sorted(set(matched_terms) | set(report_matched)),
                    score=report_score,
                    reason=(
                        f"Retrieved because the '{concept_name}' concept matched the "
                        f"report term(s) {report_matched} in this sentence. The excerpt "
                        f"is quoted verbatim from the real report; it is not a diagnosis."
                    ),
                )
            )

    # Deduplicate identical radiology excerpts triggered by multiple concepts,
    # keeping the highest-scoring occurrence and merging matched terms.
    items = _merge_duplicate_report_items(items)

    # Deterministic ranking: score desc, then a stable key.
    source_rank = {"clinical": 0, "radiology": 1, "image_metadata": 2}
    items.sort(
        key=lambda it: (
            -it.score,
            source_rank.get(it.source, 9),
            it.field,
            str(it.value),
        )
    )
    return items[:top_k]


def _merge_duplicate_report_items(items: List[EvidenceItem]) -> List[EvidenceItem]:
    """Merge radiology items that quote the exact same excerpt."""
    merged: Dict[Tuple[str, str, str], EvidenceItem] = {}
    passthrough: List[EvidenceItem] = []
    for it in items:
        if it.source != "radiology":
            passthrough.append(it)
            continue
        key = (it.source, it.field, str(it.value))
        existing = merged.get(key)
        if existing is None:
            merged[key] = it
        else:
            merged[key] = EvidenceItem(
                source=it.source,
                field=it.field,
                value=it.value,
                matched_terms=sorted(set(existing.matched_terms) | set(it.matched_terms)),
                score=max(existing.score, it.score),
                reason=existing.reason,
            )
    return passthrough + list(merged.values())


def _validate_concept_mappings() -> None:
    """Ensure no concept maps to a forbidden label/outcome field.

    Runs at import time so a mistake is caught immediately rather than leaking
    a label into retrieval.
    """
    for concept_name, concept in CONCEPTS.items():
        for field_name in concept["fields"]:
            if field_name in FORBIDDEN_LABEL_FIELDS:
                raise RetrievalError(
                    f"Concept '{concept_name}' maps to forbidden label field "
                    f"'{field_name}'. Labels must never be used for retrieval."
                )


_validate_concept_mappings()


# --- Pretty-printing + demo --------------------------------------------------
def format_results(items: List[EvidenceItem]) -> str:
    """Render ranked evidence for display."""
    if not items:
        return "  (no matching evidence — nothing was invented)"
    lines = []
    for i, it in enumerate(items, 1):
        value = it.value
        label = "Excerpt" if it.source == "radiology" else "Value"
        lines.append(
            f"{i}. Source: {it.source}\n"
            f"   Field: {it.field}\n"
            f"   {label}: {value!r}\n"
            f"   Score: {it.score}\n"
            f"   Matched terms: {it.matched_terms}\n"
            f"   Reason: {it.reason}"
        )
    return "\n".join(lines)


def run_validation() -> None:
    """Verify the Day 2 baseline end-to-end, including a label-leakage spy.

    Proves, at runtime and against the real data, the eight validation points:
    the Day 1 loader still works, all 1,601 records load, study 50543252 is
    retrievable, Cardiology/Pulmonology queries surface the right evidence, an
    unrelated kidney question invents nothing, and no label/outcome column is
    ever read by scoring.
    """
    from .data_loader import CLINICAL_FIELDS, load_records

    print("#" * 78)
    print("VALIDATION")
    print("#" * 78)

    # 1 & 2. Day 1 loader still works; all 1,601 records load.
    records = load_records()
    assert len(records) == 1601, f"expected 1601 records, got {len(records)}"
    print(f"[1,2] Day 1 loader OK — {len(records)} records loaded.")

    # 3. study_id 50543252 still retrievable.
    rec = get_record(50543252)
    assert rec.study_id == 50543252 and rec.subject_id == 10003019
    print("[3]   get_record(50543252) OK — subject 10003019.")

    # 4. Cardiology returns cardiovascular structured evidence.
    card = retrieve_evidence(50543252, "Cardiology",
                             "What cardiovascular information is available?")
    card_fields = {it.field for it in card if it.source == "clinical"}
    assert {"triage_heartrate", "triage_sbp", "cci_CHF"} <= card_fields, card_fields
    print(f"[4]   Cardiology surfaced cardiovascular fields: {sorted(card_fields)}")

    # 5. Pulmonology returns respiratory evidence.
    pulm = retrieve_evidence(50543252, "Pulmonology",
                             "What respiratory information is available?")
    assert any(it.field == "cci_Pulmonary" for it in pulm)
    assert any(it.source == "radiology" for it in pulm)
    print(f"[5]   Pulmonology surfaced {len(pulm)} items incl. cci_Pulmonary + report.")

    # 6. Unrelated kidney question invents nothing.
    kidney = retrieve_evidence(50543252, "General",
                               "What information about kidney function is available?")
    assert kidney == [], f"kidney query should be empty, got {len(kidney)} items"
    print("[6]   Kidney question returned 0 items (nothing invented).")

    # 7. Label-leakage spy: wrap the scorer and record every field it sees, then
    #    run a broad sweep and assert no forbidden label field ever appears.
    import src.retrieval.retriever as this_module

    seen_fields: List[str] = []
    original = this_module._score_structured_field

    def spy(field_name, value, terms, umbrella_only):
        seen_fields.append(field_name)
        return original(field_name, value, terms, umbrella_only)

    this_module._score_structured_field = spy
    try:
        sweep_questions = [
            "heart rate blood pressure heart failure chf",
            "respiratory oxygen cough fever shortness of breath pneumonia",
            "pneumonia consolidation infiltrate opacity infection outcome",
            "comorbidity acuity age gender x-ray view",
        ]
        for sp in ("Cardiology", "Pulmonology", "General"):
            for q in sweep_questions:
                retrieve_evidence(50543252, sp, q, top_k=50)
    finally:
        this_module._score_structured_field = original

    leaked = FORBIDDEN_LABEL_FIELDS & set(seen_fields)
    assert not leaked, f"LABEL LEAKAGE: scorer received {leaked}"
    # Labels are also absent from the record's clinical evidence and from every
    # concept field mapping.
    assert not (FORBIDDEN_LABEL_FIELDS & set(rec.clinical_evidence.keys()))
    assert not (FORBIDDEN_LABEL_FIELDS & set(CLINICAL_FIELDS))
    mapped_fields = {f for c in CONCEPTS.values() for f in c["fields"]}
    assert not (FORBIDDEN_LABEL_FIELDS & mapped_fields)
    print(f"[7]   No label leakage — scorer saw {len(set(seen_fields))} distinct "
          f"fields, none in {sorted(FORBIDDEN_LABEL_FIELDS)}.")

    print("[8]   (File isolation is verified separately via git.)")
    print("\nALL VALIDATION CHECKS PASSED.\n")


def _demo() -> None:
    def run(title, study_id, specialty, question, top_k=6):
        print("=" * 78)
        print(title)
        print(f"study_id={study_id}  specialty={specialty!r}")
        print(f"question={question!r}")
        print("-" * 78)
        results = retrieve_evidence(study_id, specialty, question, top_k=top_k)
        print(format_results(results))
        print()

    run(
        "TEST 1 — Cardiology",
        50543252, "Cardiology",
        "What cardiovascular information is available for this patient?",
    )
    run(
        "TEST 2 — Pulmonology",
        50543252, "Pulmonology",
        "What respiratory information is available?",
    )
    run(
        "TEST 3 — Different real study (Cardiology)",
        52858944, "Cardiology",
        "Any heart failure or heart rate information for this patient?",
    )
    run(
        "TEST 4 — Unrelated kidney question (should be empty)",
        50543252, "General",
        "What information about kidney function is available?",
    )


if __name__ == "__main__":
    _demo()
    run_validation()
