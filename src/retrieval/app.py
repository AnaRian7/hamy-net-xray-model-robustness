"""Minimal local web UI for the transparent retrieval baseline.

A tiny, dependency-free viewer for the Day 2 retriever. It is built entirely on
Python's standard-library ``http.server`` — no FastAPI, Flask, database, vector
store, LLM, or new model. It only reads the existing real cohort through
``retrieve_evidence`` / ``get_record`` and serves the patient's real chest X-ray
from disk.

Run it with::

    python3 -m src.retrieval.app        # serves http://127.0.0.1:8000

Routes
------
* ``GET /``                     the single-page UI (form + results area).
* ``GET /api/studies``          the list of real study ids (for the dropdown).
* ``GET /api/retrieve``         JSON: study context + ranked evidence.
* ``GET /image?study_id=...``   the real chest X-ray JPEG for a study.

Non-diagnostic by design: the UI shows why each piece of evidence was retrieved
(matched terms + activation) and never presents a diagnosis. The pneumonia
label is never read or displayed.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .data_loader import IMAGE_ROOT, get_record, load_records
from .retriever import EvidenceItem, retrieve_evidence

HOST = "127.0.0.1"
PORT = int(os.environ.get("RETRIEVAL_UI_PORT", "8000"))

# Human-friendly labels for structured fields (presentation only). These never
# reinterpret the value — a 0 is still shown as 0.
FIELD_LABELS = {
    "triage_heartrate": "Heart rate",
    "triage_sbp": "Systolic BP",
    "triage_dbp": "Diastolic BP",
    "triage_o2sat": "Oxygen saturation",
    "triage_resprate": "Respiratory rate",
    "triage_temperature": "Temperature",
    "triage_acuity": "Triage acuity",
    "chiefcom_shortness_of_breath": "Chief complaint: shortness of breath",
    "chiefcom_cough": "Chief complaint: cough",
    "chiefcom_fever_chills": "Chief complaint: fever/chills",
    "cci_CHF": "CHF-related field (comorbidity)",
    "cci_Pulmonary": "Pulmonary comorbidity",
    "score_CCI": "Charlson comorbidity index",
    "age": "Age",
    "gender": "Gender (encoded)",
    "raw_report": "Radiology report",
    "view_position": "Image view position",
    "study_date": "Study date",
    "image_path": "Image path",
}

SPECIALTIES_UI = ["Cardiology", "Pulmonology", "General"]

# Cache the sorted list of real study ids for the dropdown.
_STUDY_IDS: list[int] | None = None


def _study_ids() -> list[int]:
    global _STUDY_IDS
    if _STUDY_IDS is None:
        _STUDY_IDS = sorted(r.study_id for r in load_records())
    return _STUDY_IDS


def _evidence_to_dict(item: EvidenceItem) -> dict:
    return {
        "source": item.source,
        "field": item.field,
        "label": FIELD_LABELS.get(item.field, item.field),
        "value": item.value,
        "matched_terms": item.matched_terms,
        "score": item.score,
        "activation": item.activation,
        "reason": item.reason,
    }


def _retrieve_payload(study_id: int, specialty: str, question: str, top_k: int) -> dict:
    record = get_record(study_id)  # pneumonia_label is intentionally NOT surfaced
    items = retrieve_evidence(study_id, specialty, question, top_k=top_k)
    return {
        "context": {
            "study_id": record.study_id,
            "stay_id": record.stay_id,
            "study_date": record.study_date,
            "view_position": record.view_position,
        },
        "image_url": f"/image?study_id={record.study_id}",
        "evidence": [_evidence_to_dict(it) for it in items],
        "count": len(items),
    }


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Clinical Evidence Retrieval</title>
<style>
  :root { --bg:#0f172a; --panel:#ffffff; --muted:#64748b; --line:#e2e8f0;
          --accent:#2563eb; --accent-d:#1d4ed8; --chip:#eff6ff; --chipbd:#bfdbfe; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
         background:linear-gradient(180deg,#f1f5f9,#e2e8f0); color:#0f172a; }
  header { background:var(--bg); color:#fff; padding:18px 24px; }
  header h1 { margin:0; font-size:18px; font-weight:600; letter-spacing:.2px; }
  header p { margin:4px 0 0; color:#94a3b8; font-size:12.5px; }
  .wrap { max-width:1080px; margin:22px auto; padding:0 20px; display:grid;
          grid-template-columns: 1fr; gap:18px; }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:12px;
           box-shadow:0 1px 2px rgba(15,23,42,.04); padding:18px 20px; }
  .panel h2 { margin:0 0 14px; font-size:14px; text-transform:uppercase;
              letter-spacing:.6px; color:var(--muted); }
  form.grid { display:grid; grid-template-columns:180px 1fr; gap:12px 16px; align-items:center; }
  label { font-size:13px; color:#334155; font-weight:500; }
  select, input[type=text] { width:100%; padding:9px 11px; border:1px solid var(--line);
           border-radius:8px; font-size:14px; background:#fff; }
  .actions { grid-column:1 / -1; display:flex; justify-content:flex-end; }
  button { background:var(--accent); color:#fff; border:0; padding:10px 18px; border-radius:8px;
           font-size:14px; font-weight:600; cursor:pointer; }
  button:hover { background:var(--accent-d); }
  .cols { display:grid; grid-template-columns: 1.6fr 1fr; gap:18px; align-items:start; }
  @media (max-width:820px){ .cols{ grid-template-columns:1fr; } form.grid{ grid-template-columns:1fr; } }
  .ctx { display:grid; grid-template-columns:auto 1fr; gap:6px 16px; font-size:14px; }
  .ctx dt { color:var(--muted); }
  .ctx dd { margin:0; font-variant-numeric:tabular-nums; font-weight:600; }
  .ev { border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin-bottom:10px; }
  .ev .top { display:flex; justify-content:space-between; align-items:baseline; gap:10px; }
  .ev .name { font-weight:600; font-size:14.5px; }
  .ev .val { font-size:20px; font-variant-numeric:tabular-nums; margin:2px 0 6px; }
  .ev .excerpt { font-size:13.5px; color:#0f172a; background:#f8fafc; border-left:3px solid var(--chipbd);
                 padding:8px 10px; border-radius:6px; font-style:italic; margin:2px 0 8px; }
  .chips { display:flex; flex-wrap:wrap; gap:6px; }
  .chip { font-size:12px; background:var(--chip); border:1px solid var(--chipbd); color:#1e40af;
          padding:2px 8px; border-radius:999px; }
  .tag { font-size:11px; padding:2px 8px; border-radius:999px; font-weight:600; }
  .tag.direct { background:#dcfce7; color:#166534; }
  .tag.umbrella { background:#fef9c3; color:#854d0e; }
  .reason { font-size:12px; color:var(--muted); margin-top:6px; line-height:1.4; }
  .src { font-size:11px; color:#fff; background:#334155; padding:2px 7px; border-radius:6px; text-transform:capitalize; }
  .src.radiology { background:#7c3aed; } .src.image_metadata { background:#0891b2; }
  figure { margin:0; } figure img { width:100%; border-radius:10px; border:1px solid var(--line); background:#000; }
  figcaption { font-size:12px; color:var(--muted); margin-top:6px; }
  .empty { color:var(--muted); font-style:italic; padding:8px 0; }
  .disclaimer { font-size:12px; color:#92400e; background:#fffbeb; border:1px solid #fde68a;
                padding:8px 10px; border-radius:8px; margin-top:10px; }
  .hidden { display:none; }
</style>
</head>
<body>
<header>
  <h1>Clinical Evidence Retrieval</h1>
  <p>Transparent lexical baseline over the existing MIMIC cohort — retrieval only, not a diagnosis.</p>
</header>

<div class="wrap">
  <section class="panel">
    <form class="grid" id="form">
      <label for="specialty">Specialty</label>
      <select id="specialty"></select>
      <label for="study">Study</label>
      <select id="study"></select>
      <label for="question">Clinical question</label>
      <input type="text" id="question"
             value="What cardiovascular information is available for this patient?" />
      <div class="actions"><button type="submit">Retrieve Evidence</button></div>
    </form>
  </section>

  <div id="results" class="hidden">
    <section class="panel">
      <h2>Patient / Study Context</h2>
      <dl class="ctx" id="ctx"></dl>
    </section>

    <div class="cols">
      <section class="panel">
        <h2>Retrieved Evidence</h2>
        <div id="evidence"></div>
        <div class="disclaimer">Each item was retrieved because its text lexically matched the
          query terms shown. Values (including 0) are shown verbatim and are not interpreted as a diagnosis.</div>
      </section>
      <section class="panel">
        <h2>Chest X-ray</h2>
        <figure>
          <img id="cxr" alt="Chest X-ray" />
          <figcaption id="cxrcap"></figcaption>
        </figure>
      </section>
    </div>
  </div>
</div>

<script>
function esc(s){ return String(s).replace(/[&<>"']/g, c => (
  {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

async function init(){
  const spec = document.getElementById('specialty');
  %SPECIALTY_OPTIONS%.forEach(s => { const o=document.createElement('option'); o.value=s; o.textContent=s; spec.appendChild(o); });
  const study = document.getElementById('study');
  const ids = await (await fetch('/api/studies')).json();
  ids.forEach(id => { const o=document.createElement('option'); o.value=id; o.textContent=id; study.appendChild(o); });
  study.value = '50543252';
}

function renderContext(ctx){
  const dl = document.getElementById('ctx');
  dl.innerHTML =
    `<dt>Study ID</dt><dd>${esc(ctx.study_id)}</dd>` +
    `<dt>ED Stay</dt><dd>${esc(ctx.stay_id)}</dd>` +
    `<dt>Study Date</dt><dd>${esc(ctx.study_date)}</dd>` +
    `<dt>View</dt><dd>${esc(ctx.view_position)}</dd>`;
}

function renderEvidence(items){
  const box = document.getElementById('evidence');
  if(!items.length){ box.innerHTML = '<div class="empty">No matching evidence — nothing was invented.</div>'; return; }
  box.innerHTML = items.map((it, i) => {
    const chips = it.matched_terms.map(t => `<span class="chip">${esc(t)}</span>`).join('');
    const body = it.source === 'radiology'
      ? `<div class="excerpt">${esc(it.value)}</div>`
      : `<div class="val">${esc(it.value)}</div>`;
    return `<div class="ev">
      <div class="top">
        <span class="name">${i+1}. ${esc(it.label)}</span>
        <span class="src ${esc(it.source)}">${esc(it.source)}</span>
      </div>
      ${body}
      <div class="chips">
        <span class="tag ${esc(it.activation)}">${esc(it.activation)}</span>
        ${chips}
        <span class="chip" title="lexical match strength, not clinical probability">score ${esc(it.score)}</span>
      </div>
      <div class="reason">${esc(it.reason)}</div>
    </div>`;
  }).join('');
}

document.getElementById('form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const specialty = document.getElementById('specialty').value;
  const study_id = document.getElementById('study').value;
  const question = document.getElementById('question').value;
  const url = `/api/retrieve?study_id=${encodeURIComponent(study_id)}` +
              `&specialty=${encodeURIComponent(specialty)}` +
              `&question=${encodeURIComponent(question)}`;
  const data = await (await fetch(url)).json();
  document.getElementById('results').classList.remove('hidden');
  renderContext(data.context);
  renderEvidence(data.evidence);
  const img = document.getElementById('cxr');
  img.src = data.image_url;
  document.getElementById('cxrcap').textContent =
    `Study ${data.context.study_id} · ${data.context.view_position} · real MIMIC-CXR image`;
});

init();
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    server_version = "RetrievalUI/1.0"

    def log_message(self, fmt, *args):  # keep the console quiet but informative
        return

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        try:
            if path == "/":
                html = INDEX_HTML.replace("%SPECIALTY_OPTIONS%", json.dumps(SPECIALTIES_UI))
                self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
                return

            if path == "/api/studies":
                self._send_json(_study_ids())
                return

            if path == "/api/retrieve":
                study_id = int(query.get("study_id", ["0"])[0])
                specialty = query.get("specialty", ["General"])[0]
                question = query.get("question", [""])[0]
                top_k = int(query.get("top_k", ["10"])[0])
                self._send_json(_retrieve_payload(study_id, specialty, question, top_k))
                return

            if path == "/image":
                self._serve_image(query)
                return

            self._send(404, b"Not found", "text/plain; charset=utf-8")
        except KeyError as exc:
            self._send_json({"error": f"Unknown study: {exc}"}, code=404)
        except Exception as exc:  # surface errors as JSON rather than a stack trace
            self._send_json({"error": str(exc)}, code=400)

    def _serve_image(self, query) -> None:
        study_id = int(query.get("study_id", ["0"])[0])
        record = get_record(study_id)
        image_path = os.path.realpath(record.image_path)
        # Safety: only ever serve files from within the dataset image root.
        if not image_path.startswith(os.path.realpath(str(IMAGE_ROOT)) + os.sep):
            self._send(403, b"Forbidden", "text/plain; charset=utf-8")
            return
        if not os.path.exists(image_path):
            self._send(404, b"Image not found", "text/plain; charset=utf-8")
            return
        with open(image_path, "rb") as fh:
            self._send(200, fh.read(), "image/jpeg")


def serve(host: str = HOST, port: int = PORT) -> None:
    # Warm the study-id cache (also validates the data) before serving.
    _study_ids()
    httpd = ThreadingHTTPServer((host, port), _Handler)
    print(f"Clinical Evidence Retrieval UI running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        httpd.shutdown()


if __name__ == "__main__":
    serve()
