"""Data loader for the retrieval prototype.

Loads the existing cohort CSV and the existing radiology-report CSV, joins them
on ``study_id`` (verifying ``subject_id`` also agrees), resolves the on-disk
image path for every row, and produces one normalized :class:`ClinicalRecord`
per cohort row.

Design notes / constraints honored here:

* Only the existing real data is read. Nothing is synthesized, rewritten, or
  silently repaired.
* Clinical fields keep their original numeric/binary dataset values. No
  natural-language interpretation is applied.
* ``raw_report`` is used verbatim as the primary report text.
* ``pneumonia_label`` is retained for later evaluation only; it is not exposed
  through, or consumed by, any relevance calculation. The loader computes no
  relevance at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# --- Paths (resolved relative to the repository root, not the caller's CWD) ---
# This file lives at <repo>/src/retrieval/data_loader.py, so the repo root is
# three levels up.
_REPO_ROOT = Path(__file__).resolve().parents[2]

COHORT_CSV = _REPO_ROOT / "data" / "mimic_ed_cxr_pneumonia_multimodal_cohort.csv"
REPORTS_CSV = _REPO_ROOT / "results" / "mimic_cxr_reports_cohort.csv"
IMAGE_ROOT = _REPO_ROOT / "data" / "files"

# Every cohort row and every report row is expected exactly once.
EXPECTED_ROWS = 1601

# The 15 clinical-evidence fields, kept exactly as encoded in the dataset.
CLINICAL_FIELDS: List[str] = [
    "age",
    "gender",
    "triage_temperature",
    "triage_heartrate",
    "triage_resprate",
    "triage_o2sat",
    "triage_sbp",
    "triage_dbp",
    "triage_acuity",
    "chiefcom_shortness_of_breath",
    "chiefcom_cough",
    "chiefcom_fever_chills",
    "cci_Pulmonary",
    "cci_CHF",
    "score_CCI",
]

# Identifier / metadata / report / label columns the loader also relies on.
_IDENTIFIER_FIELDS = ["subject_id", "stay_id", "study_id", "dicom_id"]
_IMAGE_META_FIELDS = ["StudyDate", "ViewPosition"]
_LABEL_FIELD = "Pneumonia"


class DataValidationError(Exception):
    """Raised when the loaded data fails an integrity check.

    The message always states exactly which check failed so the caller does not
    have to guess. The loader never silently repairs the data.
    """


@dataclass(frozen=True)
class ClinicalRecord:
    """One normalized cohort row.

    ``clinical_evidence`` holds the 15 raw dataset values keyed by column name.
    ``pneumonia_label`` is retained for later evaluation only and must not be
    used as a retrieval input.
    """

    subject_id: int
    stay_id: int
    study_id: int
    dicom_id: str
    study_date: Any
    view_position: str
    clinical_evidence: Dict[str, Any]
    raw_report: str
    image_path: str
    pneumonia_label: Any


def _build_image_path(subject_id: int, study_id: int, dicom_id: str) -> Path:
    """Construct the MIMIC-CXR-JPG path for a row.

    Layout: ``data/files/p{subject_id[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg``
    where ``{subject_id[:2]}`` is the first two digits of ``subject_id``.
    """
    prefix = str(subject_id)[:2]
    return (
        IMAGE_ROOT
        / f"p{prefix}"
        / f"p{subject_id}"
        / f"s{study_id}"
        / f"{dicom_id}.jpg"
    )


def _require_columns(df: pd.DataFrame, columns: List[str], source: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"{source} is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def _validate_frames(cohort: pd.DataFrame, reports: pd.DataFrame) -> None:
    """Run every integrity check, raising a clear error on the first failure."""

    # Required columns present in each source.
    _require_columns(
        cohort,
        _IDENTIFIER_FIELDS + _IMAGE_META_FIELDS + [_LABEL_FIELD] + CLINICAL_FIELDS,
        "Cohort CSV",
    )
    _require_columns(
        reports, ["subject_id", "study_id", "raw_report"], "Reports CSV"
    )

    # Row counts.
    if len(cohort) != EXPECTED_ROWS:
        raise DataValidationError(
            f"Cohort CSV has {len(cohort)} rows; expected {EXPECTED_ROWS}."
        )
    if len(reports) != EXPECTED_ROWS:
        raise DataValidationError(
            f"Reports CSV has {len(reports)} rows; expected {EXPECTED_ROWS}."
        )

    # study_id uniqueness in both sources.
    if cohort["study_id"].duplicated().any():
        dups = cohort.loc[cohort["study_id"].duplicated(), "study_id"].tolist()
        raise DataValidationError(
            f"Cohort CSV has duplicate study_id value(s): {dups[:10]}"
        )
    if reports["study_id"].duplicated().any():
        dups = reports.loc[reports["study_id"].duplicated(), "study_id"].tolist()
        raise DataValidationError(
            f"Reports CSV has duplicate study_id value(s): {dups[:10]}"
        )

    # Every cohort study_id has a matching report.
    report_ids = set(reports["study_id"])
    missing_reports = [s for s in cohort["study_id"] if s not in report_ids]
    if missing_reports:
        raise DataValidationError(
            f"{len(missing_reports)} cohort study_id(s) have no matching report, "
            f"e.g. {missing_reports[:10]}"
        )

    # subject_id agrees between cohort and report after joining on study_id.
    rep_subject = reports.set_index("study_id")["subject_id"]
    for study_id, subject_id in zip(cohort["study_id"], cohort["subject_id"]):
        if int(rep_subject.loc[study_id]) != int(subject_id):
            raise DataValidationError(
                f"subject_id mismatch for study_id {study_id}: "
                f"cohort={subject_id}, report={int(rep_subject.loc[study_id])}"
            )

    # No required clinical fields missing.
    for field in CLINICAL_FIELDS:
        n_missing = int(cohort[field].isna().sum())
        if n_missing:
            raise DataValidationError(
                f"Clinical field '{field}' has {n_missing} missing value(s)."
            )

    # raw_report present and non-empty for every row.
    rep_indexed = reports.set_index("study_id")
    empty_reports = []
    for study_id in cohort["study_id"]:
        text = rep_indexed.loc[study_id, "raw_report"]
        if not isinstance(text, str) or text.strip() == "":
            empty_reports.append(int(study_id))
    if empty_reports:
        raise DataValidationError(
            f"{len(empty_reports)} study_id(s) have an empty raw_report, "
            f"e.g. {empty_reports[:10]}"
        )


def _build_records(cohort: pd.DataFrame, reports: pd.DataFrame) -> List[ClinicalRecord]:
    """Join the frames and build one record per cohort row, verifying images."""

    reports_by_study = reports.set_index("study_id")

    records: List[ClinicalRecord] = []
    missing_images: List[str] = []

    for row in cohort.itertuples(index=False):
        subject_id = int(getattr(row, "subject_id"))
        study_id = int(getattr(row, "study_id"))
        dicom_id = str(getattr(row, "dicom_id"))

        image_path = _build_image_path(subject_id, study_id, dicom_id)
        if not image_path.exists():
            missing_images.append(str(image_path))
            continue

        clinical_evidence = {
            field: getattr(row, field) for field in CLINICAL_FIELDS
        }

        raw_report = str(reports_by_study.loc[study_id, "raw_report"])

        records.append(
            ClinicalRecord(
                subject_id=subject_id,
                stay_id=int(getattr(row, "stay_id")),
                study_id=study_id,
                dicom_id=dicom_id,
                study_date=getattr(row, "StudyDate"),
                view_position=str(getattr(row, "ViewPosition")),
                clinical_evidence=clinical_evidence,
                raw_report=raw_report,
                image_path=str(image_path),
                pneumonia_label=getattr(row, _LABEL_FIELD),
            )
        )

    if missing_images:
        raise DataValidationError(
            f"{len(missing_images)} image file(s) do not exist on disk, "
            f"e.g. {missing_images[:5]}"
        )

    return records


# Module-level cache so repeated calls (e.g. get_record) do not re-read the CSVs.
_RECORDS_CACHE: List[ClinicalRecord] | None = None
_RECORDS_BY_STUDY: Dict[int, ClinicalRecord] | None = None


def load_records(*, validate: bool = True, use_cache: bool = True) -> List[ClinicalRecord]:
    """Load and return all normalized :class:`ClinicalRecord` objects.

    Parameters
    ----------
    validate:
        When ``True`` (default) every integrity check runs before records are
        built. A failure raises :class:`DataValidationError` describing exactly
        what failed.
    use_cache:
        When ``True`` (default) results are cached for the process so repeated
        calls are cheap.
    """
    global _RECORDS_CACHE, _RECORDS_BY_STUDY

    if use_cache and _RECORDS_CACHE is not None:
        return _RECORDS_CACHE

    if not COHORT_CSV.exists():
        raise DataValidationError(f"Cohort CSV not found at: {COHORT_CSV}")
    if not REPORTS_CSV.exists():
        raise DataValidationError(f"Reports CSV not found at: {REPORTS_CSV}")

    cohort = pd.read_csv(COHORT_CSV)
    reports = pd.read_csv(REPORTS_CSV)

    if validate:
        _validate_frames(cohort, reports)

    records = _build_records(cohort, reports)

    if use_cache:
        _RECORDS_CACHE = records
        _RECORDS_BY_STUDY = {r.study_id: r for r in records}

    return records


def get_record(study_id: int) -> ClinicalRecord:
    """Return the real record for ``study_id``.

    Loads (and caches) all records on first use. Raises :class:`KeyError` if the
    study is not part of the cohort.
    """
    global _RECORDS_BY_STUDY

    if _RECORDS_BY_STUDY is None:
        load_records()

    assert _RECORDS_BY_STUDY is not None  # populated by load_records()
    key = int(study_id)
    if key not in _RECORDS_BY_STUDY:
        raise KeyError(f"study_id {key} is not present in the cohort.")
    return _RECORDS_BY_STUDY[key]


def _demo() -> None:
    """Validate against the real repository data and print a small summary.

    Prints only aggregate counts and a single study's metadata (never the full
    report body or the full dataset), matching the prototype's validation spec.
    """
    records = load_records()

    n_reports_matched = sum(1 for r in records if r.raw_report.strip())
    n_images_found = sum(1 for r in records if Path(r.image_path).exists())

    print(f"Records loaded: {len(records)}")
    print(f"Reports matched: {n_reports_matched}")
    print(f"Images found: {n_images_found}")

    demo_study_id = 50543252
    rec = get_record(demo_study_id)
    print("\n--- Retrieved record for study_id", demo_study_id, "---")
    print("subject_id     :", rec.subject_id)
    print("study_id       :", rec.study_id)
    print("image_path     :", rec.image_path)
    print("StudyDate      :", rec.study_date)
    print("ViewPosition   :", rec.view_position)
    print("clinical keys  :", list(rec.clinical_evidence.keys()))
    print("report length  :", len(rec.raw_report))
    print("pneumonia_label:", rec.pneumonia_label)


if __name__ == "__main__":
    _demo()
