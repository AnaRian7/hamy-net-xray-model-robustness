"""Isolated retrieval prototype built on the existing MIMIC-ED + MIMIC-CXR cohort.

This package is intentionally separate from the pneumonia modeling/experiment
code. It only reads the existing real data (the 1,601-row cohort, the matching
radiology reports, and the on-disk chest X-ray images) and normalizes each
cohort row into a single :class:`ClinicalRecord`.

The ``Pneumonia`` value is retained on each record as ``pneumonia_label`` for
later evaluation only. It is never used to compute retrieval relevance.
"""

from .data_loader import (
    ClinicalRecord,
    CLINICAL_FIELDS,
    DataValidationError,
    get_record,
    load_records,
)
from .retriever import (
    CONCEPTS,
    EvidenceItem,
    FORBIDDEN_LABEL_FIELDS,
    RetrievalError,
    SPECIALTIES,
    format_results,
    retrieve_evidence,
)

__all__ = [
    # Day 1 — data loading
    "ClinicalRecord",
    "CLINICAL_FIELDS",
    "DataValidationError",
    "get_record",
    "load_records",
    # Day 2 — transparent retrieval baseline
    "CONCEPTS",
    "EvidenceItem",
    "FORBIDDEN_LABEL_FIELDS",
    "RetrievalError",
    "SPECIALTIES",
    "format_results",
    "retrieve_evidence",
]
