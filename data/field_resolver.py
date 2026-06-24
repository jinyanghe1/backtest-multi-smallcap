"""Field conflict resolution and source-priority merging."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from itertools import combinations
from typing import Mapping, Optional

import numpy as np
import pandas as pd


DEFAULT_PRIORITY_RULES: dict[str, list[str]] = {
    "bps": ["eastmoney", "em", "ths", "sina", "akshare"],
    "eps": ["eastmoney", "em", "ths", "sina", "akshare"],
    "revenue": ["eastmoney", "em", "ths", "sina", "akshare"],
    "net_profit": ["eastmoney", "em", "ths", "sina", "akshare"],
    "total_equity": ["eastmoney", "em", "sina", "ths", "akshare"],
    "mcap": ["westock", "eastmoney", "akshare", "sina"],
    "pb": ["eastmoney", "westock", "akshare", "sina"],
    "pe": ["eastmoney", "westock", "akshare", "sina"],
    "industry": ["akshare", "eastmoney", "ths"],
}

RATIO_FIELDS = {
    "roe",
    "roe_ttm",
    "roa",
    "roa_ttm",
    "gross_margin",
    "net_margin",
    "operating_margin",
    "revenue_growth_yoy",
    "profit_growth_yoy",
    "revenue_growth_ttm",
    "profit_growth_ttm",
    "eps_growth_yoy",
    "debt_ratio",
    "current_ratio",
    "quick_ratio",
}


@dataclass(frozen=True)
class ConflictReport:
    has_conflict: bool
    threshold: float
    conflicts: dict[str, float] = dataclass_field(default_factory=dict)
    overlaps: dict[str, int] = dataclass_field(default_factory=dict)
    insufficient_overlap: list[str] = dataclass_field(default_factory=list)


@dataclass(frozen=True)
class ResolveResult:
    data: pd.Series
    source: str
    field: str
    conflict_report: ConflictReport
    metadata: dict


def _as_series(value: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.copy()
    if isinstance(value, pd.DataFrame) and value.shape[1] == 1:
        return value.iloc[:, 0].copy()
    raise TypeError("field resolver expects Series or one-column DataFrame values")


def _normalise_unit(field: str, series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce") if series.dtype == object else series.copy()
    if field in RATIO_FIELDS:
        valid = out.dropna()
        if not valid.empty and valid.abs().median() > 1.5:
            out = out / 100.0
    return out.sort_index()


def _priority_for(field: str, sources: Mapping[str, pd.Series], priority_rules: Optional[Mapping[str, list[str]]]) -> list[str]:
    rules = priority_rules or DEFAULT_PRIORITY_RULES
    configured = rules.get(field, rules.get(field.lower(), []))
    ordered = [src for src in configured if src in sources]
    ordered.extend(src for src in sources if src not in ordered)
    return ordered


def detect_conflict(
    sources: Mapping[str, pd.Series | pd.DataFrame],
    threshold: float = 0.01,
    min_overlap: int = 3,
) -> ConflictReport:
    """Detect pairwise relative conflicts on overlapping non-null dates."""
    normalised = {name: _as_series(series).sort_index() for name, series in sources.items()}
    conflicts: dict[str, float] = {}
    overlaps: dict[str, int] = {}
    insufficient: list[str] = []

    for left, right in combinations(normalised, 2):
        pair_name = f"{left}:{right}"
        aligned = pd.concat([normalised[left], normalised[right]], axis=1, join="inner").dropna()
        overlaps[pair_name] = len(aligned)
        if len(aligned) < min_overlap:
            insufficient.append(pair_name)
            continue
        a = pd.to_numeric(aligned.iloc[:, 0], errors="coerce")
        b = pd.to_numeric(aligned.iloc[:, 1], errors="coerce")
        denom = pd.concat([a.abs(), b.abs()], axis=1).max(axis=1).clip(lower=1e-12)
        rel_diff = ((a - b).abs() / denom).dropna()
        median_diff = float(rel_diff.median()) if not rel_diff.empty else 0.0
        if median_diff > threshold:
            conflicts[pair_name] = median_diff

    return ConflictReport(
        has_conflict=bool(conflicts),
        threshold=threshold,
        conflicts=conflicts,
        overlaps=overlaps,
        insufficient_overlap=insufficient,
    )


def resolve(
    field: str,
    sources: Mapping[str, pd.Series | pd.DataFrame],
    priority_rules: Optional[Mapping[str, list[str]]] = None,
    threshold: float = 0.01,
) -> ResolveResult:
    """Resolve a field using source priority, with lower-priority gap filling."""
    if not sources:
        raise ValueError("sources must not be empty")

    normalised = {
        name: _normalise_unit(field, _as_series(series))
        for name, series in sources.items()
    }
    priority = _priority_for(field, normalised, priority_rules)
    report = detect_conflict(normalised, threshold=threshold)

    selected_source = ""
    resolved: pd.Series | None = None
    fallback_sources: list[str] = []

    for source in priority:
        series = normalised[source].dropna()
        if series.empty:
            continue
        if resolved is None:
            selected_source = source
            resolved = normalised[source].copy()
        else:
            before = int(resolved.notna().sum())
            resolved = resolved.combine_first(normalised[source])
            if int(resolved.notna().sum()) > before:
                fallback_sources.append(source)

    if resolved is None or resolved.dropna().empty:
        raise ValueError(f"no usable source data for field {field}")

    return ResolveResult(
        data=resolved.sort_index(),
        source=selected_source,
        field=field,
        conflict_report=report,
        metadata={
            "priority": priority,
            "fallback_sources": fallback_sources,
            "unit_normalized": field in RATIO_FIELDS,
        },
    )


class FieldResolver:
    """Object-oriented wrapper for code that prefers injectable state."""

    def __init__(self, priority_rules: Optional[Mapping[str, list[str]]] = None):
        self.priority_rules = priority_rules

    def resolve(
        self,
        field: str,
        sources: Mapping[str, pd.Series | pd.DataFrame],
        threshold: float = 0.01,
    ) -> ResolveResult:
        return resolve(field, sources, self.priority_rules, threshold)

    def detect_conflict(
        self,
        sources: Mapping[str, pd.Series | pd.DataFrame],
        threshold: float = 0.01,
        min_overlap: int = 3,
    ) -> ConflictReport:
        return detect_conflict(sources, threshold, min_overlap)

