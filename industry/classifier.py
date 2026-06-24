"""Cached industry classification interface for neutralized factor ranking."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd


def default_cache_path() -> Path:
    """Return the default path for the Shenwan industry parquet cache."""
    return Path(__file__).resolve().parent.parent / "industry_cache" / "shenwan_map.parquet"


class IndustrySource(Enum):
    SW_INDUSTRY_1 = "sw_industry_1"
    SW_INDUSTRY_2 = "sw_industry_2"


class IndustryClassifier:
    def __init__(self, industry_map: Optional[pd.DataFrame] = None, default: str = "Unknown"):
        self.default = default
        self.industry_map = self._normalise_map(industry_map) if industry_map is not None else pd.DataFrame()

    @staticmethod
    def _normalise_map(industry_map: pd.DataFrame) -> pd.DataFrame:
        df = industry_map.copy()
        if "symbol" not in df.columns and df.index.name == "symbol":
            df = df.reset_index()
        if "symbol" not in df.columns:
            raise ValueError("industry_map must include a symbol column")
        for col in [IndustrySource.SW_INDUSTRY_1.value, IndustrySource.SW_INDUSTRY_2.value]:
            if col not in df.columns:
                df[col] = "Unknown"
        if "source" not in df.columns:
            df["source"] = "cache"
        if "updated_at" not in df.columns:
            df["updated_at"] = pd.Timestamp.now()
        else:
            df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce").fillna(pd.Timestamp.now())
        return df.drop_duplicates("symbol", keep="last")

    def load_industry_map(self, cache_path: str | Path, refresh: bool = False) -> pd.DataFrame:
        path = Path(cache_path)
        if refresh:
            raise NotImplementedError("refreshing industry data is intentionally out of scope for the first implementation")
        if not path.exists():
            self.industry_map = pd.DataFrame(columns=["symbol", "sw_industry_1", "sw_industry_2", "source", "updated_at"])
            return self.industry_map
        if path.suffix == ".parquet":
            raw = pd.read_parquet(path)
        elif path.suffix == ".csv":
            raw = pd.read_csv(path)
        else:
            raise ValueError(f"unsupported industry map format: {path.suffix}")
        self.industry_map = self._normalise_map(raw)
        return self.industry_map

    def get(
        self,
        symbol: str,
        source: IndustrySource = IndustrySource.SW_INDUSTRY_2,
        default: Optional[str] = None,
    ) -> str:
        fallback = self.default if default is None else default
        if self.industry_map.empty:
            return fallback
        row = self.industry_map[self.industry_map["symbol"] == symbol]
        if row.empty:
            return fallback
        value = row.iloc[0].get(source.value, fallback)
        return fallback if pd.isna(value) or value == "" else str(value)

    def attach_industry(
        self,
        factor_panel: pd.DataFrame,
        industry_map: Optional[pd.DataFrame] = None,
        default: Optional[str] = None,
    ) -> pd.DataFrame:
        fallback = self.default if default is None else default
        mapping = self._normalise_map(industry_map) if industry_map is not None else self.industry_map
        result = factor_panel.copy()
        if mapping.empty:
            for col in [IndustrySource.SW_INDUSTRY_1.value, IndustrySource.SW_INDUSTRY_2.value]:
                result[col] = fallback
            return result

        cols = ["symbol", IndustrySource.SW_INDUSTRY_1.value, IndustrySource.SW_INDUSTRY_2.value]
        if isinstance(result.index, pd.MultiIndex):
            reset = result.reset_index()
            merged = reset.merge(mapping[cols], on="symbol", how="left")
            for col in cols[1:]:
                merged[col] = merged[col].fillna(fallback)
            return merged.set_index(result.index.names)

        if "symbol" not in result.columns:
            raise ValueError("factor_panel must have a symbol column or MultiIndex with symbol level")
        merged = result.merge(mapping[cols], on="symbol", how="left")
        for col in cols[1:]:
            merged[col] = merged[col].fillna(fallback)
        return merged


_DEFAULT_CLASSIFIER = IndustryClassifier()


def attach_industry(factor_panel: pd.DataFrame, industry_map: pd.DataFrame, default: str = "Unknown") -> pd.DataFrame:
    return IndustryClassifier(industry_map, default=default).attach_industry(factor_panel)


def load_default_industry_map(refresh: bool = False) -> pd.DataFrame:
    """Load the default Shenwan cache if it exists.

    This is a convenience wrapper that never calls an external API.
    """
    return IndustryClassifier().load_industry_map(default_cache_path(), refresh=refresh)


def get_shenwan_industry(symbol: str) -> str:
    return _DEFAULT_CLASSIFIER.get(symbol, IndustrySource.SW_INDUSTRY_1)


def get_shenwan_industry_2(symbol: str) -> str:
    return _DEFAULT_CLASSIFIER.get(symbol, IndustrySource.SW_INDUSTRY_2)

