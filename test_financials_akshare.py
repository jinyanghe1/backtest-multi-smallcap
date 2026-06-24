import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from tools.backtest_mvp import financials_akshare


def test_main_non_merge_calls_build_financial_panel(monkeypatch, capsys):
    calls = {}

    def fake_build_financial_panel(data_dir, symbols=None, refresh=False):
        calls["data_dir"] = data_dir
        calls["symbols"] = symbols
        calls["refresh"] = refresh
        return pd.DataFrame({"symbol": ["sh600000"], "roe": [1.2]})

    monkeypatch.setattr(financials_akshare, "build_financial_panel", fake_build_financial_panel)
    monkeypatch.setattr(
        sys,
        "argv",
        ["financials_akshare.py", "--symbols", "sh600000", "--refresh", "--data-dir", "/tmp/data"],
    )

    financials_akshare.main()
    output = capsys.readouterr().out

    assert calls == {"data_dir": "/tmp/data", "symbols": ["sh600000"], "refresh": True}
    assert "获取完成" in output


def test_module_exposes_expected_builder():
    assert callable(financials_akshare.build_financial_panel)

