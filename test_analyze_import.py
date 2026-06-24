import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_analyze_import_has_no_report_side_effect(capsys):
    module = importlib.import_module("tools.backtest_mvp.analyze")
    captured = capsys.readouterr()

    assert callable(module.compute_ic_weights)
    assert "加载因子面板" not in captured.out
    assert "因子冗余审计" not in captured.out

