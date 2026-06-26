#!/usr/bin/env python3
"""D0.3 退市数据增量更新

检查并更新退市数据到最新。
"""

from pathlib import Path
import sys

# 使用 delisted.py 中的现有逻辑
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.delisted import DelistManager


def update_delisted():
    print("[D0.3] 更新退市数据...")
    mgr = DelistManager()
    df = mgr.fetch_all(force=True)  # 强制重新拉取
    print(f"  ✅ 退市数据已更新: {len(df)} 只")
    return df


if __name__ == "__main__":
    update_delisted()
