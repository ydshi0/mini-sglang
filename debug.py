#!/usr/bin/env python3
"""
诊断脚本：逐级测试 speculative 模块的 import 链。
在 mini-sglang 根目录运行：python tests/diagnose_speculative.py
"""

import importlib
import sys
import traceback

CHECKS = [
    ("minisgl.core (swap_global_ctx)",
     "from minisgl.core import swap_global_ctx"),
    ("minisgl.speculative.verify",
     "from minisgl.speculative.verify import verify_greedy, verify_stochastic"),
    ("minisgl.speculative.draft_engine",
     "from minisgl.speculative.draft_engine import DraftEngine"),
    ("minisgl.speculative.spec_llm",
     "from minisgl.speculative.spec_llm import SpeculativeLLM"),
    ("minisgl.speculative (顶层)",
     "from minisgl.speculative import SpeculativeLLM"),
]


def main():
    print("=" * 60)
    print("Speculative Decoding 导入诊断")
    print("=" * 60)

    # 检查 speculative 目录是否存在
    import minisgl
    pkg_dir = minisgl.__path__[0]
    spec_dir = f"{pkg_dir}/speculative"
    import os
    files = os.listdir(spec_dir) if os.path.isdir(spec_dir) else []
    print(f"\nminisgl 包路径: {pkg_dir}")
    print(f"speculative/ 目录: {'存在' if os.path.isdir(spec_dir) else '不存在 !'}")
    if files:
        print(f"  文件: {', '.join(sorted(files))}")
    print()

    all_ok = True
    for name, stmt in CHECKS:
        try:
            exec(stmt)
            print(f"  ✓  {name}")
        except Exception as e:
            all_ok = False
            print(f"  ✗  {name}")
            print(f"     错误: {type(e).__name__}: {e}")
            # 打印完整 traceback 方便定位
            traceback.print_exc()
            print()

    print()
    if all_ok:
        print("全部通过 ✓")
    else:
        print("存在导入错误 ✗")
        print()
        print("常见修复方法:")
        print("  1. 确认 python/minisgl/speculative/ 下有以下文件:")
        print("       __init__.py, draft_engine.py, verify.py, spec_llm.py")
        print("  2. 确认 python/minisgl/core.py 包含 swap_global_ctx 函数")
        print("  3. 重新安装包:  pip install -e .  (在 mini-sglang 根目录)")
        print("  4. 清除缓存:   find . -name '__pycache__' -exec rm -rf {} +")


if __name__ == "__main__":
    main()