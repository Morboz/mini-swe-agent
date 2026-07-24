#!/usr/bin/env python3
"""
合并原始数据集（有 image_name）和自制 locbench 数据集（有 edit_functions），
输出一个完整的 jsonl，每条记录同时具备 image_name 和 edit_functions。
"""

import json
import sys
from pathlib import Path


def load_index(path: str) -> dict:
    """Load a jsonl file into a dict keyed by instance_id."""
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            data[rec["instance_id"]] = rec
    return data


def main():
    raw_path = sys.argv[1]  # 原始数据集
    loc_path = sys.argv[2]  # 自制 locbench 数据集
    out_path = sys.argv[3]  # 输出 jsonl

    raw_index = load_index(raw_path)
    loc_index = load_index(loc_path)

    matched = 0
    missing = 0
    with open(out_path, "w") as out:
        for iid, rec in sorted(loc_index.items()):
            raw = raw_index.get(iid)
            if raw:
                # 从原始记录复制 image_name
                rec["image_name"] = raw.get("image_name", "")
                matched += 1
            else:
                print(
                    f"  ⚠️  {iid}: 原始数据集中未找到，image_name 留空",
                    file=sys.stderr,
                )
                rec["image_name"] = rec.get("image_name", "")
                missing += 1

            # 确保 edit_functions 存在
            if "edit_functions" not in rec:
                print(f"  ⚠️  {iid}: 缺少 edit_functions", file=sys.stderr)

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n完成: {len(loc_index)} 条")
    print(f"  已匹配（有 image_name）: {matched}")
    print(f"  未匹配（image_name 空）: {missing}")
    print(f"  输出: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python merge_image_name.py <原始.jsonl> <locbench.jsonl> <输出.jsonl>")
        sys.exit(1)
    main()