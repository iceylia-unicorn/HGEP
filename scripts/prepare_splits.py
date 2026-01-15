from __future__ import annotations # 兼容 Python 3.7+

import argparse # 命令行参数解析
from pathlib import Path # 文件路径操作

from gpbench.data.loaders import load_hgb_node_task 
from gpbench.data.splits import SplitManager, make_fewshot_split


def main(): 
    ap = argparse.ArgumentParser()# 创建命令行参数解析器
    ap.add_argument("--root", type=str, default="data")
    ap.add_argument("--dataset", type=str, default="ACM", choices=["ACM", "DBLP", "IMDB"])
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])  # 对齐 EdgePrompt 5 seeds 
    ap.add_argument("--out", type=str, default="splits")
    ap.add_argument("--val_shot_per_class", type=int, default=-1)  # -1 表示用全量 val
    args = ap.parse_args()

    task = load_hgb_node_task(args.root, args.dataset)
    sm = SplitManager(args.out)

    for shot in args.shots:
        for seed in args.seeds:
            split = make_fewshot_split(
                y=task.y,
                train_idx_base=task.train_idx,
                val_idx_base=task.val_idx,
                test_idx_base=task.test_idx,
                shot=shot,
                seed=seed,
                val_shot_per_class=None if args.val_shot_per_class < 0 else args.val_shot_per_class,
            )
            sm.save(task.dataset_name, shot, seed, split)

    print(f"Done. Splits saved to: {Path(args.out) / task.dataset_name}")

if __name__ == "__main__":
    main()
