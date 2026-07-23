"""Validate all DGL-PCA YAML configs."""

from __future__ import annotations

import argparse

from dgl_pca.utils.config import iter_config_paths, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DGL-PCA YAML configs.")
    parser.add_argument("config_root", nargs="?", default="configs", help="Directory containing YAML configs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = iter_config_paths(args.config_root)
    if not paths:
        raise FileNotFoundError(f"No YAML configs found under {args.config_root}")
    for path in paths:
        load_config(path)
        print(f"validated: {path}")
    print(f"Validated {len(paths)} config files")


if __name__ == "__main__":
    main()

