#!/usr/bin/env python3
"""
Copy a W&B run's config/history/summary into another project/entity.

Example:
  python scripts/copy_wandb_run.py \
    --src xingyu20/nanochat/e3nn9g17 \
    --dst-project nanochat
"""

import argparse
import json
from collections.abc import Mapping
from typing import Dict, Any

import wandb


def _clean_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in (cfg or {}).items():
        if k.startswith("_"):
            continue
        out[k] = v
    return out


def _clean_row(row: Dict[str, Any]) -> Dict[str, Any]:
    # Ignore internal runtime bookkeeping keys.
    drop = {"_runtime", "_timestamp", "_step"}
    return {k: v for k, v in row.items() if k not in drop}


def _to_builtin(value: Any) -> Any:
    """Convert nested W&B containers (e.g. SummarySubDict) to plain Python types."""
    if isinstance(value, Mapping):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_builtin(v) for v in value]
    if hasattr(value, "items"):
        try:
            return {k: _to_builtin(v) for k, v in value.items()}
        except Exception:
            pass
    return value


def _is_jsonable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Source run path: entity/project/run_id")
    p.add_argument("--dst-project", required=True, help="Destination project name")
    p.add_argument("--dst-entity", default=None, help="Destination entity/team (defaults to current account)")
    p.add_argument("--dst-name", default=None, help="Destination run name (default: '<src_name>-copy')")
    p.add_argument("--max-rows", type=int, default=None, help="Optional cap on history rows for quick tests")
    args = p.parse_args()

    api = wandb.Api()
    src = api.run(args.src)
    dst_name = args.dst_name or f"{src.name}-copy"

    dst = wandb.init(
        project=args.dst_project,
        entity=args.dst_entity,
        name=dst_name,
        config=_clean_config(src.config),
        tags=list(src.tags or []),
        notes=src.notes,
        reinit=True,
    )
    if dst is None:
        raise RuntimeError("wandb.init returned None")

    rows = src.scan_history()
    count = 0
    for row in rows:
        step = row.get("_step", None)
        payload = _clean_row(row)
        if payload:
            if step is None:
                dst.log(payload)
            else:
                dst.log(payload, step=int(step))
        count += 1
        if args.max_rows is not None and count >= args.max_rows:
            break

    skipped_summary_keys = []
    for k, v in dict(src.summary).items():
        clean_v = _to_builtin(v)
        if _is_jsonable(clean_v):
            dst.summary[k] = clean_v
        else:
            skipped_summary_keys.append(k)

    if skipped_summary_keys:
        print(f"Skipped non-JSON summary keys: {', '.join(skipped_summary_keys)}")
    dst.finish()
    print(
        f"Copied {count} history rows from {args.src} "
        f"to {args.dst_entity or '<default_entity>'}/{args.dst_project}/{dst.id}"
    )


if __name__ == "__main__":
    main()
