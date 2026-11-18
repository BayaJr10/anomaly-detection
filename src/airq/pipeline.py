"""Command-line entry point for the reproducible pipeline.

    python -m airq.pipeline ingest   [--download] [--raw-dir DIR] [--no-network] [--no-xlsx]
    python -m airq.pipeline validate-data [--rebuilt FILE] [--reference FILE]
    python -m airq.pipeline train    [--data FILE] [--quick]
    python -m airq.pipeline predict  --input FILE --output FILE [--explain]
    python -m airq.pipeline all      (ingest from data/raw if present, then train)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from airq.config import load_config, resolve


def cmd_ingest(a, cfg):
    from airq.data import ingest
    if a.download:
        ingest.download_archive(cfg, a.raw_dir)
    files = ingest.find_raw_files(a.raw_dir or cfg["ingest"]["raw_dir"])
    if not files:
        sys.exit(f"No raw files in {a.raw_dir or cfg['ingest']['raw_dir']}. Use --download, or "
                 "--raw-dir data/sample_raw for the offline sample.")
    df = ingest.build_dataset(files, cfg, allow_network_geocoding=not a.no_network)
    paths = ingest.save_dataset(df, cfg, xlsx=not a.no_xlsx)
    print(f"Hourly dataset: {len(df):,} rows, {df['LOCATION'].nunique()} sensors -> "
          + ", ".join(str(p) for p in paths.values()))


def cmd_validate(a, cfg):
    from airq.data import ingest, validate
    new = ingest.load_dataset(a.rebuilt or cfg["ingest"]["output_csv"])
    print(json.dumps(validate.schema_report(new), indent=2))
    if a.reference:
        validate.print_report(validate.compare(new, ingest.load_dataset(a.reference)))


def cmd_train(a, cfg):
    from airq.train import run_training
    rep = run_training(cfg, a.data, quick=a.quick)
    m = rep["metrics"]
    print(json.dumps({"threshold": rep["threshold"], "val": m["val"], "test": m["test"]}, indent=2, default=str))


def cmd_predict(a, cfg):
    import pandas as pd
    from airq.data.ingest import load_dataset
    from airq.scoring import AnomalyScorer
    scorer = AnomalyScorer(cfg=cfg)
    raw = load_dataset(a.input)
    out, info = scorer.score(raw, explain=a.explain)
    if a.only_flagged:
        out = out[out["is_anomaly"] == 1]
    out_path = resolve(a.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(json.dumps(info, indent=2, default=str))
    print(f"Wrote {len(out):,} rows to {out_path}")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="airq")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest", help="raw sensors.AFRICA archive -> hourly dataset")
    p.add_argument("--download", action="store_true", help="download the monthly archive from openAFRICA first")
    p.add_argument("--raw-dir", default=None)
    p.add_argument("--no-network", action="store_true", help="use only the geocode cache")
    p.add_argument("--no-xlsx", action="store_true", help="skip the (slow) Excel export")
    p = sub.add_parser("validate-data", help="schema checks, optional comparison with a reference file")
    p.add_argument("--rebuilt", default=None)
    p.add_argument("--reference", default=None)
    p = sub.add_parser("train", help="fit detectors, meta-model, cause model; write models/ and reports/")
    p.add_argument("--data", default=None)
    p.add_argument("--quick", action="store_true", help="fewer trees and no CV (smoke test)")
    p = sub.add_parser("predict", help="batch-score a CSV/XLSX of hourly readings")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--explain", action="store_true")
    p.add_argument("--only-flagged", action="store_true")
    p = sub.add_parser("all", help="ingest (if raw files exist) then train")
    p.add_argument("--raw-dir", default=None)
    p.add_argument("--no-network", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config()
    if a.cmd == "ingest":
        cmd_ingest(a, cfg)
    elif a.cmd == "validate-data":
        cmd_validate(a, cfg)
    elif a.cmd == "train":
        cmd_train(a, cfg)
    elif a.cmd == "predict":
        cmd_predict(a, cfg)
    elif a.cmd == "all":
        from airq.data import ingest
        if ingest.find_raw_files(a.raw_dir or cfg["ingest"]["raw_dir"]):
            a.download, a.no_xlsx = False, True
            cmd_ingest(a, cfg)
        a.data, a.quick = None, False
        cmd_train(a, cfg)


if __name__ == "__main__":
    main()
