"""Pipeline driver.

    python -m src.main download --config config/default.yaml     # fetch raw data (full run only)
    python -m src.main prepare  --config config/quickstart.yaml   # clean, weak labels, mask, split
    python -m src.main train    --config config/quickstart.yaml   # fit CoRE, tune thresholds
    python -m src.main evaluate --config config/quickstart.yaml   # test metrics, baselines, rationales
    python -m src.main predict  --config config/quickstart.yaml --text "I can't focus ..."
    python -m src.main all      --config config/quickstart.yaml   # prepare + train + evaluate + demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.cmn.config import load_config, resolve
from src.cmn.io import get_logger, write_json
from src.cmn.seed import set_seed

log = get_logger()


def _model_path(cfg) -> Path:
    return resolve(cfg["output_dir"]) / "model.joblib"


def cmd_download(cfg, args):
    from src.dataprep.load import download_low2020
    if cfg["data"]["source"] != "low2020":
        log.info("data.source is '%s'; nothing to download.", cfg["data"]["source"])
        return
    download_low2020(cfg)


def cmd_prepare(cfg, args):
    from src.dataprep.prepare import prepare
    return prepare(cfg)


def cmd_train(cfg, args):
    from src.core.model import CoRE
    from src.dataprep.prepare import load_splits
    splits = load_splits(cfg)
    model = CoRE.from_config(cfg).fit(splits["train"], splits["val"])
    model.save(_model_path(cfg))
    write_json({d: float(t) for d, t in zip(cfg["conditions"], model.thresholds_)},
               resolve(cfg["output_dir"]) / "thresholds.json")
    log.info("model saved to %s; thresholds %s", _model_path(cfg),
             dict(zip(cfg["conditions"], model.thresholds_.round(2).tolist())))
    return model


def cmd_evaluate(cfg, args):
    from src.core.model import CoRE
    from src.dataprep.prepare import load_splits
    from src.eval.evaluate import evaluate
    return evaluate(cfg, load_splits(cfg), CoRE.load(_model_path(cfg)))


def cmd_predict(cfg, args):
    from src.core.model import CoRE, format_output
    model = CoRE.load(_model_path(cfg))
    if args.input:
        df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
        rows = [{"input": t, **model.explain(t)} for t in df[args.column]]
        out = Path(args.output or resolve(cfg["output_dir"]) / "predictions_custom.jsonl")
        from src.cmn.io import write_jsonl
        write_jsonl(rows, out)
        log.info("wrote %d predictions to %s", len(rows), out)
        return rows
    text = args.text or cfg["demo_text"]
    out = model.explain(text)
    print(format_output(out))
    if args.json:
        print(json.dumps(out, indent=2))
    write_json({"input": text, **out}, resolve(cfg["output_dir"]) / "demo_prediction.json")
    return out


def cmd_all(cfg, args):
    cmd_prepare(cfg, args)
    cmd_train(cfg, args)
    cmd_evaluate(cfg, args)
    args.text, args.input = None, None
    cmd_predict(cfg, args)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.main", description="CoRE pipeline driver")
    ap.add_argument("command", choices=["download", "prepare", "train", "evaluate", "predict", "all"])
    ap.add_argument("--config", default="config/quickstart.yaml")
    ap.add_argument("--text", help="predict: a single self-reported description")
    ap.add_argument("--input", help="predict: CSV file with one description per row")
    ap.add_argument("--column", default="text", help="predict: text column in --input")
    ap.add_argument("--output", help="predict: output .jsonl path for --input")
    ap.add_argument("--json", action="store_true", help="predict: also print the full JSON output")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    log.info("command=%s config=%s seed=%d", args.command, args.config, cfg["seed"])
    {"download": cmd_download, "prepare": cmd_prepare, "train": cmd_train, "evaluate": cmd_evaluate,
     "predict": cmd_predict, "all": cmd_all}[args.command](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
