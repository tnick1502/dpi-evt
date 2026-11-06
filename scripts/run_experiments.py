#!/usr/bin/env python3
"""Run data preparation, site-held-out experiments and external evaluation."""
from pathlib import Path
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "external_validation"))

from dpi_evt.experiments.final_run import (
    prepare_populations, preflight, notebook_run, LANDMARKS, CORE_MODELS,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "prepare", "train", "internal", "prefix", "external", "report", "verify", "all"))
    parser.add_argument("--smoke", action="store_true", help="One epoch and one fold, written to separate smoke-test outputs")
    parser.add_argument("--device", default="auto", help="auto selects CUDA, MPS or CPU; use cpu to reproduce the recorded results")
    args = parser.parse_args()
    os.environ["LIQ_FINAL_SMOKE"] = "1" if args.smoke else "0"
    os.environ["LIQ_FINAL_DEVICE"] = args.device
    if args.stage == "check":
        print(preflight(ROOT, require_data=True, deep=True).to_string(index=False))
        return
    if args.stage in ("prepare", "all"):
        print(prepare_populations(ROOT, raw_root=os.environ.get("LIQ_REAL_ROOT")).to_string(index=False))
    if args.stage == "prepare":
        return
    preflight(ROOT, deep=args.stage == "all")
    run = notebook_run(ROOT)
    print("Output:", run.out, flush=True)
    if args.stage == "train":
        print(run.train_preview(CORE_MODELS).to_string(index=False))
    if args.stage in ("internal", "all"):
        raw, samples = run.grouped()
        if args.stage == "internal":
            run.summarize(raw, samples, n_boot=50 if args.smoke else 2000)
    if args.stage in ("prefix", "all"):
        for n0 in LANDMARKS:
            run.grouped(n0=n0, common=True)
    if args.stage in ("external", "all"):
        from extval.final_run import ExternalRun
        external = ExternalRun(run)
        external.evaluate()
        for n0 in LANDMARKS:
            external.evaluate(n0=n0, common=True)
    if args.stage in ("report", "all"):
        from dpi_evt.experiments.final_reports import export_reports
        print("Final tables and figures:", export_reports(run, n_boot=50 if args.smoke else 2000), flush=True)
    if args.stage in ("verify", "report", "all"):
        from dpi_evt.experiments.run_checks import verify_run
        print(verify_run(run).to_string(index=False))


if __name__ == "__main__":
    main()
