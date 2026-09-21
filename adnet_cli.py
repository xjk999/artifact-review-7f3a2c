"""Command-line interface for AD-NET."""
import argparse
import csv
import importlib
import json
from pathlib import Path

from adnet.data import load_config, load_manifest, select

PROJECT = Path(__file__).resolve().parent


def parser():
    root = argparse.ArgumentParser(description="AD-NET: SegResNet -> Syn-CTA -> final AD detection")
    commands = root.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=str(PROJECT / "config.json"))
    common.add_argument("--device", help="Override cpu/cuda:0 in config")
    audit = commands.add_parser("audit", help="Check a patient manifest")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--skip-file-check", action="store_true", help="Check manifest schema only")
    commands.add_parser("doctor", parents=[common], help="Report dependencies and CUDA")
    commands.add_parser("check-weights", parents=[common], help="Check model checkpoint compatibility")
    seg = commands.add_parser("train-segmentation", parents=[common])
    seg.add_argument("--manifest", required=True)
    seg.add_argument("--output", required=True)
    seg.add_argument("--resume")
    prep = commands.add_parser("prepare", parents=[common], help="Prepare paired training cases with predicted masks")
    prep.add_argument("--manifest", required=True)
    prep.add_argument("--output", required=True)
    for stage in ("generation", "detection"):
        train = commands.add_parser(f"train-{stage}", parents=[common])
        train.add_argument("--index", required=True)
        train.add_argument("--output", required=True)
        train.add_argument("--resume")
    synth = commands.add_parser("synthesize", parents=[common], help="Generate Syn-CTA cache for detector training")
    synth.add_argument("--index", required=True)
    synth.add_argument("--output", required=True)
    synth.add_argument("--quiet-progress", action="store_true")
    infer = commands.add_parser("infer", parents=[common], help="Run inference on NCCT images")
    source = infer.add_mutually_exclusive_group(required=True)
    source.add_argument("--ncct")
    source.add_argument("--manifest")
    infer.add_argument("--patient-id", default="case")
    infer.add_argument("--split", choices=["train", "internal_test", "external_test"], default="internal_test")
    infer.add_argument("--output", required=True)
    infer.add_argument("--quiet-progress", action="store_true")
    return root


def main():
    args = parser().parse_args()
    if args.command == "audit":
        rows = load_manifest(args.manifest, check_files=not args.skip_file_check)
        print(json.dumps({"patients": len(rows), "split_counts": {s: sum(r["split"] == s for r in rows)
                         for s in ("train", "internal_test", "external_test")},
                         "paired_training_cases": sum(r["split"] == "train" and bool(r["cta"]) for r in rows)},
                         ensure_ascii=False, indent=2))
        return
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    if args.command == "doctor":
        versions = {}
        for module in ("torch", "monai", "numpy", "nibabel", "scipy", "pywt", "tqdm"):
            loaded = importlib.import_module(module)
            versions[module] = loaded.__version__
        import torch
        versions["cuda_available"] = torch.cuda.is_available()
        versions["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        print(json.dumps(versions, indent=2))
    elif args.command == "check-weights":
        import gc
        from adnet.models import diffusion_model, load_weights, segmentation_model
        reports = []
        for role, checkpoint_key in (("segmentation", "segmentation_checkpoint"),
                                      ("generation", "diffusion_checkpoint"), ("detection", "detection_checkpoint")):
            model = segmentation_model(config["segmentation"]) if role == "segmentation" else diffusion_model(config["diffusion"])
            reports.append(load_weights(model, config[checkpoint_key], role))
            del model
            gc.collect()
        print(json.dumps(reports, ensure_ascii=False, indent=2))
    elif args.command == "train-segmentation":
        from adnet.training import train_segmentation
        print(train_segmentation(args.manifest, config, args.output, resume=args.resume))
    elif args.command == "prepare":
        from adnet.cache import prepare
        print(prepare(args.manifest, config, args.output))
    elif args.command in {"train-generation", "train-detection"}:
        from adnet.training import train_wavelet
        print(train_wavelet(args.index, config, args.output, args.command.removeprefix("train-"),
                            resume=args.resume))
    elif args.command == "synthesize":
        from adnet.cache import synthesize
        print(synthesize(args.index, config, args.output, progress=not args.quiet_progress))
    elif args.command == "infer":
        from adnet.pipeline import ADNet, new_output
        from adnet.training import seed_everything
        from adnet.data import manifest_digest
        if args.manifest:
            rows = select(load_manifest(args.manifest, check_files=False), args.split)
            for row in rows:
                if not Path(row["ncct"]).is_file():
                    raise FileNotFoundError(row["ncct"])
        else:
            if any(c in args.patient_id for c in '/\\:') or args.patient_id in {".", "..", ""}:
                raise ValueError("Unsafe patient-id")
            rows = [{"patient_id": args.patient_id, "ncct": args.ncct, "split": "inference", "label": None}]
        model = ADNet(config)
        output = new_output(args.output)
        seed_everything(config["seed"])
        predictions = []
        for row in rows:
            result = model.infer_case(row["ncct"], row["patient_id"], output / row["patient_id"],
                                      progress=not args.quiet_progress)
            predictions.append({"patient_id": row["patient_id"], "split": row["split"],
                                "label": row["label"], "probability": result["probability"],
                                "prediction": result["prediction"], "threshold": result["threshold"]})
        with (output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
        run_info = {"config": config, "weights": model.weights,
                    "manifest_sha256": manifest_digest(args.manifest) if args.manifest else None,
                    "ordered_stages": ["SegResNet segmentation", "Syn-CTA generation", "AD detection"],
                    "threshold_selection": "configured value"}
        (output / "run.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Completed: {output / 'predictions.csv'}")


if __name__ == "__main__":
    main()
