"""Patient-level manifests and framework configuration."""
import csv
import hashlib
import json
from pathlib import Path

SPLITS = {"train", "internal_test", "external_test"}
PATH_FIELDS = ("ncct", "cta", "mask")


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    for key in ("segmentation_checkpoint", "diffusion_checkpoint", "detection_checkpoint"):
        if config.get(key):
            config[key] = str((path.parent / config[key]).resolve())
    validate_training_config(config)
    for stage in ("generation", "detection"):
        initial = config["training"][stage].get("initial_checkpoint")
        if initial:
            config["training"][stage]["initial_checkpoint"] = str((path.parent / initial).resolve())
    if not 0 <= config["threshold"] <= 1:
        raise ValueError("threshold must be in [0, 1]")
    d = config["diffusion"]
    if d["volume_size"] < 128 or d["volume_size"] % 128:
        raise ValueError("Six-downsample WavUNet requires volume_size divisible by 128 and >= 128")
    if not 0 <= d["tile_overlap"] < 1:
        raise ValueError("tile_overlap must be in [0, 1)")
    if d["case_aggregation"] not in {"max", "mean"}:
        raise ValueError("case_aggregation must be max or mean")
    return config


def validate_training_config(config):
    training = config.get("training", {})
    if training.get("mode") != "final":
        raise ValueError("training.mode must be final")
    for stage, loss in (("segmentation", "DiceCE"), ("generation", "MSE"), ("detection", "BCE")):
        settings = training[stage]
        for key in ("epochs" if stage == "segmentation" else "steps", "save_every", "batch_size"):
            if not isinstance(settings[key], int) or isinstance(settings[key], bool) or settings[key] < 1:
                raise ValueError(f"training.{stage}.{key} must be a positive integer")
        if settings["optimizer"] != "AdamW" or settings["loss"] != loss:
            raise ValueError(f"{stage} requires AdamW/{loss}")
        if settings["learning_rate"] <= 0 or settings["weight_decay"] < 0 or settings["eps"] <= 0:
            raise ValueError(f"Invalid {stage} optimizer settings")
        if len(settings["betas"]) != 2 or any(not 0 <= v < 1 for v in settings["betas"]):
            raise ValueError(f"Invalid {stage} betas")
        if stage != "segmentation" and (not isinstance(settings["log_every"], int) or settings["log_every"] < 1):
            raise ValueError(f"Invalid {stage} log_every")
        if stage != "detection":
            probabilities = settings["flip_probabilities"]
            if len(probabilities) != 3 or any(not 0 <= p <= 1 for p in probabilities):
                raise ValueError(f"Invalid {stage} flip probabilities")
    s = training["segmentation"]
    if s["scheduler"] != "cosine" or not 0 <= s["min_learning_rate"] <= s["learning_rate"]:
        raise ValueError("Invalid segmentation scheduler")
    if not isinstance(s["num_workers"], int) or s["num_workers"] < 0:
        raise ValueError("Invalid segmentation num_workers")
    if s["samples_per_case"] < 1 or s["crop_positive"] < 0 or s["crop_negative"] < 0 or s["crop_positive"] + s["crop_negative"] <= 0:
        raise ValueError("Invalid segmentation crop settings")
    if not 0 <= s["intensity_shift_probability"] <= 1 or s["intensity_shift_offset"] < 0:
        raise ValueError("Invalid intensity shift settings")
    if min(s["dice_weight"], s["ce_weight"], s["smooth_nr"]) < 0 or s["smooth_dr"] <= 0 or s["dice_weight"] + s["ce_weight"] <= 0:
        raise ValueError("Invalid segmentation loss settings")


def load_manifest(path, check_files=True):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"patient_id", "center", "split", "ncct"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest requires patient_id,center,split,ncct")
        rows = [dict(row) for row in reader]
    seen_ids, seen_paths = set(), {}
    for row in rows:
        identifier = row["patient_id"].strip()
        if not identifier or identifier in seen_ids:
            raise ValueError(f"Empty/duplicate patient_id: {identifier!r}")
        seen_ids.add(identifier)
        row["patient_id"] = identifier
        if any(c in identifier for c in '/\\:') or identifier in {".", ".."}:
            raise ValueError("patient_id must be a safe filename component")
        if row["split"] not in SPLITS:
            raise ValueError(f"Unknown split: {row['split']}")
        if not row["ncct"].strip():
            raise ValueError(f"Missing NCCT: {identifier}")
        for field in PATH_FIELDS:
            value = (row.get(field) or "").strip()
            row[field] = str((path.parent / value).resolve()) if value else ""
            if not value:
                continue
            canonical = row[field].casefold()
            if canonical in seen_paths:
                raise ValueError(f"Repeated image/label path: {field} for {identifier}")
            seen_paths[canonical] = identifier
            if check_files and not Path(row[field]).is_file():
                raise FileNotFoundError(row[field])
        label = (row.get("label") or "").strip()
        row["label"] = int(label) if label else None
        if row["label"] not in {None, 0, 1}:
            raise ValueError(f"Binary label required: {identifier}")
    if not rows:
        raise ValueError("Empty manifest")
    return rows


def select(rows, split, required=()):
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"No cases in split {split}")
    for row in selected:
        for field in required:
            if row.get(field) is None or row.get(field) == "":
                raise ValueError(f"{row['patient_id']} needs {field} for this stage")
    return selected


def manifest_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_training_split(split):
    if split != "train":
        raise ValueError("Training/cache preparation requires split=train")
