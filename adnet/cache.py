"""Prepare intermediate data for generation and detection training."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .data import load_manifest, manifest_digest, require_training_split
from .diffusion import ConditionalDiffusion
from .geometry import (add_tile, extract_tile, load_volume, normalize_hu, require_same_grid,
                       save_volume, tensor_to_xyz, tile_origins, xyz_to_tensor)
from .models import diffusion_model, load_weights
from .pipeline import Segmenter, new_output, resolve_device


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(manifest, config, output_dir):
    rows = [r for r in load_manifest(manifest, check_files=False) if r["split"] == "train"]
    if not rows:
        raise ValueError("No train cases")
    for row in rows:
        if not row["cta"]:
            raise ValueError(f"Paired registered CTA required: {row['patient_id']}")
    output = new_output(output_dir)
    segmenter = Segmenter(config)
    index = {"kind": "paired_generation", "manifest_sha256": manifest_digest(manifest),
             "config": config, "segmentation_sha256": file_digest(config["segmentation_checkpoint"]),
             "cases": [], "tiles": []}
    d = config["diffusion"]
    for row in rows:
        identifier = row["patient_id"]
        print(f"Preparing predicted aortic ROI: {identifier}", flush=True)
        case_dir = output / identifier
        case_dir.mkdir()
        reference, ncct_raw = load_volume(row["ncct"])
        cta_image, cta_raw = load_volume(row["cta"])
        require_same_grid(reference, cta_image, "paired CTA")
        mask = segmenter.segment(row["ncct"])
        mask_file = case_dir / "predicted_aorta.nii.gz"
        save_volume(mask, reference, mask_file, np.uint8)
        ncct = normalize_hu(ncct_raw, d["ncct_hu"]) * mask
        cta = normalize_hu(cta_raw, d["cta_hu"]) * mask
        origins = tile_origins(mask, d["volume_size"], d["roi_margin"], d["tile_overlap"])
        index["cases"].append({**row, "shape_xyz": reference.shape,
                                "predicted_mask": str(mask_file.relative_to(output))})
        for tile_number, origin in enumerate(origins):
            file = case_dir / f"tile_{tile_number:04d}.npz"
            np.savez_compressed(file, ncct=extract_tile(ncct, origin, d["volume_size"]),
                                cta=extract_tile(cta, origin, d["volume_size"]),
                                mask=extract_tile(mask, origin, d["volume_size"]))
            index["tiles"].append({"patient_id": identifier, "split": row["split"],
                                    "label": row["label"], "origin_xyz": origin,
                                    "file": str(file.relative_to(output))})
    (output / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return output / "index.json"


def load_index(path, kind):
    path = Path(path).resolve()
    index = json.loads(path.read_text(encoding="utf-8"))
    if index["kind"] != kind:
        raise ValueError(f"Expected {kind} cache, got {index['kind']}")
    seen = set()
    case_splits = {row["patient_id"]: row["split"] for row in index["cases"]}
    if len(case_splits) != len(index["cases"]):
        raise ValueError("Duplicate cached patient_id")
    for case in index["cases"]:
        identifier = case["patient_id"]
        if not identifier or any(c in identifier for c in '/\\:') or identifier in {".", ".."}:
            raise ValueError("Unsafe cached patient_id")
        require_training_split(case["split"])
        mask_path = (path.parent / case["predicted_mask"]).resolve()
        if not mask_path.is_relative_to(path.parent) or not mask_path.is_file():
            raise ValueError("Unsafe/missing predicted-mask cache path")
    for tile in index["tiles"]:
        require_training_split(tile["split"])
        if case_splits.get(tile["patient_id"]) != tile["split"]:
            raise ValueError("Case/tile partition mismatch")
        file = (path.parent / tile["file"]).resolve()
        if not file.is_relative_to(path.parent) or str(file) in seen:
            raise ValueError("Unsafe/duplicate cache path")
        seen.add(str(file))
        if not file.is_file():
            raise FileNotFoundError(file)
    if not index["tiles"]:
        raise ValueError("Empty cache")
    if set(case_splits) != {tile["patient_id"] for tile in index["tiles"]}:
        raise ValueError("Every cached case must have tiles")
    return index, path.parent


@torch.no_grad()
def synthesize(index_path, config, output_dir, progress=True):
    source, root = load_index(index_path, "paired_generation")
    if source["config"]["diffusion"] != config["diffusion"]:
        raise ValueError("Preparation/synthesis diffusion configs differ; prepare with this config first")
    output = new_output(output_dir)
    device = resolve_device(config["device"])
    model = diffusion_model(config["diffusion"])
    load_weights(model, config["diffusion_checkpoint"], "generation")
    model.to(device).eval()
    diffusion = ConditionalDiffusion(config["diffusion"])
    groups = defaultdict(list)
    for tile in source["tiles"]:
        groups[tile["patient_id"]].append(tile)
    index = {**source, "kind": "final_syncta_detection", "tiles": [], "cases": [],
             "config": config, "generation_sha256": file_digest(config["diffusion_checkpoint"])}
    for case in source["cases"]:
        identifier = case["patient_id"]
        case_dir = output / identifier
        case_dir.mkdir()
        total = np.zeros(case["shape_xyz"], dtype=np.float32)
        counts = np.zeros_like(total)
        mask_reference, mask = load_volume(root / case["predicted_mask"])
        reference, _ = load_volume(case["ncct"])
        require_same_grid(reference, mask_reference, "cached predicted mask")
        for tile in groups[identifier]:
            with np.load(root / tile["file"]) as cached:
                condition = xyz_to_tensor(cached["ncct"], device)
            prediction = diffusion.sample(model, condition, progress=progress)
            add_tile(total, counts, tensor_to_xyz(prediction), tile["origin_xyz"])
        final = np.divide(total, counts, out=np.zeros_like(total), where=counts > 0) * mask
        if np.any((mask > 0) & (counts == 0)):
            raise RuntimeError("Incomplete aortic tile coverage")
        save_volume(final, reference, case_dir / "syncta_normalized.nii.gz")
        save_volume(mask, reference, case_dir / "predicted_aorta.nii.gz", np.uint8)
        index["cases"].append({**case, "predicted_mask": str((case_dir / "predicted_aorta.nii.gz").relative_to(output))})
        for number, tile in enumerate(groups[identifier]):
            with np.load(root / tile["file"]) as cached:
                ncct, tile_mask = cached["ncct"], cached["mask"]
            file = case_dir / f"tile_{number:04d}.npz"
            np.savez_compressed(file, ncct=ncct, mask=tile_mask,
                                syncta=extract_tile(final, tile["origin_xyz"], config["diffusion"]["volume_size"]))
            index["tiles"].append({**tile, "file": str(file.relative_to(output))})
    (output / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return output / "index.json"
