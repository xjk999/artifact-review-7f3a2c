import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import (Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged,
                              Orientationd, ScaleIntensityRanged, Spacingd)
from nibabel.processing import resample_from_to

from .diffusion import ConditionalDiffusion, detect_final
from .geometry import (add_tile, extract_tile, load_volume, normalize_hu,
                       save_volume, tensor_to_xyz, tile_origins, xyz_to_tensor)
from .models import PIPELINE_VERSION, diffusion_model, load_weights, segmentation_model


def resolve_device(name):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use device=cpu or select a CUDA device")
    return device


def new_output(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


class Segmenter:
    def __init__(self, config):
        self.config, self.device = config["segmentation"], resolve_device(config["device"])
        self.model = segmentation_model(self.config)
        self.weights = load_weights(self.model, config["segmentation_checkpoint"], "segmentation")
        self.model.eval()
        self.transforms = Compose([
            LoadImaged(keys="image"), EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", axcodes="RAS"),
            Spacingd(keys="image", pixdim=tuple(self.config["spacing"]), mode="bilinear"),
            ScaleIntensityRanged(keys="image", a_min=self.config["intensity_min"],
                                 a_max=self.config["intensity_max"], b_min=0, b_max=1, clip=True),
            EnsureTyped(keys="image", dtype=torch.float32),
        ])

    @torch.no_grad()
    def segment(self, ncct_path):
        original, _ = load_volume(ncct_path)
        transformed = self.transforms({"image": str(ncct_path)})["image"]
        self.model.to(self.device)
        logits = sliding_window_inference(
            transformed.unsqueeze(0).to(self.device), roi_size=tuple(self.config["roi_size"]),
            sw_batch_size=self.config["sw_batch_size"], predictor=self.model,
            overlap=self.config["overlap"], mode="gaussian", sw_device=self.device,
            device=torch.device("cpu"),
        )
        prediction = logits.argmax(dim=1)[0].numpy().astype(np.uint8)
        processed = nib.Nifti1Image(prediction, transformed.affine.cpu().numpy())
        restored = resample_from_to(processed, (original.shape, original.affine), order=0,
                                    mode="constant", cval=0)
        mask = (np.asanyarray(restored.dataobj) == 1).astype(np.uint8)
        self.model.cpu()
        if not mask.any():
            raise ValueError(f"Empty predicted aorta: {ncct_path}")
        return mask


class ADNet:
    """Three ordered stages, with CPU offloading between large networks."""
    def __init__(self, config):
        self.config, self.device = config, resolve_device(config["device"])
        self.segmenter = Segmenter(config)
        self.generator = diffusion_model(config["diffusion"])
        generation_info = load_weights(self.generator, config["diffusion_checkpoint"], "generation")
        if config["diffusion_checkpoint"] == config["detection_checkpoint"]:
            self.detector = self.generator
            detection_info = load_weights(self.detector, config["detection_checkpoint"], "detection")
        else:
            self.detector = diffusion_model(config["diffusion"])
            detection_info = load_weights(self.detector, config["detection_checkpoint"], "detection")
        if detection_info["pipeline_version"] != PIPELINE_VERSION:
            raise ValueError("Detector checkpoint is incompatible with this pipeline")
        from .cache import file_digest
        if detection_info["generation_sha256"] != file_digest(config["diffusion_checkpoint"]):
            raise ValueError("Inference generator differs from detector's Syn-CTA cache generator")
        if detection_info["segmentation_sha256"] != file_digest(config["segmentation_checkpoint"]):
            raise ValueError("Inference SegResNet differs from detector's ROI preparation model")
        trained_config = detection_info["integrated_config"]
        if trained_config["diffusion"] != config["diffusion"]:
            raise ValueError("Detector preprocessing/sampling config changed since training")
        self.generator.eval()
        self.detector.eval()
        self.diffusion = ConditionalDiffusion(config["diffusion"])
        self.weights = {"segmentation": self.segmenter.weights,
                        "generation": generation_info, "detection": detection_info}

    @torch.no_grad()
    def infer_case(self, ncct_path, patient_id, output_dir, progress=True):
        output = new_output(output_dir)
        reference, raw = load_volume(ncct_path)
        # Segment the aorta and restore the mask to the original NCCT grid.
        mask = self.segmenter.segment(ncct_path)
        save_volume(mask, reference, output / "aorta_mask.nii.gz", np.uint8)
        d = self.config["diffusion"]
        ncct = normalize_hu(raw, d["ncct_hu"]) * mask
        save_volume(ncct, reference, output / "aorta_ncct_normalized.nii.gz")
        origins = tile_origins(mask, d["volume_size"], d["roi_margin"], d["tile_overlap"])
        # Generate each tile and blend them into one Syn-CTA volume.
        total, counts = np.zeros_like(ncct), np.zeros_like(ncct)
        self.generator.to(self.device)
        for index, origin in enumerate(origins, 1):
            print(f"[{patient_id}] generation tile {index}/{len(origins)}", flush=True)
            condition = xyz_to_tensor(extract_tile(ncct, origin, d["volume_size"]), self.device)
            generated = self.diffusion.sample(self.generator, condition, progress=progress)
            add_tile(total, counts, tensor_to_xyz(generated), origin)
        syncta = np.divide(total, counts, out=np.zeros_like(total), where=counts > 0) * mask
        if np.any((mask > 0) & (counts == 0)):
            raise RuntimeError("Incomplete aortic tile coverage")
        save_volume(syncta, reference, output / "syncta_normalized.nii.gz")
        self.generator.cpu()
        # Run detection using the NCCT and the assembled Syn-CTA volume.
        probabilities = []
        self.detector.to(self.device)
        for origin in origins:
            synthetic = xyz_to_tensor(extract_tile(syncta, origin, d["volume_size"]), self.device)
            condition = xyz_to_tensor(extract_tile(ncct, origin, d["volume_size"]), self.device)
            probability = detect_final(self.detector, self.diffusion.codec, synthetic, condition)
            probabilities.append(float(probability.reshape(-1)[0].cpu()))
        self.detector.cpu()
        reducer = max if d["case_aggregation"] == "max" else lambda values: float(np.mean(values))
        probability = reducer(probabilities)
        result = {"patient_id": patient_id, "probability": probability,
                  "prediction": int(probability >= self.config["threshold"]),
                  "threshold": self.config["threshold"], "tile_probabilities": probabilities,
                  "case_aggregation": d["case_aggregation"], "tile_origins_xyz": origins,
                  "ncct_path": str(Path(ncct_path).resolve()), "original_shape_xyz": reference.shape,
                  "pipeline_version": PIPELINE_VERSION,
                  "weights": self.weights,
                  "syncta_units": "per-volume normalized [0,1], NOT calibrated HU"}
        (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
