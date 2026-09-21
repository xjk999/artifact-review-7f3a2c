"""Fixed-configuration stage-wise training."""
import json
import random

import nibabel as nib
import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.losses import DiceCELoss
from monai.transforms import (Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged,
                              MapLabelValued, Orientationd, RandCropByPosNegLabeld,
                              RandFlipd, RandShiftIntensityd, ScaleIntensityRanged,
                              Spacingd, SpatialPadd)
from monai.utils import set_determinism

from .cache import file_digest, load_index
from .data import load_manifest, manifest_digest, select, validate_training_config
from .diffusion import ConditionalDiffusion, detect_final
from .geometry import require_same_grid, xyz_to_tensor
from .models import PIPELINE_VERSION, diffusion_model, load_weights, segmentation_model
from .pipeline import new_output, resolve_device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_optimizer(parameters, settings):
    return torch.optim.AdamW(parameters, lr=settings["learning_rate"], weight_decay=settings["weight_decay"],
                             betas=tuple(settings["betas"]), eps=settings["eps"])


def seg_transforms(config, settings):
    transforms = [
        LoadImaged(keys=["image", "label"]), EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=tuple(config["spacing"]), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys="image", a_min=config["intensity_min"], a_max=config["intensity_max"],
                             b_min=0, b_max=1, clip=True),
        MapLabelValued(keys="label", orig_labels=[0, 1, 2], target_labels=[0, 1, 0]),
    ]
    if settings:
        transforms += [SpatialPadd(keys=["image", "label"], spatial_size=tuple(config["roi_size"])),
                       RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                                             spatial_size=tuple(config["roi_size"]),
                                             pos=settings["crop_positive"], neg=settings["crop_negative"],
                                             num_samples=settings["samples_per_case"], image_key="image",
                                             image_threshold=settings["image_threshold"])]
        transforms += [RandFlipd(keys=["image", "label"], prob=p, spatial_axis=axis)
                       for axis, p in enumerate(settings["flip_probabilities"]) if p]
        transforms += [RandShiftIntensityd(keys="image", offsets=settings["intensity_shift_offset"],
                                           prob=settings["intensity_shift_probability"])]
    return Compose(transforms + [EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.long))])


def train_segmentation(manifest, config, output_dir, resume=None):
    validate_training_config(config)
    settings = config["training"]["segmentation"]
    epochs = settings["epochs"]
    set_determinism(config["seed"])
    rows = load_manifest(manifest, check_files=False)
    train_rows = select(rows, "train", ("mask",))
    for row in train_rows:
        reference, mask = nib.load(row["ncct"]), nib.load(row["mask"])
        require_same_grid(reference, mask, "manual segmentation mask")
    output = new_output(output_dir)
    device = resolve_device(config["device"])
    s = config["segmentation"]
    as_samples = lambda cases: [{"image": r["ncct"], "label": r["mask"]} for r in cases]
    train_loader = DataLoader(Dataset(as_samples(train_rows), seg_transforms(s, settings)),
                              batch_size=settings["batch_size"], shuffle=True, num_workers=settings["num_workers"])
    model = segmentation_model(s).to(device)
    optimizer = make_optimizer(model.parameters(), settings)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=settings["min_learning_rate"])
    criterion = DiceCELoss(to_onehot_y=True, softmax=True, include_background=settings["include_background"],
                          lambda_dice=settings["dice_weight"], lambda_ce=settings["ce_weight"],
                          smooth_nr=settings["smooth_nr"], smooth_dr=settings["smooth_dr"])
    start = 0
    digest = manifest_digest(manifest)
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if (checkpoint.get("stage") != "segmentation" or checkpoint.get("manifest_sha256") != digest
                or checkpoint.get("integrated_config") != config):
            raise ValueError("Resume requires identical stage, manifest and config")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start = checkpoint["epoch"]
    if start >= epochs:
        raise ValueError("Resume checkpoint already reached requested epochs")
    (output / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for epoch in range(start + 1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            image, label = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(image), label)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite segmentation loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "epoch": epoch,
                      "manifest_sha256": digest, "integrated_config": config,
                      "selection": "fixed final epoch",
                      "stage": "segmentation"}
        if epoch % settings["save_every"] == 0 or epoch == epochs:
            torch.save(checkpoint, output / "latest.pt")
        record = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(record, flush=True)
    torch.save(checkpoint, output / "segmentation_final.pt")
    return output / "segmentation_final.pt"


def train_wavelet(index_path, config, output_dir, stage, resume=None):
    validate_training_config(config)
    if stage not in {"generation", "detection"}:
        raise ValueError("Invalid stage")
    settings = config["training"][stage]
    steps, save_every = settings["steps"], settings["save_every"]
    seed_everything(config["seed"])
    kind = "paired_generation" if stage == "generation" else "final_syncta_detection"
    index, root = load_index(index_path, kind)
    if index["config"]["diffusion"] != config["diffusion"]:
        raise ValueError("Training/cache diffusion configs differ")
    tiles = [t for t in index["tiles"] if t["split"] == "train"]
    if not tiles:
        raise ValueError("No train tiles; held-out cases cannot be used for training")
    if stage == "detection" and any(t["label"] not in {0, 1} for t in tiles):
        raise ValueError("AD labels required for every detection-training tile")
    device = resolve_device(config["device"])
    model = diffusion_model(config["diffusion"])
    initial = settings.get("initial_checkpoint") or (config["diffusion_checkpoint"] if stage == "detection" else None)
    if stage == "detection" and (not initial or file_digest(initial) != index["generation_sha256"]):
        raise ValueError("Detector backbone must use the generator that produced this cache")
    if initial:
        load_weights(model, initial, "generation")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("dissection_head.") if stage == "detection" else not name.startswith("dissection_head.")
    model.to(device)
    optimizer = make_optimizer([p for p in model.parameters() if p.requires_grad], settings)
    digest = file_digest(index_path)
    start = 0
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if (checkpoint.get("stage") != stage or checkpoint.get("cache_sha256") != digest
                or checkpoint.get("integrated_config") != config):
            raise ValueError("Resume requires identical stage, cache and config")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start = checkpoint["step"]
    if start >= steps:
        raise ValueError("Resume checkpoint already reached requested total steps")
    output = new_output(output_dir)
    (output / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    diffusion = ConditionalDiffusion(config["diffusion"])
    if stage == "generation":
        model.train()
    else:
        model.eval()
        model.dissection_head.train()
    for step in range(start + 1, steps + 1):
        batch_tiles = random.choices(tiles, k=settings["batch_size"])
        conditions, targets = [], []
        for tile in batch_tiles:
            with np.load(root / tile["file"]) as cached:
                conditions.append(xyz_to_tensor(cached["ncct"], device))
                targets.append(xyz_to_tensor(cached["cta" if stage == "generation" else "syncta"], device))
        ncct, target = torch.cat(conditions), torch.cat(targets)
        if stage == "generation":
            for axis, probability in enumerate(settings["flip_probabilities"], 2):
                if random.random() < probability:
                    ncct, target = ncct.flip(axis), target.flip(axis)
        optimizer.zero_grad(set_to_none=True)
        if stage == "generation":
            clean = diffusion.codec.encode(target)
            timesteps = torch.randint(diffusion.steps, (settings["batch_size"],), device=device)
            noisy = diffusion.q_sample(clean, timesteps)
            prediction, _ = model(torch.cat([noisy, diffusion.codec.encode(ncct)], 1), timesteps)
            loss = torch.mean((prediction - clean) ** 2)
        else:
            probability = detect_final(model, diffusion.codec, target, ncct)
            label = torch.tensor([[float(batch_tile["label"])] for batch_tile in batch_tiles], device=device)
            loss = torch.nn.functional.binary_cross_entropy(probability, label)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {stage} loss at step {step}")
        loss.backward()
        optimizer.step()
        if step == 1 or step % settings["log_every"] == 0 or step == steps:
            record = {"step": step, "stage": stage, "loss": float(loss.detach())}
            print(record, flush=True)
            with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        if step % save_every == 0 or step == steps:
            checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "stage": stage, "step": step, "cache_sha256": digest,
                          "manifest_sha256": index["manifest_sha256"], "integrated_config": config,
                          "selection": "fixed final step",
                          "initial_checkpoint": str(initial) if initial else None}
            if stage == "detection":
                checkpoint["pipeline_version"] = PIPELINE_VERSION
                checkpoint["generation_sha256"] = index["generation_sha256"]
                checkpoint["segmentation_sha256"] = index["segmentation_sha256"]
            torch.save(checkpoint, output / f"{stage}_{step:06d}.pt")
            if step == steps:
                torch.save(checkpoint, output / f"{stage}_final.pt")
    return output / f"{stage}_final.pt"
