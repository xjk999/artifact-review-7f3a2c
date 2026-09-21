import csv
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

from adnet.data import load_config, load_manifest, require_training_split, validate_training_config
from adnet.diffusion import ConditionalDiffusion, WaveletCodec
from adnet.geometry import (add_tile, extract_tile, normalize_hu, require_same_grid,
                            save_volume, tile_origins, xyz_to_tensor, tensor_to_xyz)
from adnet.models import PIPELINE_VERSION, diffusion_model, load_weights, segmentation_model
from adnet.pipeline import ADNet, Segmenter, new_output

torch.set_num_threads(2)


def base_config():
    return load_config(Path(__file__).resolve().parents[1] / "config.json")


class DataTests(unittest.TestCase):
    def test_manifest_disjoint_ids_and_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            fields = ["patient_id", "center", "split", "ncct", "cta", "mask", "label"]
            rows = [dict(patient_id="A", center="1", split="train", ncct="A.nii.gz", label="1"),
                    dict(patient_id="B", center="1", split="internal_test", ncct="B.nii.gz", label="0")]
            def write():
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
            write()
            self.assertEqual(len(load_manifest(path, check_files=False)), 2)
            rows[1]["patient_id"] = "A"
            write()
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_manifest(path, check_files=False)
            rows[1]["patient_id"] = "B"
            rows[1]["ncct"] = "A.nii.gz"
            write()
            with self.assertRaisesRegex(ValueError, "Repeated"):
                load_manifest(path, check_files=False)

    def test_config_relative_paths_and_network_size(self):
        config = load_config(Path(__file__).resolve().parents[1] / "config.json")
        self.assertTrue(Path(config["segmentation_checkpoint"]).is_absolute())
        self.assertEqual(config["threshold"], 0.65)

    def test_final_mode_and_locked_parameters(self):
        config = base_config()
        self.assertEqual(config["training"]["mode"], "final")
        for split in ("val", "internal_test", "external_test"):
            with self.assertRaises(ValueError):
                require_training_split(split)
        config["training"]["generation"]["steps"] = 0
        with self.assertRaises(ValueError):
            validate_training_config(config)

    def test_cli_training_parameters_only_from_config(self):
        from adnet_cli import parser
        args = parser().parse_args(["train-generation", "--index", "cache.json", "--output", "run"])
        self.assertFalse(hasattr(args, "steps"))
        args = parser().parse_args(["train-segmentation", "--manifest", "manifest.csv", "--output", "run"])
        self.assertFalse(hasattr(args, "epochs"))

    def test_manifest_rejects_validation_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            path.write_text("patient_id,center,split,ncct\nA,1,val,A.nii.gz\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown split"):
                load_manifest(path, check_files=False)

    def test_constant_hu_is_finite(self):
        result = normalize_hu(np.ones((4, 5, 6)), [0, 200])
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(float(result.sum()), 0)

    def test_original_grid_and_axis_roundtrip(self):
        array = np.arange(3 * 4 * 5).reshape(3, 4, 5).astype(np.float32)
        np.testing.assert_array_equal(tensor_to_xyz(xyz_to_tensor(array, "cpu")), array)
        affine = np.array([[0, -0.8, 0, 20], [0.7, 0, 0, -10], [0, 0, 1.5, 3], [0, 0, 0, 1]])
        reference = nib.Nifti1Image(array, affine)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "saved.nii.gz"
            save_volume(array, reference, path)
            saved = nib.load(path)
            self.assertEqual(saved.shape, array.shape)
            np.testing.assert_allclose(saved.affine, affine)
            different = nib.Nifti1Image(array, np.eye(4))
            with self.assertRaisesRegex(ValueError, "registered"):
                require_same_grid(reference, different, "CTA")

    def test_long_aorta_tiles_cover_without_resizing(self):
        mask = np.zeros((20, 18, 70), dtype=np.uint8)
        mask[4:18, 3:17, 2:68] = 1
        total, counts = np.zeros(mask.shape), np.zeros(mask.shape)
        origins = tile_origins(mask, 16, margin=1, overlap=0.25)
        for origin in origins:
            add_tile(total, counts, extract_tile(mask, origin, 16), origin)
        self.assertFalse(np.any((mask > 0) & (counts == 0)))
        combined = np.divide(total, counts, out=np.zeros_like(total), where=counts > 0)
        np.testing.assert_array_equal(combined, mask)
        with self.assertRaisesRegex(ValueError, "empty"):
            tile_origins(mask * 0, 16)

    def test_refuse_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outputs"
            new_output(path)
            (path / "existing.txt").write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                new_output(path)


class DiffusionTests(unittest.TestCase):
    def config(self, sampler="ddim", steps=2):
        return {"diffusion_steps": 1000, "sampler": sampler, "sampling_steps": steps, "clip_denoised": True}

    def test_wavelet_cpu_reconstruction_and_gradient(self):
        codec = WaveletCodec()
        source = torch.rand(1, 1, 8, 8, 8, requires_grad=True)
        restored = codec.decode(codec.encode(source))
        torch.testing.assert_close(restored, source, atol=1e-6, rtol=1e-6)
        restored.sum().backward()
        torch.testing.assert_close(source.grad, torch.ones_like(source))
        constant = codec.encode(torch.ones(1, 1, 8, 8, 8))
        torch.testing.assert_close(constant[:, 0], torch.full_like(constant[:, 0], np.sqrt(8) / 3))
        self.assertLess(float(constant[:, 1:].abs().max()), 1e-6)

    def test_wavelet_gpu_device(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        codec = WaveletCodec()
        source = torch.rand(1, 1, 8, 8, 8, device="cuda:0")
        torch.testing.assert_close(codec.decode(codec.encode(source)), source, atol=1e-6, rtol=1e-6)

    def test_q_sample_and_full_ddpm_schedule(self):
        diffusion = ConditionalDiffusion(self.config())
        target, noise = torch.zeros(1, 8, 2, 2, 2), torch.ones(1, 8, 2, 2, 2)
        time = torch.tensor([10])
        expected = torch.full_like(target, float((1 - diffusion.alpha[10]).sqrt()))
        torch.testing.assert_close(diffusion.q_sample(target, time, noise), expected)
        betas = np.linspace(1e-4, 0.02, 1000)
        alpha = np.cumprod(1 - betas)
        previous = np.append(1, alpha[:-1])
        np.testing.assert_allclose(diffusion.coef1.numpy(), betas * np.sqrt(previous) / (1 - alpha))
        np.testing.assert_allclose(diffusion.variance.numpy()[1:], betas[1:])

    def test_complete_sampling_oracle(self):
        for sampler, steps in (("ddim", 2), ("ddpm", 1000)):
            diffusion = ConditionalDiffusion(self.config(sampler, steps))
            codec = diffusion.codec
            class Oracle(torch.nn.Module):
                def forward(self, wavelets, time):
                    return codec.encode(torch.full((1, 1, 4, 4, 4), 0.4)), torch.tensor([[0.9]])
            result = diffusion.sample(Oracle(), torch.ones(1, 1, 4, 4, 4), progress=False)
            torch.testing.assert_close(result, torch.full_like(result, 0.4), atol=1e-5, rtol=1e-5)


class PipelineTests(unittest.TestCase):
    def test_three_ordered_stages_ncct_only_and_geometry(self):
        events = []
        mask = np.zeros((20, 24, 40), dtype=np.uint8)
        mask[3:17, 4:20, 2:38] = 1
        class FakeSegmenter:
            def segment(self, path):
                events.append("segmentation")
                return mask
        class FakeModel:
            def to(self, device):
                return self
            def cpu(self):
                return self
        class FakeDiffusion:
            codec = None
            def sample(self, model, ncct, progress):
                events.append("generation")
                return ncct * 0.5
        def fake_detect(model, codec, synthetic, ncct):
            events.append("detection")
            self.assertTrue(torch.all(synthetic <= ncct))
            return torch.tensor([[0.7]])
        pipeline = ADNet.__new__(ADNet)
        pipeline.config = {"threshold": 0.65, "diffusion": {"volume_size": 16, "roi_margin": 1,
                           "tile_overlap": 0.25, "case_aggregation": "max", "ncct_hu": [0, 200]}}
        pipeline.device = torch.device("cpu")
        pipeline.segmenter, pipeline.diffusion = FakeSegmenter(), FakeDiffusion()
        pipeline.generator = pipeline.detector = FakeModel()
        pipeline.weights = {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = nib.Nifti1Image(np.arange(mask.size, dtype=np.float32).reshape(mask.shape) % 200,
                                   np.diag([-0.7, 0.8, 1.25, 1]))
            source = root / "NCCT_only.nii.gz"
            nib.save(image, source)
            with patch("adnet.pipeline.detect_final", side_effect=fake_detect):
                result = pipeline.infer_case(source, "case", root / "output", progress=False)
            self.assertEqual(events[0], "segmentation")
            self.assertGreater(events.count("generation"), 1)
            self.assertGreater(events.index("detection"), max(i for i, e in enumerate(events) if e == "generation"))
            self.assertEqual(result["prediction"], 1)
            self.assertEqual(result["threshold"], 0.65)
            for value, expected in ((0.60, 0), (0.65, 1), (0.70, 1)):
                with self.subTest(probability=value), patch("adnet.pipeline.detect_final",
                        return_value=torch.tensor([[value]], dtype=torch.float64)):
                    updated = pipeline.infer_case(source, "case", root / f"threshold_{value}", progress=False)
                self.assertEqual(updated["prediction"], expected)
                self.assertEqual(updated["probability"], value)
                self.assertEqual(updated["threshold"], 0.65)
            for name in ("aorta_mask.nii.gz", "syncta_normalized.nii.gz"):
                saved = nib.load(root / "output" / name)
                self.assertEqual(saved.shape, mask.shape)
                np.testing.assert_allclose(saved.affine, image.affine)
            syn = nib.load(root / "output" / "syncta_normalized.nii.gz").get_fdata()
            self.assertEqual(float(syn[mask == 0].sum()), 0)

    def test_threshold_change_does_not_require_retraining(self):
        config = base_config()
        config.update(device="cpu", diffusion_checkpoint="gen.pt", detection_checkpoint="det.pt",
                      segmentation_checkpoint="seg.pt")
        trained = copy.deepcopy(config)
        trained["threshold"] = 0.5
        metadata = {"pipeline_version": PIPELINE_VERSION, "generation_sha256": "genhash",
                    "segmentation_sha256": "seghash", "integrated_config": trained}
        with patch("adnet.pipeline.Segmenter"), patch("adnet.pipeline.diffusion_model",
                side_effect=lambda settings: torch.nn.Identity()), patch("adnet.pipeline.load_weights",
                side_effect=[{}, metadata]), patch("adnet.cache.file_digest",
                side_effect=lambda path: "genhash" if path == "gen.pt" else "seghash"), \
                patch("adnet.pipeline.ConditionalDiffusion"):
            pipeline = ADNet(config)
        self.assertEqual(pipeline.config["threshold"], 0.65)
        changed = copy.deepcopy(config)
        changed["diffusion"]["tile_overlap"] = 0.50
        with patch("adnet.pipeline.Segmenter"), patch("adnet.pipeline.diffusion_model",
                side_effect=lambda settings: torch.nn.Identity()), patch("adnet.pipeline.load_weights",
                side_effect=[{}, metadata]), patch("adnet.cache.file_digest",
                side_effect=lambda path: "genhash" if path == "gen.pt" else "seghash"):
            with self.assertRaisesRegex(ValueError, "preprocessing/sampling"):
                ADNet(changed)

    def test_real_network_forward_and_weight_roles(self):
        config = {"volume_size": 128, "model_channels": 4, "num_groups": 4}
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = diffusion_model(config).to(device).eval()
        with torch.no_grad():
            prediction, probability = model(torch.zeros(1, 16, 64, 64, 64, device=device),
                                            torch.zeros(1, dtype=torch.long, device=device))
        self.assertEqual(prediction.shape, (1, 8, 64, 64, 64))
        self.assertEqual(probability.shape, (1, 1))
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "generator.pt"
            state = {k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("dissection_head.")}
            torch.save(state, file)
            fresh = diffusion_model(config)
            self.assertTrue(load_weights(fresh, file, "generation")["missing_head"])
            with self.assertRaises(ValueError):
                load_weights(fresh, file, "detection")

    def test_real_segmentation_network_forward(self):
        model = segmentation_model({"init_filters": 8}).eval()
        with torch.no_grad():
            result = model(torch.rand(1, 1, 32, 32, 32))
        self.assertEqual(result.shape, (1, 2, 32, 32, 32))

    def test_segmentation_training_and_original_grid_restoration(self):
        from adnet.training import train_segmentation
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = base_config()
            config.update({"device": "cpu", "seed": 42, "segmentation_checkpoint": str(root / "placeholder.pt"),
                      "segmentation": {"init_filters": 8, "spacing": [1, 1, 1],
                                       "intensity_min": -150, "intensity_max": 250,
                                       "roi_size": [32, 32, 32], "overlap": 0.25, "sw_batch_size": 1}})
            config["training"]["segmentation"].update(epochs=1, learning_rate=0.0003)
            affine = np.diag([-0.8, 0.9, 1.2, 1])
            image = np.zeros((32, 32, 32), dtype=np.float32)
            image[8:24, 8:24, 8:24] = 100
            mask = (image > 0).astype(np.uint8)
            rows = []
            for split in ("train", "internal_test"):
                ncct, label = root / f"{split}_ncct.nii.gz", root / f"{split}_mask.nii.gz"
                nib.save(nib.Nifti1Image(image, affine), ncct)
                nib.save(nib.Nifti1Image(mask, affine), label)
                rows.append(dict(patient_id=split, center="1", split=split, ncct=str(ncct), mask=str(label)))
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["patient_id", "center", "split", "ncct", "mask"])
                writer.writeheader()
                writer.writerows(rows)
            # Test files may be absent during training.
            Path(rows[1]["ncct"]).unlink()
            Path(rows[1]["mask"]).unlink()
            checkpoint_path = train_segmentation(manifest, config, root / "segmentation")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["selection"], "fixed final epoch")
            self.assertEqual(checkpoint_path.name, "segmentation_final.pt")
            self.assertFalse((root / "segmentation" / "best_model.pt").exists())
            self.assertNotIn("best_dice", checkpoint)
            self.assertEqual(checkpoint["optimizer"]["param_groups"][0]["initial_lr"], 0.0003)
            config["segmentation_checkpoint"] = str(checkpoint_path)
            segmenter = Segmenter(config)
            class Foreground(torch.nn.Module):
                def forward(self, value):
                    return torch.cat([torch.zeros_like(value), torch.ones_like(value)], dim=1)
            # Known logits isolate RAS/spacing -> original-grid restoration from model accuracy.
            segmenter.model = Foreground()
            restored = segmenter.segment(rows[0]["ncct"])
            self.assertEqual(restored.shape, image.shape)
            self.assertTrue(restored.any())

    def test_one_step_generation_synthesis_detection_training(self):
        from adnet.cache import file_digest, synthesize
        from adnet.training import train_wavelet
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "paired"
            cache.mkdir()
            config = base_config()
            config.update({"device": "cuda:0" if torch.cuda.is_available() else "cpu", "seed": 42,
                      "threshold": 0.65, "segmentation_checkpoint": str(root / "seg.pt"),
                      "diffusion": {"volume_size": 128, "model_channels": 4, "num_groups": 4,
                                    "diffusion_steps": 1000, "sampling_steps": 2, "sampler": "ddim",
                                    "clip_denoised": True}})
            for stage in ("generation", "detection"):
                config["training"][stage].update(steps=1, save_every=1, learning_rate=0.00002, batch_size=2)
            (root / "seg.pt").write_bytes(b"metadata-only segmentation file")
            shape = (128, 128, 128)
            mask = np.ones(shape, dtype=np.uint8)
            reference = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4))
            nib.save(reference, cache / "NCCT.nii.gz")
            save_volume(mask, reference, cache / "mask.nii.gz", np.uint8)
            np.savez_compressed(cache / "tile.npz", ncct=np.full(shape, 0.2, dtype=np.float32),
                                cta=np.full(shape, 0.4, dtype=np.float32), mask=mask)
            index = {"kind": "paired_generation", "config": config, "manifest_sha256": "test-only",
                     "segmentation_sha256": file_digest(root / "seg.pt"),
                     "cases": [{"patient_id": "A", "split": "train", "shape_xyz": shape,
                                 "ncct": str(cache / "NCCT.nii.gz"), "predicted_mask": "mask.nii.gz"}],
                     "tiles": [{"patient_id": "A", "split": "train", "label": 1,
                                 "origin_xyz": [0, 0, 0], "file": "tile.npz"}]}
            index_file = cache / "index.json"
            index_file.write_text(json.dumps(index), encoding="utf-8")
            generator = train_wavelet(index_file, config, root / "generator", "generation")
            generator_state = torch.load(generator, map_location="cpu", weights_only=True)["model"]
            config["diffusion_checkpoint"] = str(generator)
            synthetic_index = synthesize(index_file, config, root / "synthetic", progress=False)
            detector = train_wavelet(synthetic_index, config, root / "detector", "detection")
            checkpoint = torch.load(detector, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["pipeline_version"], PIPELINE_VERSION)
            self.assertEqual(checkpoint["generation_sha256"], file_digest(generator))
            self.assertEqual(checkpoint["optimizer"]["param_groups"][0]["lr"], 0.00002)
            self.assertEqual(checkpoint["selection"], "fixed final step")
            changed_head = False
            for key, before in generator_state.items():
                after = checkpoint["model"][key]
                if key.startswith("dissection_head."):
                    changed_head |= not torch.equal(before, after)
                else:
                    self.assertTrue(torch.equal(before, after), f"Frozen backbone changed: {key}")
            self.assertTrue(changed_head)


if __name__ == "__main__":
    unittest.main()
