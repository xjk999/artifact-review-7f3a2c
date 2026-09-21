# AD-NET

AD-NET uses NCCT images for aorta segmentation, synthetic CTA generation and aortic dissection detection.

```text
NCCT -> aorta segmentation -> Syn-CTA generation -> AD detection
```

The command-line entry point is `adnet_cli.py`.

## Data

Patient information is supplied through a CSV manifest. See `examples/manifest.csv`.

| Column | Description |
|---|---|
| `patient_id` | Unique patient ID |
| `center` | Center ID |
| `split` | `train`, `internal_test` or `external_test` |
| `ncct` | NCCT image |
| `cta` | Registered CTA image; required for generation training |
| `mask` | Aorta mask; required for segmentation training |
| `label` | `0` for non-AD and `1` for AD |

All splits are patient-level. Hyperparameters and the classification threshold were selected by five-fold cross-validation on the training cohort. The internal and external test cohorts were used only for final evaluation.

Check a manifest before training:

```bash
python adnet_cli.py audit --manifest manifest.csv
```

## Setup

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python adnet_cli.py doctor
python -m unittest discover -s tests -v
```

The default configuration uses 256 x 256 x 256 patches and full 1000-step DDPM sampling. A CUDA GPU with sufficient memory is recommended.

## Configuration

Training and inference settings are in `config.json`. The released configuration uses the parameters selected during cross-validation, including a classification threshold of 0.65.

Model weights are not included. By default, the code reads:

```text
weights/segmentation_final.pt
weights/generation_final.pt
weights/detection_final.pt
```

Place the weights at these locations or update the three checkpoint paths in `config.json`. Run the following command to check them:

```bash
python adnet_cli.py check-weights
```

## Training

### 1. Aorta segmentation

```bash
python adnet_cli.py train-segmentation --manifest manifest.csv --output runs/segmentation
```

Copy `runs/segmentation/segmentation_final.pt` to `weights/segmentation_final.pt` before preparing the generation data.

### 2. Syn-CTA generation

```bash
python adnet_cli.py prepare --manifest manifest.csv --output prepared
python adnet_cli.py train-generation --index prepared/index.json --output runs/generation
```

The CTA images must be registered to the corresponding NCCT images. Copy `runs/generation/generation_final.pt` to `weights/generation_final.pt` before synthesizing the detector-training images.

### 3. AD detection

```bash
python adnet_cli.py synthesize --index prepared/index.json --output synthetic
python adnet_cli.py train-detection --index synthetic/index.json --output runs/detection
```

Copy `runs/detection/detection_final.pt` to `weights/detection_final.pt`.

Use `--resume CHECKPOINT` to resume training for any stage.

## Inference

Run the internal and external test cohorts:

```bash
python adnet_cli.py infer --manifest manifest.csv --split internal_test --output runs/internal_test
python adnet_cli.py infer --manifest manifest.csv --split external_test --output runs/external_test
```

Run one NCCT scan:

```bash
python adnet_cli.py infer --ncct /path/to/NCCT.nii.gz --patient-id case001 --output runs/case001
```

The output directory contains the aorta mask, normalized NCCT, normalized Syn-CTA, case-level probability and binary prediction. Syn-CTA values are normalized to `[0, 1]` and are not calibrated HU values.
