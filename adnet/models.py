import torch
from monai.networks.nets import SegResNet

from .vendor.guided_diffusion.wunet import WavUNetModel

PIPELINE_VERSION = "segresnet-syncta-final-detection-v1"


def segmentation_model(config):
    return SegResNet(spatial_dims=3, in_channels=1, out_channels=2,
                     init_filters=config["init_filters"], blocks_down=(1, 2, 2, 4),
                     blocks_up=(1, 1, 1), dropout_prob=config.get("dropout", 0.1))


def diffusion_model(config):
    model = WavUNetModel(image_size=config["volume_size"], in_channels=16,
                        model_channels=config["model_channels"], out_channels=8,
                        num_res_blocks=2, attention_resolutions=(),
                        channel_mult=(1, 2, 2, 4, 4, 4), dims=3,
                        num_groups=config["num_groups"], use_scale_shift_norm=False,
                        bottleneck_attention=False, additive_skips=False,
                        resample_2d=False, use_freq=True, dropout=config.get("dropout", 0.0))
    for layer in model.dissection_head.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = config.get("detection_dropout", 0.5)
    return model


def load_weights(model, path, role):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint)
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    missing = set(incompatible.missing_keys)
    allowed = {key for key in model.state_dict() if key.startswith("dissection_head.")}
    if incompatible.unexpected_keys or (missing and not (role == "generation" and missing <= allowed)):
        raise ValueError(f"Incompatible {role} weights {path}: missing={sorted(missing)}, "
                         f"unexpected={incompatible.unexpected_keys}")
    return {"role": role, "path": str(path), "missing_head": bool(missing),
            "pipeline_version": checkpoint.get("pipeline_version"),
            "generation_sha256": checkpoint.get("generation_sha256"),
            "segmentation_sha256": checkpoint.get("segmentation_sha256"),
            "integrated_config": checkpoint.get("integrated_config")}
