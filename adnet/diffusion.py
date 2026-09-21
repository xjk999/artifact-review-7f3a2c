"""Clean START_X conditional diffusion with full DDPM coefficients."""
import numpy as np
import torch
from tqdm.auto import tqdm

from .vendor.DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D


class WaveletCodec:
    def __init__(self):
        self.dwt, self.idwt = DWT_3D("haar"), IDWT_3D("haar")

    def encode(self, image):
        bands = list(self.dwt(image))
        bands[0] = bands[0] / 3.0
        return torch.cat(bands, dim=1)

    def decode(self, wavelets):
        bands = list(wavelets.split(1, dim=1))
        bands[0] = bands[0] * 3.0
        return self.idwt(*bands)


class ConditionalDiffusion:
    def __init__(self, config, codec=None):
        self.config = config
        self.codec = codec or WaveletCodec()
        self.steps = int(config["diffusion_steps"])
        betas = np.linspace(1000 / self.steps * 1e-4, 1000 / self.steps * 0.02,
                            self.steps, dtype=np.float64)
        if self.steps < 2 or np.any(betas <= 0) or np.any(betas >= 1):
            raise ValueError("Invalid linear beta schedule; use >= 21 diffusion_steps")
        alphas = np.cumprod(1.0 - betas)
        previous = np.append(1.0, alphas[:-1])
        posterior_variance = betas * (1.0 - previous) / (1.0 - alphas)
        self.alpha = torch.tensor(alphas, dtype=torch.float64)
        self.coef1 = torch.tensor(betas * np.sqrt(previous) / (1.0 - alphas), dtype=torch.float64)
        self.coef2 = torch.tensor((1.0 - previous) * np.sqrt(1.0 - betas) / (1.0 - alphas), dtype=torch.float64)
        self.variance = torch.tensor(np.append(posterior_variance[1], betas[1:]), dtype=torch.float64)

    @staticmethod
    def at(values, timesteps, reference):
        return values.to(reference.device)[timesteps].float().view(-1, 1, 1, 1, 1)

    def q_sample(self, target, timesteps, noise=None):
        noise = torch.randn_like(target) if noise is None else noise
        alpha = self.at(self.alpha, timesteps, target)
        return alpha.sqrt() * target + (1 - alpha).sqrt() * noise

    def clean(self, prediction):
        if self.config.get("clip_denoised", True):
            return self.codec.encode(self.codec.decode(prediction).clamp(0, 1))
        return prediction

    @torch.no_grad()
    def sample(self, model, ncct, progress=True):
        condition = self.codec.encode(ncct)
        state = torch.randn_like(condition)
        mode = self.config["sampler"]
        requested = int(self.config["sampling_steps"])
        if mode == "ddpm":
            if requested != self.steps:
                raise ValueError("DDPM uses all training steps; use sampler=ddim for acceleration")
            indices = list(range(self.steps - 1, -1, -1))
        elif mode == "ddim":
            if not 2 <= requested <= self.steps:
                raise ValueError("DDIM sampling_steps must be in [2, diffusion_steps]")
            indices = np.linspace(self.steps - 1, 0, requested, dtype=int).tolist()
        else:
            raise ValueError("sampler must be ddpm or ddim")
        for position, index in enumerate(tqdm(indices, desc="Syn-CTA", disable=not progress)):
            times = torch.full((state.shape[0],), index, device=state.device, dtype=torch.long)
            prediction, _ = model(torch.cat([state, condition], dim=1), times)
            prediction = self.clean(prediction)
            if mode == "ddpm":
                state = self.at(self.coef1, times, state) * prediction + self.at(self.coef2, times, state) * state
                if index:
                    state = state + self.at(self.variance, times, state).sqrt() * torch.randn_like(state)
            else:
                alpha = self.at(self.alpha, times, state)
                epsilon = (state - alpha.sqrt() * prediction) / (1 - alpha).sqrt()
                next_index = indices[position + 1] if position + 1 < len(indices) else -1
                next_alpha = self.alpha[next_index].to(state.device).float() if next_index >= 0 else state.new_tensor(1.0)
                state = next_alpha.sqrt() * prediction + (1 - next_alpha).sqrt() * epsilon
        return self.codec.decode(state).clamp(0, 1)


def detect_final(model, codec, syncta, ncct):
    """Classify final Syn-CTA conditioned on NCCT at t=0."""
    times = torch.zeros(ncct.shape[0], dtype=torch.long, device=ncct.device)
    _, probability = model(torch.cat([codec.encode(syncta), codec.encode(ncct)], dim=1), times)
    return probability
