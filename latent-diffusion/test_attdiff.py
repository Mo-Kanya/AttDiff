import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import numpy as np
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model(ckpt=None):
    config = OmegaConf.load("models/ldm/att_ffhq/config_v2.yaml")
    if ckpt:
        model = load_model_from_config(config, ckpt)
    else:
        model = instantiate_from_config(config.model)
    return model


model = get_model()
model = model.to("cuda")
sampler = DDIMSampler(model)

n_samples = 6
ddim_steps = 20
ddim_eta = 0.0
scale = 3.0  # for unconditional guidance

all_samples = list()

with torch.no_grad():
    with model.ema_scope():
        uc = model.get_learned_conditioning(
            {model.cond_stage_key: torch.rand(n_samples, 514).to(model.device)}
        )

        # xc = torch.tensor(n_samples * [class_label])
        xc = torch.rand(n_samples, 514)
        c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
        print(c.shape)

        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                         conditioning=c,
                                         batch_size=n_samples,
                                         shape=[3, 64, 64],
                                         verbose=False,
                                         unconditional_guidance_scale=scale,
                                         unconditional_conditioning=uc,
                                         eta=ddim_eta)

        x_samples_ddim = model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                                     min=0.0, max=1.0)
        all_samples.append(x_samples_ddim)

print("Finished sampling")
# # display as grid
# grid = torch.stack(all_samples, 0)
# grid = rearrange(grid, 'n b c h w -> (n b) c h w')
# grid = make_grid(grid, nrow=n_samples)
#
# # to image
# grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
# Image.fromarray(grid.astype(np.uint8))
