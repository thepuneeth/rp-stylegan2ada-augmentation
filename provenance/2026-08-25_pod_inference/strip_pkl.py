"""Strip a StyleGAN2-ADA snapshot to G_ema only, for inference/metrics upload.

legacy.load_network_pkl asserts G, D and G_ema are all nn.Module, so they cannot
be dropped. Pointing all three at the same G_ema object makes pickle store the
weights once (~99 MB instead of ~296 MB). Only G_ema is ever read by
generate.py and calc_metrics.py. Originals are never touched.
"""
import sys, os, pickle, copy
sys.path.insert(0, "/Users/puneeth/Documents/Research/GAN/pigmentGAN/stylegan2-ada-pytorch-main")
import dnnlib, legacy, torch

src, dst = sys.argv[1], sys.argv[2]
with dnnlib.util.open_url(src) as f:
    d = legacy.load_network_pkl(f)
g = d["G_ema"]
out = dict(G=g, D=g, G_ema=g,
           training_set_kwargs=d.get("training_set_kwargs"),
           augment_pipe=None)
with open(dst, "wb") as f:
    pickle.dump(out, f)
print(f"{os.path.basename(src)} {os.path.getsize(src)/1e6:.0f}MB -> "
      f"{os.path.basename(dst)} {os.path.getsize(dst)/1e6:.0f}MB")

# self-check: the stripped file must reload and produce identical output
with dnnlib.util.open_url(dst) as f:
    g2 = legacy.load_network_pkl(f)["G_ema"]
z = torch.randn([2, g.z_dim], generator=torch.Generator().manual_seed(0))
c = torch.zeros([2, g.c_dim])
with torch.no_grad():
    a = g(z, c, noise_mode="const")
    b = g2(z, c, noise_mode="const")
assert torch.equal(a, b), "stripped G_ema does not reproduce original output"
print("  self-check OK: identical output")
