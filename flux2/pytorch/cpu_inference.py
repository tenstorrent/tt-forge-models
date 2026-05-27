import torch
from diffusers import Flux2Pipeline

repo_id = "black-forest-labs/FLUX.2-dev"
device = "cpu"
torch_dtype = torch.float32

pipe = Flux2Pipeline.from_pretrained(
    repo_id, torch_dtype=torch_dtype
).to(device)

prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text `BFL Diffusers` on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."

# 128x128 for TT bringup; omit height/width to use pipeline default (1024x1024).
image = pipe(
    prompt=prompt,
    height=128,
    width=128,
    generator=torch.Generator(device=device).manual_seed(42),
    num_inference_steps=4,  # 28 steps can be a good trade-off
    guidance_scale=4,
).images[0]

image.save("flux2_output_128x128.png")
