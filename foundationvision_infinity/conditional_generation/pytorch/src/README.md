# Vendored third-party code

`infinity/` is vendored (unmodified) from the upstream Infinity repository:

- Source: https://github.com/FoundationVision/Infinity
- Paper: Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution
  Image Synthesis (arXiv:2412.04431)
- License: MIT (see upstream `LICENSE`)

Only the modules needed to construct and run the Infinity transformer are
vendored: `infinity/models/{infinity,basic,flex_attn,fused_op,init_param}.py`,
`infinity/models/bsq_vae/*` and `infinity/utils/{dist,misc,dynamic_resolution}.py`.

`flash_attn/` is **not** upstream code. It is a small pure-PyTorch shim
(Apache-2.0, Tenstorrent) that implements the two `flash_attn` entry points the
model imports, using `scaled_dot_product_attention`, so the model runs without
the CUDA-only `flash_attn` package. See `flash_attn/__init__.py`.
