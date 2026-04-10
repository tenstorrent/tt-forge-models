# Quietbox Forge Repro Candidates

This branch is a focused local repro branch for quietbox failures that are worth investigating in `tt-forge-models`.

Current candidate set:
- `dandelin/vilt-b32-mlm`
  - test: `test_all_models_torch[vilt/masked_lm/pytorch-Mlm-single_device-inference]`
  - loader: `vilt/masked_lm/pytorch/loader.py`
  - wave: `hf100k-quietbox-direct-wave36`

Observed failure signals from quietbox artifacts:
- `Failed to legalize operation 'ttir.gather'`
- `module 'spacy' has no attribute 'Language'`

Primary local evidence:
- `vulcan_repro/quietbox_forge_repro_candidates.json`

This branch is intentionally branch-local and not intended for upstream merge as-is.
