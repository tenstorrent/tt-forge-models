#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Boltz model loader implementation (structure prediction).

This provides a ForgeModel-compatible loader with:
  - load_model: returns a Boltz2 LightningModule loaded from checkpoint
  - load_inputs: returns a single preprocessed batch (feats dict) suitable for forward(...)
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional, Any, Literal

import torch
from types import MethodType
from rdkit import Chem
import os
import click
from pytorch_lightning import seed_everything

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

# Import Boltz package (installed in venv)
from boltz.model.models.boltz2 import Boltz2
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import Manifest
from boltz.main import (
    download_boltz2,
    get_cache_path,
    PairformerArgsV2,
    MSAModuleArgs,
    Boltz2DiffusionParams,
    BoltzSteeringParams,
)
from third_party.tt_forge_models.boltz.pytorch.src.model_utils import (
    process_inputs,
    filter_inputs_structure,
    BoltzProcessedInput,
    check_inputs,
)


class ModelVariant(StrEnum):
    """Available Boltz model variants."""

    BOLTZ2 = "boltz2"


class ModelLoader(ForgeModel):
    """Boltz model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BOLTZ2: ModelConfig(
            pretrained_model_name="boltz2",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BOLTZ2

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="boltz",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.ATOMIC_ML,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.model: Optional[torch.nn.Module] = None

    def load_model(
        self,
        *,
        checkpoint: Optional[str | Path] = None,
        cache_dir: Optional[
            str | Path
        ] = "/proj_sw/user_dev/mramanathan/bgdlab19_nov13_xla/tt-xla/third_party/tt_forge_models/boltz/boltz_results_fast_protein",
        dtype_override: Optional[torch.dtype] = torch.float32,
        use_kernels: bool = False,
        recycling_steps: int = 3,
        sampling_steps: int = 200,
        diffusion_samples: int = 1,
        max_parallel_samples: Optional[int] = None,
        write_confidence_summary: bool = False,
        write_full_pae: bool = False,
        write_full_pde: bool = False,
        run_confidence_sequentially: bool = True,
        run_trunk_and_structure: bool = True,
        skip_run_structure: bool = True,
        confidence_prediction: bool = False,
        step_scale: Optional[float] = None,
        default_feats: Optional[dict[str, Any]] = None,
    ) -> torch.nn.Module:
        # Resolve cache and ensure required assets are present
        cache = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir is not None
            else Path(get_cache_path()).expanduser().resolve()
        )
        cache.mkdir(parents=True, exist_ok=True)
        download_boltz2(cache)

        # Resolve checkpoint path (default to standard Boltz-2 checkpoint in cache)
        checkpoint_path = (
            Path(checkpoint).expanduser().resolve()
            if checkpoint is not None
            else cache / "boltz2_conf.ckpt"
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}. "
                f"Expected the default Boltz-2 checkpoint at {cache / 'boltz2_conf.ckpt'} "
                f"or pass a valid path via `checkpoint`."
            )
        model_cls = Boltz2
        diffusion_params = Boltz2DiffusionParams()
        step_scale = 1.5 if step_scale is None else step_scale
        diffusion_params.step_scale = step_scale
        pairformer_args = PairformerArgsV2()
        predict_args = {
            "recycling_steps": recycling_steps,
            "sampling_steps": sampling_steps,
            "diffusion_samples": diffusion_samples,
            "max_parallel_samples": max_parallel_samples,
            "write_confidence_summary": True,
            "write_full_pae": write_full_pae,
            "write_full_pde": write_full_pde,
        }
        no_kernels = False
        msa_args = MSAModuleArgs(
            subsample_msa=True,
            num_subsampled_msa=1024,
            use_paired_feature=True,
        )
        steering_args = BoltzSteeringParams()
        steering_args.fk_steering = False
        steering_args.physical_guidance_update = False

        model_module = model_cls.load_from_checkpoint(
            str(checkpoint_path),
            strict=True,
            predict_args=predict_args,
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            use_kernels=not no_kernels,
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
            steering_args=asdict(steering_args),
        )
        model_module.eval()
        self.model = model_module
        self.model.use_kernels = False

        # Inject default forward kwargs so callers need not pass them in tests
        # Map loader parameters to model.forward signature keys
        _default_forward_kwargs = {
            "recycling_steps": recycling_steps,
            "num_sampling_steps": sampling_steps,
            "diffusion_samples": diffusion_samples,
            "max_parallel_samples": max_parallel_samples,
            "run_confidence_sequentially": run_confidence_sequentially,
        }

        # Preserve original forward (bound) to delegate to and optionally store default feats
        _orig_forward_bound = self.model.forward
        setattr(self.model, "_default_feats", default_feats)

        def _forward_with_defaults(module_self, *args, **kwargs):
            # Determine feats from positional arg, kwarg, or model's default
            if args:
                feats = args[0]
            else:
                feats = kwargs.pop("feats", None)
                if feats is None:
                    feats = getattr(module_self, "_default_feats", None)
                    if feats is None:
                        raise TypeError(
                            "feats not provided and no default feats set on model. "
                            "Pass feats=... or provide default_feats to load_model()."
                        )
            merged_kwargs = {**_default_forward_kwargs, **kwargs}
            return _orig_forward_bound(feats, **merged_kwargs)

        # Bind the wrapper as the instance method
        self.model.forward = MethodType(_forward_with_defaults, self.model)

        return self.model

    def load_inputs(
        self,
        *,
        out_dir: str
        | Path = "/proj_sw/user_dev/mramanathan/bgdlab19_nov13_xla/tt-xla/third_party/tt_forge_models/boltz/boltz_results_fast_protein",
        cache_dir: Optional[
            str | Path
        ] = "/proj_sw/user_dev/mramanathan/bgdlab19_nov13_xla/tt-xla/third_party/tt_forge_models/boltz_git/new",
        # optional: raw input path to auto-preprocess if manifest is missing
        data: Optional[
            str | Path
        ] = "/proj_sw/user_dev/mramanathan/bgdlab19_nov13_xla/tt-xla/tests/torch/single_chip/models/boltz/fast_protein.yaml",
        num_workers: int = 0,
        affinity: bool = False,
        override_method: Optional[str] = None,
        dtype_override: Optional[torch.dtype] = None,
        method: Optional[str] = None,
        model: Literal["boltz1", "boltz2"] = "boltz2",
        # preprocessing knobs (mirrors standalone defaults)
        use_msa_server: bool = True,
        msa_server_url: str = "https://api.colabfold.com",
        msa_pairing_strategy: str = "greedy",
        max_msa_seqs: int = 8192,
        preprocessing_threads: int = 1,
        msa_server_username: Optional[str] = None,
        msa_server_password: Optional[str] = None,
        api_key_header: Optional[str] = None,
        api_key_value: Optional[str] = None,
    ) -> list[Any]:

        # Set no grad
        torch.set_grad_enabled(False)

        # Ignore matmul precision warning
        torch.set_float32_matmul_precision("highest")

        # seed = 42
        # seed_everything(seed)
        # torch.manual_seed(seed)

        #  # Set seed if desired
        # if seed is not None:
        #     seed_everything(seed)

        # Set rdkit pickle logic
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

        # Set cache path
        cache = Path(cache_dir).expanduser()
        cache.mkdir(parents=True, exist_ok=True)

        # Get MSA server credentials from environment variables if not provided
        if use_msa_server:
            if msa_server_username is None:
                msa_server_username = os.environ.get("BOLTZ_MSA_USERNAME")
            if msa_server_password is None:
                msa_server_password = os.environ.get("BOLTZ_MSA_PASSWORD")
            if api_key_value is None:
                api_key_value = os.environ.get("MSA_API_KEY_VALUE")

            click.echo(f"MSA server enabled: {msa_server_url}")
            if api_key_value:
                click.echo("MSA server authentication: using API key header")
            elif msa_server_username and msa_server_password:
                click.echo("MSA server authentication: using basic auth")
            else:
                click.echo("MSA server authentication: no credentials provided")
            # Resolve cache and ensure required assets are present for feature loading
            cache = (
                Path(cache_dir).expanduser().resolve()
                if cache_dir is not None
                else Path(get_cache_path()).expanduser().resolve()
            )

        # Create output directories
        data = Path(data).expanduser()
        out_dir = Path(out_dir).expanduser()
        out_dir = out_dir / f"boltz_results_{data.stem}"
        out_dir.mkdir(parents=True, exist_ok=True)
        download_boltz2(cache)
        # Process inputs
        ccd_path = cache / "ccd.pkl"
        mol_dir = cache / "mols"

        # Validate inputs
        data = check_inputs(data)

        # Run preprocessing with the same defaults as standalone Boltz-2
        process_inputs(
            data=data,
            out_dir=out_dir,
            ccd_path=ccd_path,
            mol_dir=mol_dir,
            use_msa_server=use_msa_server,
            msa_server_url=msa_server_url,
            msa_pairing_strategy=msa_pairing_strategy,
            msa_server_username=msa_server_username,
            msa_server_password=msa_server_password,
            api_key_header=api_key_header,
            api_key_value=api_key_value,
            boltz2=(model == "boltz2"),
            preprocessing_threads=preprocessing_threads,
            max_msa_seqs=max_msa_seqs,
        )

        # Load manifest
        manifest = Manifest.load(out_dir / "processed" / "manifest.json")

        # Optionally filter out existing predictions (keep all for input loading)
        filtered_manifest = filter_inputs_structure(
            manifest=manifest,
            outdir=out_dir,
            override=True,
        )

        # Load processed data
        processed_dir = out_dir / "processed"
        processed = BoltzProcessedInput(
            manifest=filtered_manifest,
            targets_dir=processed_dir / "structures",
            msa_dir=processed_dir / "msa",
            constraints_dir=(
                (processed_dir / "constraints")
                if (processed_dir / "constraints").exists()
                else None
            ),
            template_dir=(
                (processed_dir / "templates")
                if (processed_dir / "templates").exists()
                else None
            ),
            extra_mols_dir=(
                (processed_dir / "mols") if (processed_dir / "mols").exists() else None
            ),
        )

        # Create data module (Boltz-2)
        data_module = Boltz2InferenceDataModule(
            manifest=processed.manifest,
            target_dir=processed.targets_dir,
            msa_dir=processed.msa_dir,
            mol_dir=mol_dir,
            num_workers=num_workers,
            constraints_dir=processed.constraints_dir,
            template_dir=processed.template_dir,
            extra_mols_dir=processed.extra_mols_dir,
            override_method=method,
        )

        # Prepare predict dataloader and return a single batch of features
        predict_loader = data_module.predict_dataloader()
        try:
            first_batch = next(iter(predict_loader))
        except StopIteration:
            raise RuntimeError("No inputs available in the processed dataset to load.")

        # Optionally cast tensors to requested dtype
        if dtype_override is not None:

            def _cast(x):
                return (
                    x.to(dtype_override)
                    if isinstance(x, torch.Tensor) and x.is_floating_point()
                    else x
                )

            if isinstance(first_batch, dict):
                first_batch = {k: _cast(v) for k, v in first_batch.items()}
            elif isinstance(first_batch, (list, tuple)):
                first_batch = type(first_batch)(_cast(x) for x in first_batch)

        # Return as single-item list to align with consumer expectations
        return [first_batch]
