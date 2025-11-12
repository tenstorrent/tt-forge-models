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
from typing import Optional, Any

import torch

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

    def _ensure_checkpoint(self, cache_dir: Optional[Path]) -> Path:
        """Ensure default Boltz2 checkpoint exists in cache and return its path."""
        cache_root = (
            Path(cache_dir) if cache_dir is not None else Path(get_cache_path())
        )
        cache_root = cache_root.expanduser().resolve()
        cache_root.mkdir(parents=True, exist_ok=True)

        # Download weights and required molecule dictionary if missing
        download_boltz2(cache_root)

        ckpt = cache_root / "boltz2_conf.ckpt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Boltz2 checkpoint not found at {ckpt}. Download should have created it."
            )
        return ckpt

    def load_model(
        self,
        *,
        checkpoint: Optional[str | Path] = None,
        cache_dir: Optional[str | Path] = None,
        dtype_override: Optional[torch.dtype] = None,
        use_kernels: bool = True,
        recycling_steps: int = 3,
        sampling_steps: int = 200,
        diffusion_samples: int = 1,
        max_parallel_samples: int = 1,
        write_confidence_summary: bool = True,
        write_full_pae: bool = False,
        write_full_pde: bool = False,
        # runtime behavior flags:
        run_trunk_and_structure: bool = True,
        skip_run_structure: bool = False,
        confidence_prediction: bool = True,
    ) -> torch.nn.Module:
        """Load and return the Boltz2 model instance.

        Args:
            checkpoint: Optional explicit path to .ckpt. If None, will use cached default.
            cache_dir: Optional cache directory for default assets (weights, mols).
            dtype_override: Optional torch.dtype to convert model to.
            use_kernels: Whether to enable custom CUDA kernels (if available).
            recycling_steps: Number of MSA/trunk recycling iterations for prediction.
            sampling_steps: Number of diffusion sampling steps for structure.
            diffusion_samples: Number of diffusion samples.
            max_parallel_samples: Max parallel diffusion samples.
            write_confidence_summary: Whether to compute/write confidence summaries.
            write_full_pae: Whether to compute full PAE.
            write_full_pde: Whether to compute full PDE.
        """
        # Resolve checkpoint
        if checkpoint is None:
            ckpt_path = self._ensure_checkpoint(
                Path(cache_dir).expanduser().resolve() if cache_dir else None
            )
        else:
            ckpt_path = Path(checkpoint).expanduser().resolve()
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # Build args matching the LightningModule expected kwargs
        pairformer_args = PairformerArgsV2()
        msa_args = MSAModuleArgs()
        diffusion_params = Boltz2DiffusionParams()
        steering_args = BoltzSteeringParams()

        predict_args = {
            "recycling_steps": recycling_steps,
            "sampling_steps": sampling_steps,
            "diffusion_samples": diffusion_samples,
            "max_parallel_samples": max_parallel_samples,
            "write_confidence_summary": write_confidence_summary,
            "write_full_pae": write_full_pae,
            "write_full_pde": write_full_pde,
        }

        # Load LightningModule from checkpoint
        model_module = Boltz2.load_from_checkpoint(
            str(ckpt_path),
            strict=confidence_prediction,
            predict_args=predict_args,
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            use_kernels=use_kernels,
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
            steering_args=asdict(steering_args),
            # control runtime behavior
            run_trunk_and_structure=run_trunk_and_structure,
            skip_run_structure=skip_run_structure,
            confidence_prediction=confidence_prediction,
        )
        model_module.eval()

        if dtype_override is not None:
            # Only convert if numeric category matches (float -> float)
            for p in model_module.parameters():
                if hasattr(p, "dtype") and p.dtype.is_floating_point:
                    model_module.to(dtype_override)
                    break

        self.model = model_module
        return self.model

    def load_inputs(
        self,
        *,
        out_dir: str | Path,
        cache_dir: Optional[str | Path] = None,
        num_workers: int = 0,
        affinity: bool = False,
        override_method: Optional[str] = None,
        dtype_override: Optional[torch.dtype] = None,
    ) -> list[Any]:
        """Load and return one preprocessed batch suitable for Boltz2.forward(feats, ...).

        This expects you have already run preprocessing to populate:
          {out_dir}/processed/manifest.json
          {out_dir}/processed/structures/*.npz
          {out_dir}/processed/msa/*.npz
          optional: constraints/templates/mols under {out_dir}/processed/

        Args:
            out_dir: Root output directory containing the 'processed' folder (see above).
            cache_dir: Optional cache directory that contains 'mols' folder used by Boltz2.
            num_workers: DataLoader workers for reading a sample batch.
            affinity: If True, loads affinity features (token cropping etc.).
            override_method: Optional featurizer override method string.
            dtype_override: Optional dtype to cast float tensors to.

        Returns:
            list containing a single element: the feats dict batch for model.forward
        """
        out_dir = Path(out_dir).expanduser().resolve()
        processed = out_dir / "processed"
        manifest_path = processed / "manifest.json"
        targets_dir = processed / "structures"
        msa_dir = processed / "msa"
        constraints_dir = processed / "constraints"
        template_dir = processed / "templates"
        extra_mols_dir = processed / "mols"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")
        if not targets_dir.exists():
            raise FileNotFoundError(f"Processed structures not found at {targets_dir}")
        if not msa_dir.exists():
            raise FileNotFoundError(f"Processed MSA not found at {msa_dir}")

        # Ensure cache with molecules exists (used by dataset to load conformers)
        cache_root = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir
            else Path(get_cache_path()).expanduser().resolve()
        )
        cache_root.mkdir(parents=True, exist_ok=True)
        download_boltz2(cache_root)  # ensures mols/ is present
        mol_dir = cache_root / "mols"
        if not mol_dir.exists():
            raise FileNotFoundError(f"Molecule cache not found at {mol_dir}")

        # Load manifest
        manifest: Manifest = Manifest.load(manifest_path)

        # Construct datamodule and extract one batch
        datamodule = Boltz2InferenceDataModule(
            manifest=manifest,
            target_dir=targets_dir,
            msa_dir=msa_dir,
            mol_dir=mol_dir,
            num_workers=num_workers,
            constraints_dir=constraints_dir if constraints_dir.exists() else None,
            template_dir=template_dir if template_dir.exists() else None,
            extra_mols_dir=extra_mols_dir if extra_mols_dir.exists() else None,
            override_method=override_method,
            affinity=affinity,
        )
        dl = datamodule.predict_dataloader()
        batch = next(iter(dl))

        # Optionally cast float tensors
        if dtype_override is not None:
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                    batch[k] = v.to(dtype_override)

        # Return as positional args list matching forward signature (feats is positional)
        return [batch]
