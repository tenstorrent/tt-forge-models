# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""VibeVoice-1.5B (microsoft/VibeVoice-1.5B) end-to-end text-to-speech on Tenstorrent.

Drives the full ``generate()`` path — text + voice prompt in, waveform out — with
a selectable set of components resident on TT and the rest on CPU. The default
set is the target shape, not a workaround: the diffusion head and both speech
connectors run on device, the acoustic/semantic tokenizers stay on CPU because
they are a small share of the compute, and the LM backbone stays on CPU because
its ``DynamicCache`` grows a token per step, so every decode step is a new shape
(see ``LM_RESIDENCY_NOTE`` below).

Residency is per component; the pipeline's tensors are not. Every CPU/TT seam
therefore needs a rule in **both** directions:

* what a *resident* module hands out — :class:`Residency`'s ``out_device``, which
  is a property of the module's **consumer**, not of the module;
* what a *CPU-resident* module is handed — :func:`cpu_pin`, because upstream
  moves inputs explicitly for the two tokenizers but not for the connectors.

Each resident module can be PCC-gated per forward against a CPU twin of itself
(a deep copy taken *before* the move, so it is the same weights at the same dtype
rather than a second load). The pipeline always consumes the TT output, so
gating reports without changing what the run produces.

**Waveform PCC is not a usable acceptance criterion for this model.** A run whose
per-forward PCC is 0.99997 lands at waveform PCC 0.734 against a CPU golden while
being the same length, stopping at the same step and transcribing identically:
there are ``ddpm_steps`` solver steps per frame and each frame's latent feeds back
through the connectors into the next LM step, so a 3e-05 per-forward difference
compounds across the autoregressive frames. Gate on the transcript instead.
"""

from __future__ import annotations

import copy
import time
from typing import List, Optional, Sequence

import torch

from .loader import ModelLoader

# Every component that can be moved onto TT, in the order they appear in the
# forward path. "connectors" moves both speech connectors together — they are
# 2.5 M parameters each and always share a consumer.
AVAILABLE_COMPONENTS = ("diffusion_head", "connectors", "lm")

# The target configuration. The LM is deliberately absent; see LM_RESIDENCY_NOTE.
DEFAULT_COMPONENTS = ("diffusion_head", "connectors")

LM_RESIDENCY_NOTE = """\
The LM backbone is not in DEFAULT_COMPONENTS, and that is a measurement, not an
oversight. Adding it needs two things this pipeline does not yet have:

1. A static, max-length-padded KV cache. Upstream's DynamicCache grows by one
   token per step, so every decode step presents a new shape and recompiles
   (~76-157 s/step), and torch._dynamo's recompile_limit trips at step 4 —
   which silently drops the LM out of the compiled path.
2. DRAM headroom. In float32 the run dies at step 5 allocating 933,494,784 B,
   which is exactly 151936 x 1536 x 4: the tied embedding / lm_head table.

Both are properties of the LM decode loop, not of this pipeline's seams, and
neither blocks the end-to-end result: with the LM on CPU the full text + voice
prompt -> waveform path completes and transcribes correctly.
"""

OUTPUT_SAMPLE_RATE = 24000

# Model-card-shaped example script. The "Speaker N:" prefix is required — the
# processor splits the script on it and assigns each speaker a voice sample.
DEFAULT_TEXT = ModelLoader.DEFAULT_TEXT

# What DEFAULT_TEXT should transcribe back to, for the transcript gate. Without
# the speaker prefix, which is markup rather than something the model speaks.
DEFAULT_REFERENCE_TRANSCRIPT = (
    "Hello from Tenstorrent. This is a VibeVoice end to end test."
)

# Denoise steps per acoustic frame in the diffusion head. Upstream's demo
# default; more steps cost linear time in the inner loop.
DEFAULT_DDPM_STEPS = 10

# Classifier-free guidance scale, upstream's demo default.
DEFAULT_CFG_SCALE = 1.3

# Per-forward correlation floor against the CPU twin.
DEFAULT_PCC_THRESHOLD = 0.99

# Stages that run outside the per-frame loop, reported as scalars. Everything
# else the timer sees — the diffusion head, the connectors, the LM decode — is
# per-frame work and is accumulated into the open step instead. The two sets are
# disjoint by construction: the benchmark harness sums ``components`` and
# ``steps`` against one wall-clock read, so anything in both would read as
# overlapping timers.
ONESHOT_STAGES = ("tokenizer_encode", "audio_decode")


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation, accumulated in float64.

    In float32 this reads 1.0018 for a 19.4 M-element tensor — accumulation
    error, not correlation — so any threshold read off an fp32 number at these
    sizes is fiction.
    """
    a = a.detach().to("cpu").double().flatten()
    b = b.detach().to("cpu").double().flatten()
    if a.numel() != b.numel():
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return float("nan") if denom == 0 else (a @ b / denom).item()


def _to(obj, device):
    """Recursively move tensors in a container, preserving the container class."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to(o, device) for o in obj)
    if isinstance(obj, dict):
        # Rebuild in the original class, not a plain dict: the LM returns a
        # transformers ``ModelOutput`` (an OrderedDict subclass) and the caller
        # reads ``.last_hidden_state`` off it. Falls back to a plain dict for
        # anything whose constructor will not take the mapping — a Cache, say.
        moved = {k: _to(v, device) for k, v in obj.items()}
        try:
            return type(obj)(**moved)
        except Exception:
            return moved
    return obj


def _sync_device() -> None:
    """Launch any pending XLA graph and wait for it.

    torch-xla is lazy, so a timer closed when a forward returns measures tracing
    rather than device work. ``wait_device_ops()`` alone is a no-op when nothing
    has been launched, so both calls are needed.
    """
    import torch_xla
    import torch_xla.core.xla_model as xm

    torch_xla.sync()
    xm.wait_device_ops()


def _compile_count() -> int:
    """Cumulative number of graph compilations observed so far."""
    import torch_xla.debug.metrics as met

    data = met.metric_data("CompileTime")
    return data[0] if data else 0


class Residency:
    """Move one submodule onto TT, optionally PCC-gating each forward.

    The twin is a deep copy taken *before* the module moves, so it is the same
    weights at the same dtype rather than a second load. It is kept on CPU and
    run in lockstep; the pipeline always consumes the TT output, so a PCC report
    never changes what the run produces.

    ``out_device`` is which device the result comes back on, and it is a property
    of the *consumer*, not of this module. The diffusion head's output is consumed
    by the DPM-solver loop, which is device-resident once its ``speech`` tensor is
    (``sample_speech_tokens`` builds it on the head's device), so that one stays on
    TT. The connectors' output is written into ``next_inputs_embeds``, built from
    the LM's own embedding table, so those follow the *LM's* device — which is not
    a constant. Getting this wrong surfaces as "Input tensor is not an XLA tensor"
    at the first mixed-device op.

    ``timer`` is an optional callable invoked as ``timer(name, seconds)`` after
    every forward, used by the benchmark to attribute device time per stage.
    """

    def __init__(
        self,
        name,
        module,
        device,
        threshold=DEFAULT_PCC_THRESHOLD,
        out_device=None,
        gate=True,
        timer=None,
    ):
        self.name = name
        self.threshold = threshold
        self.module = module
        self.pccs: List[float] = []
        self.gate = gate
        self.out_device = device if out_device is None else out_device

        self.twin = copy.deepcopy(module).to("cpu").eval() if gate else None
        module.to(device)
        # Compile the *unbound* original forward: torch.compile(module) would
        # call module.forward, which we are about to replace, and recurse.
        inner = module.forward
        compiled = torch.compile(inner, backend="tt")

        def forward(*args, **kwargs):
            dev_args = _to(args, device)
            dev_kwargs = _to(kwargs, device)
            started = time.perf_counter()
            out = compiled(*dev_args, **dev_kwargs)
            if timer is not None:
                _sync_device()
                timer(self.name, time.perf_counter() - started)
            if self.gate:
                golden = self.twin(*_to(args, "cpu"), **_to(kwargs, "cpu"))
                got = out[0] if isinstance(out, (tuple, list)) else out
                exp = golden[0] if isinstance(golden, (tuple, list)) else golden
                score = pcc(got, exp)
                self.pccs.append(score)
                if score < self.threshold:
                    raise AssertionError(
                        f"{self.name} forward {len(self.pccs)}: PCC {score:.6f} "
                        f"below {self.threshold}"
                    )
            # Short-circuit the common case. Matters for the LM: its output
            # carries the KV cache, which must stay on device and must not be
            # rebuilt by _to() — generate() mutates it in place across the CFG
            # branches and relies on holding the same objects.
            if self.out_device == device:
                return out
            return _to(out, self.out_device)

        module.forward = forward

    @property
    def forwards(self) -> int:
        return len(self.pccs)

    def summary(self) -> str:
        if not self.pccs:
            return f"  {self.name:16s} not gated (no CPU twin)"
        return (
            f"  {self.name:16s} {len(self.pccs):4d} forwards  "
            f"min={min(self.pccs):.6f}  mean={sum(self.pccs) / len(self.pccs):.6f}"
        )


def time_method(obj, name, stage, timer):
    """Attribute a CPU-resident entry point's wall time to a named stage.

    The benchmark harness requires the stage timers to account for a real share
    of the generation, and in the default residency the largest single consumer
    of wall time is the *CPU* LM decode loop. Timing only the device components
    would under-report so badly that the harness reads it as a missing
    ``sync_device()``. No device sync here: CPU work is synchronous already.
    """
    inner = getattr(obj, name)

    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        out = inner(*args, **kwargs)
        timer(stage, time.perf_counter() - started)
        return out

    setattr(obj, name, wrapper)


def cpu_pin(module, out_device="cpu"):
    """Run a CPU-resident module on CPU whatever device its inputs arrive on.

    Needed because residency is per component but the pipeline's tensors are not:
    with only ``diffusion_head`` on TT, ``sample_speech_tokens`` returns
    ``speech_latent`` on TT (it builds ``speech`` on the head's device) and hands
    it straight to a CPU ``acoustic_connector``. Upstream moves inputs explicitly
    for the two tokenizers (``.to(self.model.X.device)``) but not for the
    connectors, so those need pinning whenever they are not resident.

    ``out_device`` is the consumer's device, exactly as in :class:`Residency`: a
    CPU connector still has to hand its result back to ``next_inputs_embeds``,
    which is on TT whenever the LM's embedding table is.
    """
    cpu_pin_method(module, "forward", out_device)


def cpu_pin_method(obj, name, out_device="cpu"):
    """:func:`cpu_pin` for an entry point that is not ``forward``.

    The tokenizers are always CPU-resident here but are called through ``.encode``
    / ``.decode``, so patching ``forward`` would not intercept them. The generation
    loop moves their inputs explicitly, but the voice-prompt conditioning path in
    ``_process_speech_inputs`` does not — so with the LM on TT the prompt arrives
    on device and hits a CPU conv.
    """
    inner = getattr(obj, name)

    def wrapper(*args, **kwargs):
        out = inner(*_to(args, "cpu"), **_to(kwargs, "cpu"))
        return _to(out, out_device) if out_device != "cpu" else out

    setattr(obj, name, wrapper)


def retie_lm_head(model, device):
    """Restore the ``lm_head`` <-> embedding weight tie, which the move breaks.

    ``lm_head`` is a child of the outer inference class, not of ``language_model``,
    but its weight *is* the embedding weight — VibeVoice-1.5B ships no
    ``lm_head.weight`` because it is tied. Moving the LM breaks that:
    ``nn.Module._apply`` only mutates a Parameter in place when the old and new
    tensors are shallow-copy compatible, and a CPU tensor and an XLA tensor are
    not, so it constructs a *new* Parameter and rebinds it on the LM. The embedding
    ends up on device, ``lm_head.weight`` keeps pointing at the orphaned CPU
    Parameter, and the tie is gone — measured, not assumed.

    The tempting fix, ``lm_head.to(device)`` on its own, reintroduces the defect
    quietly: it puts a *copy* of the embedding table on device, correct by value
    today, untied, and 0.93 GB of duplicated weights. So rebind, then assert
    pointer identity. Any tied-embedding model moved onto a device
    submodule-by-submodule has this exposure.
    """
    model.lm_head.to(device)
    model.lm_head.weight = model.get_input_embeddings().weight
    tied = (
        model.lm_head.weight.data_ptr()
        == model.get_input_embeddings().weight.data_ptr()
    )
    if not tied or model.lm_head.weight.device.type != device.type:
        raise AssertionError(
            f"lm_head re-tie failed: tied={tied}, device={model.lm_head.weight.device}"
        )


class VibeVoiceConfig:
    """Inputs and residency choices for one :class:`VibeVoicePipeline` run."""

    def __init__(
        self,
        text: Optional[str] = None,
        voice_samples: Optional[Sequence[str]] = None,
        components: Sequence[str] = DEFAULT_COMPONENTS,
        dtype: torch.dtype = torch.float32,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        ddpm_steps: int = DEFAULT_DDPM_STEPS,
        seed: int = 0,
        max_new_tokens: Optional[int] = None,
        gate: bool = True,
        pcc_threshold: float = DEFAULT_PCC_THRESHOLD,
        collect_perf: bool = False,
    ):
        self.text = DEFAULT_TEXT if text is None else text
        self.voice_samples = voice_samples
        self.components = tuple(components)
        # float32 by default: the CPU twin is the PCC golden for the TT run, so
        # it should not itself be quantised.
        self.dtype = dtype
        self.cfg_scale = cfg_scale
        self.ddpm_steps = ddpm_steps
        # Re-applied before every generate(), so repeat runs are bit-identical.
        # The diffusion loop draws noise per frame and that noise feeds back into
        # the LM, so without this the generated length itself drifts run to run.
        self.seed = seed
        # None lets the model choose its own cap from the script length.
        self.max_new_tokens = max_new_tokens
        self.gate = gate
        self.pcc_threshold = pcc_threshold
        # Stage/step timing costs a device sync per forward, so it is opt-in.
        self.collect_perf = collect_perf

        unknown = set(self.components) - set(AVAILABLE_COMPONENTS)
        if unknown:
            raise ValueError(
                f"unknown components {sorted(unknown)}; "
                f"available: {list(AVAILABLE_COMPONENTS)}"
            )


class VibeVoicePipeline:
    """The full VibeVoice TTS path with a selectable set of components on TT."""

    def __init__(self, config: Optional[VibeVoiceConfig] = None):
        self.config = config or VibeVoiceConfig()
        self.model = None
        self.processor = None
        self.inputs = None
        self.residencies: List[Residency] = []
        self.device = None
        self.saw_eos = False
        self.steps = 0
        self._perf = {}
        self.stage_totals = {}
        self._stage_s = {}
        self._step_open_s = 0.0
        self._head_forwards = 0

    # -- setup ------------------------------------------------------------

    def setup(self):
        """Load the model on CPU, then move the requested components to TT."""
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr

        xr.set_device_type("TT")
        self.device = xm.xla_device()

        loader = ModelLoader()
        self.model = loader.load_tts_model(
            dtype_override=self.config.dtype, ddpm_steps=self.config.ddpm_steps
        )
        self.processor = loader.load_processor()
        self.inputs = loader.load_tts_inputs(
            text=self.config.text,
            voice_samples=(
                list(self.config.voice_samples) if self.config.voice_samples else None
            ),
            processor=self.processor,
        )
        self._build_residencies()
        return self

    def _timer(self, name, seconds):
        """Attribute one forward's time to either a one-shot stage or the open step.

        The split has to be exact: the harness sums ``components`` and ``steps``
        and compares that against a single wall-clock read, so a stage counted in
        both would read as overlapping timers.
        """
        self._stage_s[name] = self._stage_s.get(name, 0.0) + seconds
        if name in ONESHOT_STAGES:
            return
        self._step_open_s += seconds
        if name == "diffusion_head":
            self._head_forwards += 1
            # A frame is ddpm_steps head forwards. Close the step on the last of
            # them rather than on a connector call: the connectors fire an
            # unequal number of times (the acoustic one also runs during
            # voice-prompt conditioning), so their ordering is not a reliable
            # frame boundary.
            if self._head_forwards % self.config.ddpm_steps == 0:
                self._perf["steps"].append(self._step_open_s)
                self._perf["compile_curve"].append(_compile_count())
                self._step_open_s = 0.0

    def _build_residencies(self):
        """Attach a :class:`Residency` per requested component.

        ``out_device`` per component follows the consumer, not the module. The
        head's output feeds the DPM-solver loop and stays on TT. The connectors'
        outputs are written into ``next_inputs_embeds``, which is built by the
        LM's own embedding table — so they follow *its* device: CPU when the LM
        is not resident, TT when it is.
        """
        cfg = self.config
        model, device = self.model, self.device
        wanted = cfg.components
        timer = self._timer if cfg.collect_perf else None

        if timer is not None and "diffusion_head" not in wanted:
            raise ValueError(
                "collect_perf needs diffusion_head resident: the per-frame step "
                "boundary is counted off its forwards"
            )

        embeds_device = device if "lm" in wanted else "cpu"

        # The tokenizers stay on CPU in every configuration — that is the target
        # shape, not a workaround — so they always get pinned.
        cpu_pin_method(model.model.acoustic_tokenizer, "encode")
        cpu_pin_method(model.model.acoustic_tokenizer, "decode")
        cpu_pin_method(model.model.semantic_tokenizer, "encode")

        if timer is not None:
            # Time the CPU side too. In the default residency the LM decode loop
            # is the largest single consumer of wall time, and the harness reads
            # an under-reporting stage total as a missing device sync rather than
            # as work that legitimately ran on the host.
            time_method(
                model.model.acoustic_tokenizer, "encode", "tokenizer_encode", timer
            )
            time_method(
                model.model.semantic_tokenizer, "encode", "tokenizer_encode", timer
            )
            time_method(model.model.acoustic_tokenizer, "decode", "audio_decode", timer)
            if "lm" not in wanted:
                time_method(model.model.language_model, "forward", "lm_cpu", timer)

        out: List[Residency] = []
        if "diffusion_head" in wanted:
            out.append(
                Residency(
                    "diffusion_head",
                    model.model.prediction_head,
                    device,
                    cfg.pcc_threshold,
                    out_device=device,
                    gate=cfg.gate,
                    timer=timer,
                )
            )
        if "connectors" in wanted:
            for name, mod in (
                ("acoustic_conn", model.model.acoustic_connector),
                ("semantic_conn", model.model.semantic_connector),
            ):
                out.append(
                    Residency(
                        name,
                        mod,
                        device,
                        cfg.pcc_threshold,
                        out_device=embeds_device,
                        gate=cfg.gate,
                        timer=timer,
                    )
                )
        else:
            cpu_pin(model.model.acoustic_connector, out_device=embeds_device)
            cpu_pin(model.model.semantic_connector, out_device=embeds_device)

        if "lm" in wanted:
            # Never twin-gated. The twin would need its own CPU KV cache running
            # in lockstep: generate() hands the LM a DynamicCache it mutates in
            # place across the CFG branches, and that object is neither a tensor
            # nor a mapping, so _to() cannot clone it onto CPU. Feeding the CPU
            # twin the device cache would compare against a cache it never wrote.
            # The LM is covered by the LM_PREFILL component test plus the
            # end-to-end transcript gate instead.
            out.append(
                Residency(
                    "lm",
                    model.model.language_model,
                    device,
                    cfg.pcc_threshold,
                    out_device=device,
                    gate=False,
                    timer=timer,
                )
            )
            retie_lm_head(model, device)

        self.residencies = out

    # -- run --------------------------------------------------------------

    def run(self) -> torch.Tensor:
        """Synthesize one waveform. Returns it and populates ``_perf``."""
        cfg = self.config
        self._stage_s = {}
        self._step_open_s = 0.0
        self._head_forwards = 0
        self._perf = {"steps": [], "compile_curve": []}

        initial_length = self.inputs["input_ids"].shape[1]
        generate_kwargs = dict(
            cfg_scale=cfg.cfg_scale,
            tokenizer=self.processor.tokenizer,
            # Note the mapping form: upstream reads do_sample off a
            # generation_config, not off a top-level do_sample= kwarg.
            generation_config={"do_sample": False},
            verbose=False,
        )
        generate_kwargs["max_new_tokens"] = cfg.max_new_tokens

        torch.manual_seed(cfg.seed)
        started = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**self.inputs, **generate_kwargs)
        total = time.perf_counter() - started

        wav = out.speech_outputs[0]
        if wav is None:
            raise RuntimeError("generate() produced no audio")

        self.steps = out.sequences.shape[1] - initial_length
        eos_id = self.processor.tokenizer.eos_token_id
        self.saw_eos = bool((out.sequences[0, initial_length:] == eos_id).any().item())

        self._perf.update(
            {
                # Only the one-shot stages; per-frame work is in "steps".
                "components": {
                    k: self._stage_s[k] for k in ONESHOT_STAGES if k in self._stage_s
                },
                "step_metric_name": "acoustic_frame",
                "total": total,
                "audio_samples": int(wav.shape[-1]),
                "text_tokens": int(initial_length),
            }
        )
        # Every stage's total, including the per-frame ones that "components"
        # deliberately leaves out. Reporting only; never summed into the
        # harness's stage accounting.
        self.stage_totals = dict(self._stage_s)
        return wav

    # -- reporting --------------------------------------------------------

    def summary(self) -> str:
        lines = [f"components on TT: {[r.name for r in self.residencies] or 'none'}"]
        for r in self.residencies:
            lines.append(r.summary())
        return "\n".join(lines)

    def worst_pcc(self) -> Optional[float]:
        """Lowest PCC across every gated forward, or ``None`` if nothing gated."""
        scores = [s for r in self.residencies for s in r.pccs]
        return min(scores) if scores else None


def save_wav(wav: torch.Tensor, filepath: str = "vibevoice_output.wav", processor=None):
    """Write a generated waveform to disk at 24 kHz."""
    if processor is None:
        processor = ModelLoader().load_processor()
    processor.save_audio(wav, output_path=filepath)
    return filepath


def run_vibevoice_pipeline(
    output_path: Optional[str] = "vibevoice_output.wav",
    config: Optional[VibeVoiceConfig] = None,
    **config_kwargs,
):
    """Build, run and (optionally) save one synthesis.

    Returns ``(waveform, pipeline)`` rather than just the waveform: the pipeline
    carries the per-forward PCC each resident component recorded against its CPU
    twin, which is what a caller needs to tell a working run from a plausible one.
    """
    if config is None:
        config = VibeVoiceConfig(**config_kwargs)
    elif config_kwargs:
        raise TypeError("pass either config= or keyword arguments, not both")
    pipeline = VibeVoicePipeline(config).setup()
    wav = pipeline.run()
    if output_path:
        save_wav(wav, output_path, processor=pipeline.processor)
    return wav, pipeline
