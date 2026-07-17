# XTTS-v2 TP=2 Tensor-Parallel Sharding Spec (n300)

Sharding spec for running the XTTS-v2 e2e pipeline (`pipeline.py`) with 2-way
tensor parallelism on an **n300 (2 chips)**. Implemented in `sharding.py`, gated
behind `pipeline.py --tp 2`. `--tp 1` (default) is the unchanged single-chip path.

> **Not validated on hardware.** This was authored on a 1-chip machine; the
> emitted TTIR/TTNN and runtime behavior must be verified on an n300. The MLP
> sharding is straightforward; the fused-QKV attention sharding is the main
> item to validate (see Caveats).

## Run it

```bash
python -u -m third_party.tt_forge_models.xtts_v2.pytorch.pipeline \
  --tp 2 --output ~/xtts_tp2.wav --warmup 1 --repeat 3
```

`--tp 2` sets `CONVERT_SHLO_TO_SHARDY=1`, calls `xr.use_spmd()`, builds a
`(data=1, model=2)` mesh, and marks the GPT2 trunk weights with the spec below.

## Strategy

The autoregressive **GPT** (HF GPT2 trunk: 30 blocks, 16 heads, hidden 1024,
`n_inner` 4096, ~443M params) dominates runtime, so it gets Megatron-style TP.
The smaller components (speaker encoder, conditioning encoder, HiFi-GAN vocoder)
are left **replicated** across the mesh — they run redundantly on both chips
(wasted compute, but no CCL, and they are cheap relative to the GPT).

Megatron transformer TP (Shoeybi et al. 2019, §3): each transformer block is one
**column-parallel → row-parallel** pair, which costs exactly **one all-reduce**
(no gather of the sharded intermediate):

| Sub-layer | Weight (Conv1D `[in,out]`) | Parallelism | Partition spec |
|---|---|---|---|
| `attn.c_attn` | `[1024, 3072]` | column (shard out) | `(None, "model")` |
| `attn.c_attn.bias` | `[3072]` | column | `("model",)` |
| `attn.c_proj` | `[1024, 1024]` | row (shard in) | `("model", None)` |
| `attn.c_proj.bias` | `[1024]` | replicated | *(unsharded)* |
| `mlp.c_fc` | `[1024, 4096]` | column (shard out) | `(None, "model")` |
| `mlp.c_fc.bias` | `[4096]` | column | `("model",)` |
| `mlp.c_proj` | `[4096, 1024]` | row (shard in) | `("model", None)` |
| `mlp.c_proj.bias` | `[1024]` | replicated | *(unsharded)* |

LayerNorms (`ln_1`, `ln_2`), embeddings, and the final norm are **replicated**.

### Conv1D transpose (why specs are flipped)

HF GPT2 uses `Conv1D`, whose weight is stored **`[in_features, out_features]`** —
transposed vs `nn.Linear` (`[out, in]`). So relative to the Linear-based LLM
specs in `tests/benchmark/test_llms.py` (column = `("model", None)`), the GPT2
specs are flipped: **column-parallel = `(None, "model")`, row-parallel =
`("model", None)`**. All sharded dims divide by 2: heads 16→8, 1024/3072/4096
all even.

## CCL accounting

Per GPT2 block: 1 all-reduce after `attn.c_proj` + 1 after `mlp.c_proj` = **2
all-reduces/block**, over 30 blocks = **60 all-reduces per GPT forward pass**.

| Pipeline GPT stage | Forward passes | All-reduces | All-reduce payload |
|---|---|---|---|
| `gpt_codes` decode loop | 1 per generated token (~140) | ~60 × 140 ≈ **8400** | `[1, 1, 1024]` (tiny) |
| `gpt_latents` (full seq) | 1 | **60** | `[1, L, 1024]` |

The decode loop dominates CCL **count** (one all-reduce chain per token), but each
is a tiny `[1,1,1024]` payload; `gpt_latents` does 60 all-reduces on the larger
`[1, L, 1024]` activation. Non-GPT components emit **0 CCLs** (replicated).

This is the standard TP trade-off: it parallelizes the GPT compute across 2 chips
at the cost of these all-reduces. Whether it's a net win over 1 chip for XTTS
(small hidden size, batch 1, latency-bound decode) is exactly what the n300 run
should measure — TP can be dominated by all-reduce latency at this scale.

## Caveats — validate on hardware

1. **Fused QKV (main risk).** `c_attn` output is `[Q(all heads) | K(all heads) |
   V(all heads)]`, each 1024 wide. A plain contiguous shard of the 3072 output
   dim splits at 1536, cutting through the K block rather than along head
   boundaries; HF's `.split(1024, dim=2)` then forces a reshard, defeating clean
   head-parallel TP. The spec annotates the naive layout as a **starting point**.
   For a correct/efficient layout, either (a) reorder the QKV columns into
   per-device `[Q_half, K_half, V_half]` groups (Megatron's approach) and adjust
   the split, or (b) split `c_attn` into separate q/k/v projections and shard
   each `(None, "model")`. Verify from the TTIR whether GSPMD inserts an
   unexpected reshard/all-gather around the attention split.
2. **Per-stage device movement.** `pipeline.py` moves each component on/off the
   device to bound DRAM. TP sharding is applied to the shared `xtts.gpt`
   parameters on their first GPT stage; confirm the sharding annotation survives
   and that no host round-trip drops it.
3. **Mesh for other components.** Speaker/conditioning/HiFi-GAN run replicated on
   the `(1,2)` mesh. If they OOM or misbehave under SPMD, gate them to a single
   device or a sub-mesh.
4. **opt-level.** Independent of TP, `--opt-level 1` currently aborts at compile
   (`allocator.cpp: Unsupported buffer type!` in FullOp constraint validation),
   so TP runs stay at opt-level 0.

## Rejected alternatives

- **Sequence/activation parallelism on the GPT** — saves DRAM, not the objective
  here (the model fits); adds reshards. Reconsider only if TP OOMs.
- **Sharding the HiFi-GAN / conditioning encoder** — small relative to the GPT;
  the conv-heavy vocoder would add CCLs for little compute benefit. Replicate.
- **Data parallelism (`(2,1)` mesh)** — batch is 1 (single utterance), so DP
  gives no speedup for one synthesis; TP is the right axis.
