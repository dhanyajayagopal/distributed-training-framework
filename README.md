# Distributed Training Framework

A small, educational distributed training framework built on PyTorch. It
demonstrates the same techniques used by large AI labs to train models
across multiple GPUs / machines — data parallelism, pipeline parallelism,
mixed precision, gradient accumulation, checkpointing — in a codebase
small enough to read in one sitting.

## What's in the box

| Feature | Where it lives |
| --- | --- |
| Process-group launch (single- or multi-process, torchrun compatible) | `trainer/distributed.py` |
| YAML config → dataclass | `trainer/config.py` |
| Unified training engine (DDP, AMP, grad accumulation, grad clipping, resume) | `trainer/engine.py` |
| Pipeline parallelism with GPipe-style micro-batching | `trainer/pipeline.py` |
| Atomic checkpoints + `latest` pointer + RNG state | `trainer/checkpoint.py` |
| Cross-rank loss aggregation, throughput, JSONL logs | `trainer/logger.py` |
| CLI: `python -m trainer --config ...` | `trainer/cli.py` |
| Example configs / scripts | `configs/`, `examples/` |
| Tests (ckpt round-trip, DDP invariants, pipeline correctness) | `tests/` |
| GitHub Actions CI (lint + tests on Ubuntu) | `.github/workflows/ci.yml` |

## Quick start

```bash
# 1. Create a virtualenv and install (CPU torch is enough for everything here)
python -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -e ".[dev]"

# 2. Train (spawns 2 worker processes, data-parallel, synthetic regression)
python -m trainer --config configs/ddp_synthetic.yaml

# 3. Try pipeline parallelism
python -m trainer --config configs/pipeline_synthetic.yaml

# 4. Real data (downloads MNIST on first run; needs torchvision)
pip install torchvision
python -m trainer --config configs/ddp_mnist.yaml
```

Every run writes:

- `logs/training_log.jsonl` — one JSON object per logged step (loss,
  throughput, learning rate, elapsed time).
- `checkpoints/checkpoint_step_N.pt` and `checkpoints/latest.pt` — the
  `latest` pointer means you can resume without knowing the step number.

Resume from the most recent checkpoint:

```bash
python -m trainer --config configs/ddp_synthetic.yaml --resume-from latest
```

## How it works (in plain words)

### Data parallelism (`strategy: ddp`)

Every worker process holds a full copy of the model. They each read a
different slice of the dataset, compute gradients, and then average the
gradients across all workers via `all_reduce`. After that they all
apply the optimiser step, so they end up with identical weights again.
The framework uses PyTorch's `DistributedDataParallel` which bucket-reduces
gradients and overlaps communication with backward computation — much
faster than the hand-rolled version this repo started with.

### Pipeline parallelism (`strategy: pipeline`)

For models too big to fit on one device, layers are split across workers:
rank 0 owns layers 0..k, rank 1 owns layers k..m, and so on. Data flows
forward through the ranks, gradients flow backward. To keep the ranks busy
instead of waiting on each other, each batch is split into
`num_micro_batches` chunks and they're pipelined through the stages.

Communication uses `dist.send` / `dist.recv`. Each tensor is preceded by
its rank (number of dimensions) and its shape so any shape works — the
original implementation hard-coded 4D tensors and silently broke for
everything else.

### Checkpointing

- Saves model, optimiser, scheduler, AMP scaler, step, and every RNG
  state (python / numpy / torch / CUDA). Resume is truly deterministic.
- Writes atomically (`.tmp` + `os.replace`) so a crash during save cannot
  corrupt the file.
- Maintains a `latest.pt` pointer so callers can say "load whatever the
  last one was".
- Only rank 0 writes; every rank calls `dist.barrier()` so no rank
  proceeds past a save until the file is on disk.

### Mixed precision & gradient accumulation

`precision: fp16|bf16` enables `torch.autocast`. `grad_accum_steps: N`
splits the optimiser step across N micro-steps; under DDP the intermediate
backwards use `model.no_sync()` to avoid wasted all-reduces.

## Testing

### Fast path (every PR should pass this)

```bash
pytest -q
```

This runs in <10 seconds on CPU and covers:

- **`test_config.py`** — YAML config loads into the right dataclass.
- **`test_checkpoint.py`** — save/load round-trip, atomic writes, the
  `latest` pointer, and that non-zero ranks refuse to write.
- **`test_models_data.py`** — MLP shapes, synthetic data has a learnable
  signal (linear regression reaches low loss), labels are in range.
- **`test_distributed.py`** — `DistributedTrainer` actually spawns workers
  and all-reduce returns the expected sum.
- **`test_pipeline.py`** — pipeline stages concatenate to the original
  MLP, remainder layers are distributed, invalid splits error, full
  end-to-end pipeline run completes and writes logs.
- **`test_ddp_parity.py`** — DDP training writes a checkpoint, loss
  actually decreases, and every rank sees identical final weights after
  loading the shared checkpoint.

On macOS, `mp.spawn` under pytest can be flaky, so the multi-process tests
auto-skip locally and run in CI. To force them locally:

```bash
CI=true pytest -q
```

### Smoke tests by hand

If you want to see each path run end to end with logs:

```bash
# DDP
python -m trainer --config configs/ddp_synthetic.yaml --max-steps 30
# expect loss to drop, e.g. 8.99 -> 4.10 over 20 steps

# Pipeline
python -m trainer --config configs/pipeline_synthetic.yaml --max-steps 15

# MNIST (requires torchvision)
python -m trainer --config configs/ddp_mnist.yaml --max-steps 100
```

After a run, inspect the logs:

```bash
tail -n 5 logs/training_log.jsonl
```

Each line looks like:

```json
{"step": 10, "loss": 8.14, "elapsed_s": 0.03, "samples_per_s": 17327.9, "world_size": 2, "timestamp": 1766..., "lr": 0.01}
```

### What to look for

- **Loss decreases** on the synthetic regression task. The synthetic
  target is a noisy linear function, so a multi-layer MLP should easily
  get loss well under 1.0 within 100 steps.
- **`samples_per_s` is greater than 1× world size**. If throughput is
  ≤1×, communication is dominating compute (fine for tiny models, a
  red flag for bigger ones).
- **Checkpoints round-trip.** Train a bit, kill the process, resume
  from `latest` — the loss curve should continue smoothly rather than
  jumping back up.
- **All ranks end with the same weights** (the `test_all_ranks_see_same_final_weights`
  test asserts this programmatically).

## Launching with torchrun (multi-node / multi-GPU)

```bash
torchrun --nproc-per-node=4 -m trainer --config configs/ddp_mnist.yaml --use-torchrun
```

`--use-torchrun` tells the CLI to read `RANK`, `WORLD_SIZE`, `LOCAL_RANK`,
`MASTER_ADDR`, and `MASTER_PORT` from the environment instead of
spawning processes itself.

## Layout

```
trainer/
  __init__.py        public API
  cli.py             python -m trainer --config ...
  config.py          TrainConfig dataclass + YAML loader
  data.py            synthetic + MNIST datasets, DistributedSampler
  distributed.py     DistributedTrainer, init_from_env, DistEnv
  engine.py          train_ddp, train_pipeline, run_training
  checkpoint.py      CheckpointManager (atomic, latest, RNG)
  logger.py          TrainingLogger (all-reduce loss, throughput)
  models.py          MLP, pipeline stage builder
  pipeline.py        PipelineStage, send/recv helpers

configs/             YAML configs
examples/            Thin wrappers around the CLI
tests/               pytest suite
.github/workflows/   CI
```

## Ideas for next steps

- **ZeRO-1** via `torch.distributed.optim.ZeroRedundancyOptimizer` (~5
  lines) to shard optimiser state.
- **FSDP** for full weight sharding on big models.
- **1F1B pipeline schedule** — reduces activation memory vs. GPipe.
- **TensorBoard / W&B logger** behind the existing `TrainingLogger`
  interface.
- **Elastic training** via `torchrun` + automatic restart on failure.
