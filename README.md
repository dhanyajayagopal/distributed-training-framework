# Distributed Training Framework

A complete implementation of distributed training techniques used by AI companies to scale model training across multiple machines and GPUs.

## What I Built

I created a distributed training framework that demonstrates how companies like OpenAI, Anthropic, and Google train massive models. The framework includes:

**Core distributed training system** - A `DistributedTrainer` class that spawns multiple processes and coordinates communication between them using PyTorch's distributed backend.

**Data parallelism implementation** - Each process trains on different chunks of data but shares gradients to stay synchronized, allowing faster training on larger datasets.

**Model parallelism system** - For models too large to fit on one GPU, the framework splits layers across processes, implementing pipeline parallelism where data flows forward and gradients flow backward through the pipeline.

**Checkpointing and recovery** - A `CheckpointManager` that saves training state periodically and can resume from any checkpoint, essential for long-running training jobs.

**Training monitoring** - A logging system that tracks loss, timing, and other metrics across all processes.

## How It Works

### Basic Concept

Instead of training on one machine, I created a system where multiple processes work together:

- Each process represents a separate GPU or machine
- Processes communicate using PyTorch's distributed communication primitives
- All processes stay synchronized by sharing gradients and model updates

### Data Parallelism

```bash
python train_realistic.py
```

This demonstrates data parallelism where:
- Process 0 trains on samples 0-99
- Process 1 trains on samples 100-199  
- Both processes share their gradients using `all_reduce`
- The model learns from all data while training in parallel

### Model Parallelism

```bash
python train_pipeline_parallel.py
```

This shows model parallelism where:
- Process 0 owns the first half of the model layers
- Process 1 owns the second half of the model layers
- Data flows: Process 0 → Process 1 → output
- Gradients flow: Process 1 → Process 0 → input

### Checkpointing

```bash
python train_with_checkpoints.py
python train_resume.py
```

The checkpoint system:
- Automatically saves model state every few steps
- Can resume training from any checkpoint
- Handles the coordination needed in distributed settings