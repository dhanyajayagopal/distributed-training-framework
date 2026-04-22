"""Pipeline-parallel training on synthetic classification data.

Run:
    python examples/pipeline_synthetic.py
"""
from pathlib import Path

from trainer import DistributedTrainer, load_config, run_training

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "pipeline_synthetic.yaml"


def main() -> None:
    cfg = load_config(CONFIG)
    trainer = DistributedTrainer(world_size=cfg.world_size, backend=cfg.backend)
    trainer.run(run_training, cfg)


if __name__ == "__main__":
    main()
