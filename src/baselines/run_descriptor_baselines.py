import os
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

sys.path.append(os.getcwd())

from src.baselines.descriptor_baselines import run_descriptor_baseline


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "configs"),
    config_name="descriptor_baselines.yaml",
)
def main(cfg: DictConfig) -> None:
    run_dir = Path(HydraConfig.get().run.dir)
    metrics, _summary = run_descriptor_baseline(cfg, output_dir=run_dir)
    print("Descriptor baseline evaluation completed.")
    for name in sorted(metrics.keys()):
        print(f"{name}: {metrics[name]:.8f}")


if __name__ == "__main__":
    if not any(arg.startswith("hydra.run.dir=") for arg in sys.argv):
        sys.argv.append("hydra.run.dir=output/${now:%Y-%m-%d}/${now:%H-%M-%S}")
    main()
