"""Curriculum learning callback for PyTorch Lightning."""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from src.data_utils.data_load import CurriculumLearningDataset


class CurriculumLearningCallback(Callback):
    """Callback to update amorphous fraction in curriculum learning dataset.

    This callback linearly increases the fraction of amorphous samples in the training
    dataset from start_fraction to end_fraction over the course of training.

    Args:
        start_fraction: Initial amorphous fraction (default: 0.0)
        end_fraction: Final amorphous fraction (default: 1.0)
        start_epoch: Epoch to start increasing fraction (default: 0)
        end_epoch: Epoch to reach end_fraction (default: None, uses max_epochs)
    """

    def __init__(
        self,
        start_fraction: float = 0.0,
        end_fraction: float = 1.0,
        start_epoch: int = 0,
        end_epoch: int = None,
    ):
        super().__init__()
        self.start_fraction = start_fraction
        self.end_fraction = end_fraction
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

        if not 0.0 <= start_fraction <= 1.0:
            raise ValueError(f"start_fraction must be in [0, 1], got {start_fraction}")
        if not 0.0 <= end_fraction <= 1.0:
            raise ValueError(f"end_fraction must be in [0, 1], got {end_fraction}")
        if start_fraction > end_fraction:
            raise ValueError(
                f"start_fraction ({start_fraction}) must be <= end_fraction ({end_fraction})"
            )

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Update amorphous fraction at the start of each training epoch."""
        current_epoch = trainer.current_epoch

        # Determine end epoch if not set
        end_epoch = self.end_epoch
        if end_epoch is None:
            end_epoch = trainer.max_epochs - 1

        # Calculate current fraction using linear interpolation
        if current_epoch < self.start_epoch:
            fraction = self.start_fraction
        elif current_epoch >= end_epoch:
            fraction = self.end_fraction
        else:
            # Linear interpolation between start and end
            progress = (current_epoch - self.start_epoch) / (end_epoch - self.start_epoch)
            fraction = self.start_fraction + progress * (self.end_fraction - self.start_fraction)

        # Update the dataset if it's a CurriculumLearningDataset
        datamodule = trainer.datamodule
        if datamodule is not None and hasattr(datamodule, 'train_dataset'):
            train_dataset = datamodule.train_dataset

            # Handle nested wrapping: Subset -> CurriculumLearningDataset
            # or just CurriculumLearningDataset directly
            curriculum_dataset = None

            if isinstance(train_dataset, CurriculumLearningDataset):
                curriculum_dataset = train_dataset
            elif hasattr(train_dataset, 'dataset') and isinstance(train_dataset.dataset, CurriculumLearningDataset):
                # train_dataset is Subset wrapping CurriculumLearningDataset
                curriculum_dataset = train_dataset.dataset

            if curriculum_dataset is not None:
                curriculum_dataset.amorphous_fraction = fraction
                trainer.logger.log_metrics(
                    {"curriculum/amorphous_fraction": fraction},
                    step=trainer.global_step
                )
            else:
                # Only warn once
                if current_epoch == 0:
                    print(
                        f"Warning: CurriculumLearningCallback is active but train_dataset "
                        f"does not contain a CurriculumLearningDataset (got {type(train_dataset).__name__}). "
                        f"Callback will have no effect."
                    )
