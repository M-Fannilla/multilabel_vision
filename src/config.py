from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    num_classes: int = 1000
    max_images: int = 32
    image_size: int = 384
    dropout_rate: float = 0.1
    vit_model_name: str = "google/vit-base-patch16-384"

    def __post_init__(self):
        assert 0 <= self.dropout_rate <= 1, "dropout_rate must be between 0 and 1"
        assert self.num_classes > 0, "num_classes must be positive"
        assert self.max_images > 0, "max_images must be positive"
        assert self.image_size > 0, "image_size must be positive"


@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-3
    decay_steps: int = 10000
    decay_rate: float = 0.9
    validation_split: float = 0.2
    shuffle_buffer: Optional[int] = 1000

    def __post_init__(self):
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.decay_steps > 0, "decay_steps must be positive"
        assert 0 < self.decay_rate <= 1, "decay_rate must be between 0 and 1"
        assert (
            0 <= self.validation_split < 1
        ), "validation_split must be between 0 and 1"
        assert (
            self.shuffle_buffer is None or self.shuffle_buffer > 0
        ), "shuffle_buffer must be positive or None"
