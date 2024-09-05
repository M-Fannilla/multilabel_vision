import wandb
from scripts.train import train_model
from src.config import ModelConfig, TrainingConfig
from src.data_processing import prepare_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    wandb.init()

    model_config = ModelConfig(dropout_rate=wandb.config.dropout_rate)
    training_config = TrainingConfig(
        batch_size=wandb.config.batch_size,
        num_epochs=wandb.config.num_epochs,
        learning_rate=wandb.config.learning_rate,
    )

    try:
        # Load your data here
        X, y = prepare_dataset(images, labels)
        train_model(model_config, training_config, X, y)
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        wandb.finish(exit_code=1)
        raise


if __name__ == "__main__":
    sweep_configuration = {
        "method": "bayes",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "val_accuracy"},
        "parameters": {
            "learning_rate": {"max": 0.01, "min": 0.0001},
            "dropout_rate": {"max": 0.5, "min": 0.1},
            "batch_size": {"values": [16, 32, 64]},
            "num_epochs": {"values": [10, 20, 30]},
        },
    }

    sweep_id = wandb.sweep(
        sweep=sweep_configuration, project="variable-size-image-classifier"
    )

    num_runs = 5  # You can adjust this value
    wandb.agent(sweep_id, function=main, count=num_runs)
