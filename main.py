import wandb
from src.config import ModelConfig, TrainingConfig
from src.data_processing import prepare_dataset
from scripts.train import train_model


def main():
    wandb.init(
        project="variable-size-image-classifier",
        config=wandb.config,
    )

    # Load your data here
    X, y = prepare_dataset(images, labels)

    model_config = ModelConfig(dropout_rate=wandb.config.dropout_rate)
    training_config = TrainingConfig(
        batch_size=wandb.config.batch_size,
        num_epochs=wandb.config.num_epochs,
        learning_rate=wandb.config.learning_rate,
    )

    train_model(model_config, training_config, X, y)

    wandb.finish()


if __name__ == "__main__":
    main()
