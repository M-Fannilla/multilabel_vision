import tensorflow as tf
import wandb
from wandb.integration.keras import WandbCallback
from src.model import VariableSizeImageClassifier
from src.data_processing import create_dataset
from src.visualization import visualize_attention, plot_training_history
from src.config import ModelConfig, TrainingConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    X: tf.Tensor,
    y: tf.Tensor,
):
    wandb.init(
        project="variable-size-image-classifier",
        config={
            "learning_rate": training_config.learning_rate,
            "dropout_rate": model_config.dropout_rate,
            "batch_size": training_config.batch_size,
            "num_epochs": training_config.num_epochs,
        },
    )

    try:
        model = VariableSizeImageClassifier(
            num_classes=model_config.num_classes,
            max_images=model_config.max_images,
            image_size=model_config.image_size,
            dropout_rate=model_config.dropout_rate,
            vit_model_name=model_config.vit_model_name,
        )
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        wandb.finish(exit_code=1)
        raise

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=training_config.learning_rate,
        decay_steps=training_config.decay_steps,
        decay_rate=training_config.decay_rate,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    train_dataset = create_dataset(
        X,
        y,
        training_config.batch_size,
        augment=True,
        shuffle_buffer=training_config.shuffle_buffer,
    )

    history = model.fit(
        train_dataset,
        epochs=training_config.num_epochs,
        validation_split=training_config.validation_split,
        callbacks=[WandbCallback()],
    )

    sample_batch = next(iter(train_dataset))
    _, attention_weights = model(sample_batch[0])
    visualize_attention(attention_weights.numpy(), title="Final Attention Weights")

    plot_training_history(history.history)

    wandb.finish()
    return history


if __name__ == "__main__":
    # Load your data here
    X, y = prepare_dataset(images, labels)

    model_config = ModelConfig()
    training_config = TrainingConfig()

    train_model(model_config, training_config, X, y)
