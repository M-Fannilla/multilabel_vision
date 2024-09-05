import tensorflow as tf
from tensorflow import keras
from transformers import TFViTModel, ViTConfig
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class VariableSizeImageClassifier(keras.Model):
    def __init__(
        self,
        num_classes: int,
        max_images: int = 32,
        image_size: int = 384,
        dropout_rate: float = 0.1,
        vit_model_name: str = "google/vit-base-patch16-384",
    ):
        super().__init__()
        self.max_images = max_images
        self.num_classes = num_classes
        self.image_size = image_size

        try:
            self.vit_config = ViTConfig.from_pretrained(vit_model_name)
            self.vit = TFViTModel.from_pretrained(
                vit_model_name, config=self.vit_config
            )
        except Exception as e:
            logger.error(f"Failed to load ViT model: {e}")
            raise

        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=1, key_dim=self.vit_config.hidden_size
        )
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.classifier = tf.keras.layers.Dense(num_classes, activation="sigmoid")

    @tf.function
    def call(
        self, inputs: tf.Tensor, training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        batch_size = tf.shape(inputs)[0]
        reshaped_inputs = tf.reshape(inputs, [-1, self.image_size, self.image_size, 3])

        vit_outputs = self.vit(reshaped_inputs, training=training).last_hidden_state[
            :, 0, :
        ]
        features = tf.reshape(vit_outputs, [batch_size, self.max_images, -1])

        attention_output, attention_weights = self.attention(
            features, features, return_attention_scores=True
        )
        aggregated_features = tf.reduce_mean(attention_output, axis=1)

        x = self.dropout(aggregated_features, training=training)
        outputs = self.classifier(x)

        return outputs, attention_weights

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "max_images": self.max_images,
                "image_size": self.image_size,
                "dropout_rate": self.dropout.rate,
            }
        )
        return config
