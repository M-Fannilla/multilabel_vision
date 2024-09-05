import tensorflow as tf
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def prepare_dataset(
    images: List[List[tf.Tensor]],
    labels: List[List[int]],
    image_size: int = 384,
    max_images: int = 32,
) -> Tuple[tf.Tensor, tf.Tensor]:
    try:
        num_classes = len(set(label for label_list in labels for label in label_list))

        resized_images = [
            [tf.image.resize(img, (image_size, image_size)) for img in img_list]
            for img_list in images
        ]
        padded_images = tf.keras.preprocessing.sequence.pad_sequences(
            resized_images, maxlen=max_images, padding="post", dtype="float32"
        )

        label_vectors = [
            tf.reduce_sum(tf.one_hot(label_list, num_classes), axis=0)
            for label_list in labels
        ]

        return tf.convert_to_tensor(padded_images), tf.stack(label_vectors)
    except Exception as e:
        logger.error(f"Error in prepare_dataset: {e}")
        raise


def create_dataset(
    X: tf.Tensor,
    y: tf.Tensor,
    batch_size: int,
    augment: bool = False,
    shuffle_buffer: Optional[int] = None,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle_buffer:
        dataset = dataset.shuffle(shuffle_buffer)

    if augment:
        data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
            ]
        )
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
