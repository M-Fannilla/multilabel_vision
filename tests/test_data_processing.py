import tensorflow as tf
import numpy as np
from src.data_processing import prepare_dataset


def test_prepare_dataset():
    # Create dummy data
    images = [
        [tf.random.normal((224, 224, 3)) for _ in range(3)],
        [tf.random.normal((224, 224, 3)) for _ in range(5)],
    ]
    labels = [[0, 1], [1, 2]]

    X, y = prepare_dataset(images, labels, image_size=224, max_images=4)

    assert X.shape == (2, 4, 224, 224, 3)
    assert y.shape == (2, 3)
    assert np.allclose(y[0], [1, 1, 0])
    assert np.allclose(y[1], [0, 1, 1])


if __name__ == "__main__":
    test_prepare_dataset()
    print("All tests passed!")
