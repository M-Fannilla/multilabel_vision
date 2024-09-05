import matplotlib.pyplot as plt
import wandb
from wandb.sdk.wandb_run import Run
import numpy as np
from typing import Optional, List, Dict, Any


def visualize_attention(
    attention_weights: np.ndarray, num_samples: int = 10, title: Optional[str] = None
) -> None:
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights[:num_samples, :num_samples], cmap="viridis")
    plt.colorbar()
    plt.title(title or "Attention Weights Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Images")

    wandb.log({"attention_heatmap": wandb.Image(plt)})
    plt.close()


def plot_training_history(
    history: Dict[str, List[float]], metrics: Optional[List[str]] = None
) -> None:
    metrics = metrics or ["loss", "accuracy", "val_loss", "val_accuracy"]

    plt.figure(figsize=(12, 6))
    for metric in metrics:
        if metric in history:
            plt.plot(history[metric], label=metric)

    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.legend()

    wandb.log({"training_history": wandb.Image(plt)})
    plt.close()
