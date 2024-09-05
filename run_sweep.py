import wandb

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

wandb.agent(sweep_id, function=main, count=5)
