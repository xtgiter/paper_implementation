"""Training utilities: metrics tracking and visualisation."""

import matplotlib.pyplot as plt


class AverageMeter:
    """Tracks the running average and current value of a scalar metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def plot_history(history: dict, save_path: str = "training.png"):
    """Plot training curves from a history dict.

    `history` is expected to have keys like 'train_loss', 'test_acc', etc.
    """
    epochs = range(1, len(next(iter(history.values()))) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    if "val_acc" in history:
        axes[1].plot(epochs, history["val_acc"], label="Val Acc")
    if "test_acc" in history:
        axes[1].plot(epochs, history["test_acc"], label="Test Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Training curves saved to {save_path}")
