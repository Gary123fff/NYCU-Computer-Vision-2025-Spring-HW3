"""
Module for plotting training loss and validation mAP curves.
"""

import matplotlib.pyplot as plt


def plot_loss_and_map(train_losses, val_maps):
    """
    Plots training loss and validation mAP curves over epochs.

    Args:
        train_losses (list of float): List of training loss values.
        val_maps (list of float): List of validation mAP values.
    """
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, train_losses, 'b-o', label='Training Loss')
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(epochs, val_maps, 'g-s', label='Validation mAP')
    axes[1].set_title("Validation mAP")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mAP")
    axes[1].grid(True)
    axes[1].legend()

    fig.suptitle("Training Progress: Loss and mAP", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig("swin_model_fpn+head.png")
    plt.close()
