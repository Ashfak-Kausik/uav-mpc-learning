"""
Stage 5: train the residual MLP on collected rollout data.

Loads the combined dataset, splits into train/val (90/10), trains a
small MLP for ~50 epochs, saves the best model (by validation MSE).

Run:
    python3 02_train_model.py
"""

import os
import sys
import time
import numpy as np

import torch
from torch.utils.data import DataLoader, random_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from stage5_src.residual_dataset import ResidualDataset
from stage5_src.residual_model import ResidualMLP


# Paths
DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
MODELS_DIR = os.path.abspath(os.path.join(HERE, "..", "models"))
os.makedirs(MODELS_DIR, exist_ok=True)

COMBINED_PATH = os.path.join(DATA_DIR, "combined.npz")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "residual_best.pt")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "residual_final.pt")

# Training hyperparameters
BATCH_SIZE = 256
N_EPOCHS = 50
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.1
SEED = 42


def main():
    print("=" * 60)
    print("Stage 5 — train residual model")
    print("=" * 60)

    # Reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load dataset
    print(f"\nLoading dataset from {COMBINED_PATH}")
    full_dataset = ResidualDataset(COMBINED_PATH)
    n_total = len(full_dataset)
    n_val = int(n_total * VAL_FRACTION)
    n_train = n_total - n_val
    print(f"  Total: {n_total}, Train: {n_train}, Val: {n_val}")

    # Split (deterministic with seeded generator)
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val], generator=generator
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    # Initialize model
    model = ResidualMLP(state_dim=9, control_dim=4, hidden_dim=64,
                        n_hidden=2)
    print(f"  Model parameters: {model.n_parameters()}")

    # Fit normalization from the FULL dataset (not just train)
    # This is fine because normalization stats don't leak target info
    print("\n  Fitting input/output normalization...")
    all_inputs = full_dataset.all_inputs()
    all_residuals = full_dataset.all_residuals()
    model.fit_normalization(all_inputs, all_residuals)
    print(f"    Input std (per dim, first 13):")
    for i, label in enumerate(["px", "py", "pz", "vx", "vy", "vz",
                                "roll", "pitch", "yaw",
                                "T", "wx", "wy", "wz"]):
        print(f"      {label:6s}: {model.input_std[i].item():.6f}")
    print(f"    Residual std (per dim, 9):")
    for i, label in enumerate(["px", "py", "pz", "vx", "vy", "vz",
                                "roll", "pitch", "yaw"]):
        print(f"      {label:6s}: {model.residual_std[i].item():.6f}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    # Training loop
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    t_start = time.time()
    print(f"\nTraining for {N_EPOCHS} epochs (batch size {BATCH_SIZE}, lr={LEARNING_RATE})")
    print(f"{'Epoch':>6} {'Train MSE':>14} {'Val MSE':>14} {'Time':>8}")

    for epoch in range(N_EPOCHS):
        # Train
        model.train()
        train_losses = []
        for batch in train_loader:
            inputs = batch["input"]
            targets = batch["residual"]
            optimizer.zero_grad()
            pred = model(inputs)
            loss = loss_fn(pred, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"]
                targets = batch["residual"]
                pred = model(inputs)
                loss = loss_fn(pred, targets)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)

        elapsed = time.time() - t_start
        print(f"  {epoch+1:>4d} {train_loss:>14.7f} {val_loss:>14.7f} {elapsed:>6.1f}s")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
            }, BEST_MODEL_PATH)

    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": N_EPOCHS,
        "val_loss": history["val_loss"][-1],
        "history": history,
    }, FINAL_MODEL_PATH)

    print(f"\nTraining complete in {time.time() - t_start:.1f} s")
    print(f"  Best val MSE: {best_val_loss:.7f} (saved to {BEST_MODEL_PATH})")
    print(f"  Final val MSE: {history['val_loss'][-1]:.7f}")

    # Plot training curve
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs = np.arange(1, N_EPOCHS + 1)
        ax.plot(epochs, history["train_loss"], label="train", color="C0")
        ax.plot(epochs, history["val_loss"], label="val", color="C1")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_yscale("log")
        ax.set_title("Residual model training curve")
        ax.legend()
        ax.grid(alpha=0.3)
        plot_path = os.path.join(HERE, "training_curve.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        print(f"  Training curve saved to {plot_path}")
    except ImportError:
        print("  matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()