"""Training loop for deep learning models.

Handles a single train/val cycle with early stopping, logging,
and best-model checkpointing (in memory).
"""

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    lr: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 8
    max_epochs: int = 200
    patience: int = 20
    class_weights: list[float] | None = None
    device: str = "cpu"


@dataclass
class TrainingHistory:
    """Records per-epoch metrics."""
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    elapsed_sec: float = 0.0


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)
        mask = batch.get("mask")
        lengths = batch.get("length")

        if mask is not None:
            mask = mask.to(device)
        if lengths is not None:
            lengths = lengths.to(device)

        optimizer.zero_grad()
        logits = model(features, lengths=lengths, mask=mask)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        n_samples += labels.size(0)

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on validation set. Returns loss, accuracy, predictions, true labels."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)
        mask = batch.get("mask")
        lengths = batch.get("length")

        if mask is not None:
            mask = mask.to(device)
        if lengths is not None:
            lengths = lengths.to(device)

        logits = model(features, lengths=lengths, mask=mask)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = max(len(all_labels), 1)
    avg_loss = total_loss / n
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    verbose: bool = True,
) -> tuple[nn.Module, TrainingHistory]:
    """Full training loop with early stopping.

    Returns the best model (by val loss) and training history.
    """
    device = torch.device(config.device)
    model = model.to(device)

    # Loss function with optional class weights
    if config.class_weights is not None:
        weights = torch.tensor(config.class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    history = TrainingHistory()
    best_state = None
    wait = 0
    start_time = time.time()

    for epoch in range(config.max_epochs):
        # Update augmentation epoch if dataset supports it
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_accuracy.append(val_acc)

        if val_loss < history.best_val_loss:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if verbose and (epoch % 20 == 0 or epoch == config.max_epochs - 1 or wait == 0):
            marker = " *" if wait == 0 else ""
            print(
                f"  Epoch {epoch:3d}: train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}{marker}"
            )

        if wait >= config.patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch} (best: {history.best_epoch})")
            break

    history.elapsed_sec = time.time() - start_time

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    return model, history