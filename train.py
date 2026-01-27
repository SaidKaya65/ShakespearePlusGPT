"""
Training Script for Shakespeare GPT

This module implements the training loop for our GPT model.
Training takes approximately 15-30 minutes on a MacBook.

Key Concepts:
1. Training Loop - Forward pass, loss calculation, backward pass, optimizer step
2. Learning Rate Scheduling - Warmup + cosine decay
3. Gradient Clipping - Prevents exploding gradients
4. Evaluation - Periodic validation and sample generation
"""

import os
import time
import math
import torch
from model import GPT
from dataset import create_dataloaders

# ==============================================================================
# Configuration
# ==============================================================================

config = {
    # Model architecture
    "vocab_size": None,  # Will be set from tokenizer
    "embedding_dim": 384,  # Size of embeddings (Hidden size D)
    "num_heads": 6,  # Number of attention heads
    "num_layers": 6,  # Number of transformer blocks
    "block_size": 256,  # Maximum sequence length
    "dropout": 0.1,  # Dropout rate

    # Training hyperparameters
    "batch_size": 64,  # Number of sequences per batch
    "max_iters": 5000,  # Total training iterations
    "eval_interval": 500,  # Evaluate every N iterations
    "eval_iters": 200,  # Number of batches for evaluation
    "learning_rate": 3e-4,  # Peak learning rate
    "warmup_iters": 100,  # Learning rate warmup iterations

    # System
    "device": "mps" if torch.backends.mps.is_available() else "cpu",
    "checkpoint_dir": "checkpoints"
}


# ==============================================================================
# Learning Rate Schedule
# ==============================================================================

def get_learning_rate(iteration: int,
                      warmup_iters: int,
                      max_iters: int,
                      max_lr: float) -> float:
    """
    Calculate learning rate with warmup and cosine decay.

    1. Warmup phase: Linearly increase from 0 to max_lr
    2. Decay phase: Cosine decay from max_lr to min_lr (10% of max)
    """
    min_lr = max_lr * 0.1

    # 1. Warmup phase: linear increase
    if iteration < warmup_iters:
        return max_lr * (iteration / warmup_iters)

    # 2. After max_iters: return minimum
    if iteration > max_iters:
        return min_lr

    # 3. Decay phase: cosine annealing
    progress = (iteration - warmup_iters) / (max_iters - warmup_iters)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine_decay * (max_lr - min_lr)


# ==============================================================================
# Evaluation Function
# ==============================================================================

@torch.no_grad()
def evaluate(model, train_dataloader, val_dataloader, eval_iters, device):
    """Evaluate the model on training and validation sets."""
    # 1. Set model to evaluation mode
    model.eval()

    losses = {}

    # 2. Evaluate on both splits
    for split_name, dataloader in [("train", train_dataloader), ("val", val_dataloader)]:
        total_loss = 0.0
        num_batches = 0

        # 3. Iterate through batches
        for batch_idx, (x, y) in enumerate(dataloader):
            if batch_idx >= eval_iters:
                break

            # 4. Move data to device and forward pass
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)

            total_loss += loss.item()
            num_batches += 1

        # 5. Calculate average loss
        losses[split_name] = total_loss / num_batches

    # 6. Set model back to training mode
    model.train()

    return losses


# ==============================================================================
# Sample Generation Function
# ==============================================================================

@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt="ROMEO:", max_tokens=200):
    """Generate a text sample from the model."""
    model.eval()

    # Encode prompt and generate
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_tokens,
        temperature=0.8,
        top_k=40
    )

    model.train()
    return tokenizer.decode(output_ids[0])


# ==============================================================================
# Training Function
# ==============================================================================

def train():
    """Main training function."""

    print("=" * 60)
    print("Shakespeare GPT Training")
    print("=" * 60)
    print(f"\nDevice: {config['device']}")

    # =========================================================================
    # 1. Load Data
    # =========================================================================
    print("\n" + "-" * 60)
    print("Loading Data")
    print("-" * 60)

    train_dataloader, val_dataloader, tokenizer = create_dataloaders(
        batch_size=config["batch_size"],
        block_size=config["block_size"]
    )

    # Update vocab size from tokenizer
    config["vocab_size"] = tokenizer.vocab_size

    # =========================================================================
    # 2. Create Model
    # =========================================================================
    print("\n" + "-" * 60)
    print("Creating Model")
    print("-" * 60)

    model = GPT(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        block_size=config["block_size"],
        dropout=config["dropout"]
    )
    model = model.to(config["device"])

    # =========================================================================
    # 3. Create Optimizer
    # =========================================================================
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    # =========================================================================
    # 4. Create Checkpoint Directory
    # =========================================================================
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    # =========================================================================
    # 5. Training Loop
    # =========================================================================
    print("\n" + "-" * 60)
    print("Training")
    print("-" * 60)
    print(f"\nMax iterations: {config['max_iters']}")
    print(f"Eval interval: {config['eval_interval']}")
    print("Starting training...\n")

    train_iterator = iter(train_dataloader)
    best_val_loss = float('inf')
    start_time = time.time()

    for iteration in range(config["max_iters"]):

        # 5.1 Get Learning Rate
        lr = get_learning_rate(
            iteration=iteration,
            warmup_iters=config["warmup_iters"],
            max_iters=config["max_iters"],
            max_lr=config["learning_rate"]
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 5.2 Get Next Batch
        try:
            x, y = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            x, y = next(train_iterator)

        x, y = x.to(config["device"]), y.to(config["device"])

        # 5.3 Forward Pass
        _, loss = model(x, y)

        # 5.4 Backward Pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 5.5 Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 5.6 Optimizer Step
        optimizer.step()

        # 5.7 Evaluation
        if iteration % config["eval_interval"] == 0 or iteration == config["max_iters"] - 1:

            losses = evaluate(
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                eval_iters=config["eval_iters"],
                device=config["device"]
            )

            elapsed_time = time.time() - start_time
            print(f"Iter {iteration:5d} | "
                  f"Train Loss: {losses['train']:.4f} | "
                  f"Val Loss: {losses['val']:.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Time: {elapsed_time:.1f}s")

            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']

                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iteration': iteration,
                    'val_loss': best_val_loss,
                    'config': config
                }

                checkpoint_path = os.path.join(config["checkpoint_dir"], 'best.pt')
                torch.save(checkpoint, checkpoint_path)
                print(f"  -> New best model saved! (val_loss: {best_val_loss:.4f})")

            # Generate a sample
            if iteration > 0:
                print("\n--- Sample Generation ---")
                sample = generate_sample(model, tokenizer, config["device"])
                print(sample[:500])
                print("--- End Sample ---\n")

    # =========================================================================
    # 6. Save Final Model
    # =========================================================================
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': config["max_iters"],
        'val_loss': losses['val'],
        'config': config
    }

    torch.save(final_checkpoint, os.path.join(config["checkpoint_dir"], 'final.pt'))

    # =========================================================================
    # 7. Print Summary
    # =========================================================================
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {config['checkpoint_dir']}/")


if __name__ == "__main__":
    train()
