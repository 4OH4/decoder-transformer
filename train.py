import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

from dataset import create_dataset, VOCAB_SIZE
from model import create_model

BATCH_SIZE = 32
MODEL_FILE_NAME = "textgen_model.pth"

# Training configuration
train_config = {
    "n_epochs": 2,
    "lr": 0.0005,
    "warmup_steps": 2000,
    "clip_norm": 6.0,
}

def create_causal_mask(seq_len, device):
    """Create a causal mask for autoregressive attention."""
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
    return mask


def train_model(model, dataloader, device):

    optimizer = optim.AdamW(model.parameters(), lr=train_config["lr"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.tokenizer.token_to_id("[pad]"))

    # Learning rate scheduling
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer=optimizer, start_factor=0.1, end_factor=1, total_iters=train_config["warmup_steps"]
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=train_config["n_epochs"]*len(dataloader) - train_config["warmup_steps"], eta_min=0
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer=optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[train_config["warmup_steps"]]
    )

    # training loop
    print(f"Training for {train_config['n_epochs']} with {len(dataloader)} steps per epoch")
    best_loss = float('inf')
    for epoch in range(train_config["n_epochs"]):
        model.train()
        epoch_loss = 0

        progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch: {epoch}/{train_config['n_epochs']}")
        for x, y in tqdm.tqdm(progress_bar, leave=False):
            x = x.to(device)
            y = y.to(device)

            # Create causal mask
            mask = create_causal_mask(x.shape[1], device=device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(x, mask.unsqueeze(0))

            loss = loss_fn(outputs.view(-1, outputs.shape[-1]), y.view(-1))

            # Backwards pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config["clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            # Show loss in tqdm
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{train_config['n_epochs']}; Avg loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            # Save checkpoint
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_FILE_NAME)


if __name__ == "__main__":
    dataset = create_dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(device=device)

    model = train_model(model, dataloader, device)
