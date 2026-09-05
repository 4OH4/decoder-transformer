import datetime
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tqdm

from dataset import create_dataset, load_tokenizer, SEQ_LEN
from model import create_model, model_config, create_causal_mask

BATCH_SIZE = 16
MODEL_FILE_NAME = "textgen_model.pth"

torch.set_float32_matmul_precision('high')

# Training configuration
train_config = {
    "n_epochs": 2,
    "lr": 0.0005,
    "warmup_steps": 2000,
    "clip_norm": 6.0,
    "val_every_n_steps": 2000,
}


@torch.no_grad()
def evaluate_model(model, dataloader, loss_fn, mask, device):
    """Full pass over the holdout set; returns mean cross-entropy loss."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        outputs = model(x, mask.unsqueeze(0))
        total_loss += loss_fn(outputs.view(-1, outputs.shape[-1]), y.view(-1)).item()
        n_batches += 1
    model.train(was_training)
    return total_loss / max(n_batches, 1)


def train_model(model, train_dataloader, val_dataloader, device):

    # load tokenizer
    tokenizer = load_tokenizer()

    # Set up for logging
    writer = SummaryWriter(f'runs/{datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")}')

    optimizer = optim.AdamW(model.parameters(), lr=train_config["lr"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[pad]"))
    mask = create_causal_mask(seq_len=SEQ_LEN, device=device)

    # Learning rate scheduling
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer=optimizer, start_factor=0.1, end_factor=1, total_iters=train_config["warmup_steps"]
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=train_config["n_epochs"]*len(train_dataloader) - train_config["warmup_steps"], eta_min=0
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer=optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[train_config["warmup_steps"]]
    )

    # training loop
    print(f"Training for {train_config['n_epochs']} with {len(train_dataloader)} steps per epoch")
    print(f"Validating every {train_config['val_every_n_steps']} steps on {len(val_dataloader)} holdout batches")
    best_val_loss = float('inf')
    global_step = 0
    for epoch in range(train_config["n_epochs"]):
        model.train()
        epoch_loss = 0

        # Mean training loss over the logging interval, so Loss/Train is comparable to Loss/Val
        interval_loss = 0
        interval_steps = 0

        progress_bar = tqdm.tqdm(train_dataloader, desc=f"Epoch: {epoch}/{train_config['n_epochs']}")
        for x, y in progress_bar:
            start_time = time.time()
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(x, mask.unsqueeze(0))

            loss = loss_fn(outputs.view(-1, outputs.shape[-1]), y.view(-1))

            # Backwards pass
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config["clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            interval_loss += loss.item()
            interval_steps += 1

            if global_step % 100 == 99:
                # Calculate Performance Metrics
                step_time = time.time() - start_time
                tokens_in_batch = BATCH_SIZE * model_config["max_seq_len"]
                throughput = tokens_in_batch / step_time

                train_loss = interval_loss / interval_steps
                interval_loss = 0
                interval_steps = 0

                # Calculate Perplexity safely
                try:
                    perplexity = math.exp(train_loss)
                except OverflowError:
                    perplexity = float('inf')

                writer.add_scalar('Meta/Epoch', epoch, global_step)
                writer.add_scalar('Loss/Train', train_loss, global_step)
                writer.add_scalar("Loss/Perplexity", perplexity, global_step)
                writer.add_scalar("Optimization/Grad_Norm", grad_norm.item(), global_step)
                writer.add_scalar("Hardware/Throughput_Tokens_Per_Sec", throughput, global_step)

                # Log GPU memory if CUDA is available
                if torch.cuda.is_available():
                    # Convert bytes to Gigabytes
                    allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                    writer.add_scalar("Hardware/GPU_Memory_Allocated_GB", allocated_gb, global_step)

            # Evaluate on the holdout set, and checkpoint on the best validation loss
            val_interval = train_config["val_every_n_steps"]
            if global_step % val_interval == val_interval - 1:
                val_loss = evaluate_model(model, val_dataloader, loss_fn, mask, device)

                try:
                    val_perplexity = math.exp(val_loss)
                except OverflowError:
                    val_perplexity = float('inf')

                writer.add_scalar("Loss/Val", val_loss, global_step)
                writer.add_scalar("Loss/Val_Perplexity", val_perplexity, global_step)
                print(f"Step {global_step}: val loss {val_loss:.4f} (ppl {val_perplexity:.2f})")

                if val_loss < best_val_loss:
                    # Save checkpoint
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), MODEL_FILE_NAME)
                    print(f"  new best val loss - saved {MODEL_FILE_NAME}")

            global_step += 1
            # Show loss in tqdm
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{train_config['n_epochs']}; Avg loss: {avg_loss:.4f}")

    writer.close()


if __name__ == "__main__":
    train_dataset, val_dataset = create_dataset()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # Fixed order and fixed batch shapes, so every validation pass is comparable
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                                 drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(device=device)

    train_model(model, train_dataloader, val_dataloader, device)
