from src.dataset import GPT2Dataset
from src.model import GPT2, GPTConfig
import torch
import math
import time
import os


dataset = "fineweb"
total_batch_size = 524288 # (closest power of 2 to 0.5mil, 2**19)
B = 64 # micro batch size
T = 1024 # sequence length

# uncomment for testing on smaller dataset
# dataset = "shakespeare"
# total_batch_size = 4096
# B = 16 # micro batch size
# T = 16 # sequence length


ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    assert torch.cuda.is_available()
    torch.distributed.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    

assert total_batch_size % (B*T*ddp_world_size) == 0
grad_accumulation_steps = total_batch_size // (B*T*ddp_world_size) 
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"grad accumulation steps: {grad_accumulation_steps}")


torch.set_float32_matmul_precision("high")

model: GPT2 = GPT2(GPTConfig(vocab_size=50304))
model.to(DEVICE)
if ddp:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank], broadcast_buffers=False)
model = torch.compile(model)
raw_model = model.module if ddp else model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715 # 375e6 / 2**19 as per GPT-3
n_epochs = 1
max_steps = n_epochs*len(GPT2Dataset(B=B, T=T, world_size=ddp_world_size, process_rank=0, split="train", dataset=dataset))


def get_lr(i):
    if i < warmup_steps:
        return max_lr * (i+1) / warmup_steps
    if i > max_steps:
        return min_lr
    decay_ratio = (i - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=DEVICE)

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.csv")
with open(log_file, "w") as f: 
    f.write("step,train_loss,val_loss\n")


val_loader = torch.utils.data.DataLoader(
    GPT2Dataset(
        B=B, 
        T=T, 
        world_size=ddp_world_size, 
        process_rank=ddp_local_rank, 
        split="val",
        dataset=dataset
    ),
    shuffle=True
)
val_it = iter(val_loader)


n_steps = 0
for epoch in range(n_epochs):

    # each epoch, shuffle the training data
    train_loader = torch.utils.data.DataLoader(
        GPT2Dataset(
            B=B, 
            T=T, 
            world_size=ddp_world_size, 
            process_rank=ddp_local_rank, 
            split="train",
            dataset=dataset
        ),
        shuffle=True
    )
    train_it = iter(train_loader)

    while True:
        t0 = time.time()

        val_loss_accum = torch.tensor(torch.nan)
        if n_steps % 250 == 0 or n_steps == len(train_loader) - 1:

            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 50
                for _ in range(val_loss_steps):

                    try:
                        x_val, y_val = next(val_it)
                    except StopIteration:
                        val_loader = torch.utils.data.DataLoader(
                            GPT2Dataset(
                                B=B, 
                                T=T, 
                                world_size=ddp_world_size, 
                                process_rank=ddp_local_rank, 
                                split="val",
                                dataset=dataset
                            ),
                            shuffle=True
                        )
                        val_it = iter(val_loader)
                        x_val, y_val = next(val_it)

                    x_val, y_val = x_val.squeeze(0).to(DEVICE), y_val.squeeze(0).to(DEVICE)
                    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                        logits, loss = model(x_val, y_val)
                    loss /= val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                torch.distributed.all_reduce(val_loss_accum, op=torch.distributed.ReduceOp.AVG) # average val loss accross all processes
            if master_process:
                print(f"step {n_steps} | val loss {val_loss_accum.item():.4f}")

        
        if master_process and n_steps > 0 and n_steps % 5000 == 0:
            checkpoint_path = os.path.join(log_dir, f"model_{n_steps}.pt")
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": raw_model.config,
                "step": n_steps,
                "val_loss": val_loss_accum.item()
            }
            print(f"saving checkpoint at step {n_steps}")
            torch.save(checkpoint, checkpoint_path)

        
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.
        epoch_done = False

        for micro_step in range(grad_accumulation_steps):

            try:
                x_train, y_train = next(train_it)
            except StopIteration:
                epoch_done = True
                break

            x_train, y_train = x_train.squeeze(0).to(DEVICE), y_train.squeeze(0).to(DEVICE)
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                logits, loss = model(x_train, y_train)
            loss /= grad_accumulation_steps
            loss_accum += loss.detach()
            if ddp:
                model.require_backward_grad_sync = micro_step == grad_accumulation_steps - 1
            loss.backward()

        if epoch_done:
            break

        if ddp:
            torch.distributed.all_reduce(loss_accum, op=torch.distributed.ReduceOp.AVG) # average loss accross all processes
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip norm
        lr = get_lr(n_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0)
        tokens_per_sec = B * T * grad_accumulation_steps * ddp_world_size / dt

        if master_process:
            print(f"step {n_steps:4d} | train_loss: {loss_accum.item():.6f} | val_loss: {val_loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tokens/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{n_steps},{loss_accum.item()},{val_loss_accum.item()}\n")

        n_steps += 1

with torch.no_grad():
    val_loss_accum = 0.0
    val_loss_steps = 50
    for _ in range(val_loss_steps):

        try:
            x_val, y_val = next(val_it)
        except StopIteration:
            val_loader = torch.utils.data.DataLoader(
                GPT2Dataset(
                    B=B, 
                    T=T, 
                    world_size=ddp_world_size, 
                    process_rank=ddp_local_rank, 
                    split="val",
                    dataset=dataset
                ),
                shuffle=True
            )
            val_it = iter(val_loader)
            x_val, y_val = next(val_it)

        x_val, y_val = x_val.squeeze(0).to(DEVICE), y_val.squeeze(0).to(DEVICE)
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            logits, loss = model(x_val, y_val)
        loss /= val_loss_steps
        val_loss_accum += loss.detach()
if ddp:
    torch.distributed.all_reduce(val_loss_accum, op=torch.distributed.ReduceOp.AVG) # average val loss accross all processes

if master_process:
    print(f"final val loss: {val_loss_accum.item()}")
    checkpoint = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": raw_model.config,
        "step": n_steps,
        "val_loss": val_loss_accum.item()
    }
    print(f"saving checkpoint at step {n_steps}")
    torch.save(checkpoint, os.path.join(log_dir, "model_final.pt"))

if ddp:
    torch.distributed.destroy_process_group()
