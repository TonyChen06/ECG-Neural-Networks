import torch
from tqdm import tqdm
import wandb

from utils.gpu_setup import is_main, train_dev_break
from utils.runner_helpers import batch_to_device
from neural_networks.xecg.augmentations import shuffle_baseline_wander_batched


def _unwrap(module):
    return module.module if hasattr(module, "module") else module


def _maybe_shuffle_baselines(batch: dict, args) -> dict:
    if not getattr(args, "xecg_shuffle_baseline_wander", False):
        return batch
    fs = float(getattr(args, "sf", 250))
    out = dict(batch)
    for key in ("global_signals", "local_signals"):
        if key not in out:
            continue
        x = out[key]  # (B, V, C, T)
        B, V, C, T = x.shape
        x_flat = x.reshape(B * V, C, T)
        x_flat = shuffle_baseline_wander_batched(x_flat, fs=fs, cutoff=0.5)
        out[key] = x_flat.view(B, V, C, T)
    return out


def _xecg_step_hook(nn_module, optimizer, args, global_step):
    inner = _unwrap(nn_module)
    if not hasattr(inner, "update_teacher"):
        return None
    max_steps = max(int(getattr(args, "max_steps", 1)), 1)
    ema_0 = getattr(args, "xecg_ema_start", 0.99)
    ema_1 = getattr(args, "xecg_ema_end", 1.0)
    beta = ema_0 + global_step * (ema_1 - ema_0) / max_steps
    beta = min(max(beta, 0.0), 1.0)
    inner.update_teacher(beta)
    wd_0 = getattr(args, "weight_decay", 1e-2)
    wd_1 = getattr(args, "xecg_final_wd", wd_0)
    wd = wd_0 + (wd_1 - wd_0) * (global_step / max_steps)
    inner_optim = getattr(optimizer, "optimizer", optimizer)
    for group in inner_optim.param_groups:
        if group.get("weight_decay", 0.0) != 0.0:
            group["weight_decay"] = wd
    return {"teacher_beta": beta, "wd": wd}


def run_train(
    nn,
    optimizer,
    dataloader,
    epoch,
    args,
    checkpoint_manager=None,
    ema=None,
):
    if getattr(args, "distributed", False) and hasattr(getattr(dataloader, "sampler", None), "set_epoch"):
        dataloader.sampler.set_epoch(epoch)
    show_progress = is_main()

    total_loss = 0
    total_steps = 0
    progress = tqdm(
        dataloader,
        desc=f"Training: {args.neural_network}; Task: {args.task};Epoch: {epoch}",
        disable=not show_progress,
        leave=False,
    )
    total_steps_per_epoch = len(dataloader)
    device = next(nn.parameters()).device
    is_xecg = args.neural_network == "xecg" and args.task == "pretrain"

    for step, batch in enumerate(progress):
        batch = {k: batch_to_device(v, device) for k, v in batch.items()}
        if is_xecg:
            batch = _maybe_shuffle_baselines(batch, args)

        optimizer.zero_grad()
        out = nn(**batch)
        loss = out.loss
        total_loss += loss.item()
        total_steps += 1
        loss.backward()
        grad_clip = getattr(args, "grad_clip", 0.0)
        if grad_clip > 0:
            params = (p for p in nn.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step_and_update_lr()
        if ema is not None:
            ema.update()
        global_step = epoch * total_steps_per_epoch + step
        hook = _xecg_step_hook(nn, optimizer, args, global_step) if is_xecg else None
        if getattr(args, "wandb", False) and is_main():
            log = {"train/step_loss": loss.item(), "train/lr": optimizer.learning_rate, "epoch": epoch}
            if is_xecg:
                log["train/compression"] = out.compression.item()
                log["train/expansion"] = out.expansion.item()
                log["train/patch"] = out.patch.item()
                if hook:
                    log["train/teacher_beta"] = hook["teacher_beta"]
                    log["train/wd"] = hook["wd"]
            wandb.log(log)
        if args.save_step and checkpoint_manager and is_main():
            if checkpoint_manager.save_step(step, total_steps_per_epoch):
                checkpoint_manager.save_checkpoint(nn, optimizer, epoch, step, prefix="step_", ema=ema)
        if train_dev_break(getattr(args, "dev", False), batch, loss.item()):
            break

    average_loss = total_loss / total_steps if total_steps > 0 else float("inf")
    return {"average_loss": average_loss, "total_steps": total_steps}
