import torch
from tqdm import tqdm
import wandb

from utils.gpu_setup import is_main, train_dev_break
from utils.runner_helpers import batch_to_device

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
    accum_steps = args.grad_accum_steps

    optimizer.zero_grad()
    for step, batch in enumerate(progress):
        batch = {k: batch_to_device(v, device) for k, v in batch.items()}

        out = nn(**batch)
        raw_loss = out.loss
        window_start = step - step % accum_steps
        window_size = min(accum_steps, total_steps_per_epoch - window_start)
        loss = raw_loss / window_size
        total_loss += raw_loss.item()
        total_steps += 1
        loss.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == total_steps_per_epoch:
            grad_clip = getattr(args, "grad_clip", 0.0)
            if grad_clip > 0:
                params = (p for p in nn.parameters() if p.grad is not None)
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step_and_update_lr()
            optimizer.zero_grad()
            if ema is not None:
                ema.update()
            if getattr(args, "wandb", False) and is_main():
                wandb.log({"train/step_loss": raw_loss.item(), "train/lr": optimizer.learning_rate, "epoch": epoch})
            if args.save_step and checkpoint_manager and is_main():
                if checkpoint_manager.save_step(step, total_steps_per_epoch):
                    checkpoint_manager.save_checkpoint(nn, optimizer, epoch, step, prefix="step_", ema=ema)

        if train_dev_break(getattr(args, "dev", False), batch, raw_loss.item()):
            break
        # if step > 4000:
        #     break

    average_loss = total_loss / total_steps if total_steps > 0 else float("inf")
    return {"average_loss": average_loss, "total_steps": total_steps}
