import gc
import faulthandler
import os
import torch

# Debug aid for DDP hangs: set DBETA_HANG_DEBUG=<seconds> to auto-dump every
# rank's full thread stacks to stderr after that many idle seconds (no sudo /
# ptrace / signals needed). e.g. DBETA_HANG_DEBUG=20 bash scripts/...
faulthandler.enable()
_hang_debug = os.environ.get("DBETA_HANG_DEBUG")
if _hang_debug:
    faulthandler.dump_traceback_later(int(_hang_debug), repeat=True)

from optimizers.scheduler import get_optimizer
from optimizers.ema import EMA

from dataloaders.build_dataloader import BuildDataLoader

from neural_networks.build_nn import BuildNN

from runners.train import run_train

from utils.checkpoint_manager import CheckpointManager
from utils.seed_setup import set_seed
from utils.gpu_setup import is_main, init_dist, cleanup, GPUSetup
from utils.dir_file import setup_experiment_folders
from utils.wandb_setup import setup_wandb, cleanup_wandb

from configs.config import get_args
from configs.constants import RUNS_FOLDER

# This return true so I guess our Pytorch/Machine
# automatically detects for flash attention SDP
# print("flash attention SDP enabled.", torch.backends.cuda.flash_sdp_enabled())
torch.set_float32_matmul_precision("high")

def main():
    mode = "pretrain"
    args = get_args(mode)
    args.mode = mode
    args.task = "pretrain"

    if args.distributed:
        init_dist()

    gc.collect()
    torch.cuda.empty_cache()

    try:
        if not args.dev:
            run_folder = setup_experiment_folders(
                f"{RUNS_FOLDER}/pretrain/{args.neural_network}",
                args,
            )
        if is_main() and not args.dev:
            print(f"Run folder: {run_folder}")
            if args.wandb:
                setup_wandb(args)
        set_seed(args.seed)
        build_dataloader = BuildDataLoader(args)
        dataloader = build_dataloader.build_dataloader()
        args.max_steps = len(dataloader) * args.epochs
        build_nn = BuildNN(args)
        nn_components = build_nn.build_nn(dataloader.dataset.data_representation)
        gpu_setup = GPUSetup(args)
        nn = gpu_setup.setup_gpu(nn_components["neural_network"], nn_components["find_unused_parameters"],
                                 static_graph=nn_components.get("static_graph", False))
        if args.dev:
            gpu_setup.print_model_device(nn, f"{args.neural_network}")
        optimizer = get_optimizer(args, nn)
        ema = EMA(nn, decay = args.ema_decay) if getattr(args, "ema", False) else None
        if args.dev:
            checkpoint_manager = None
        else:
            checkpoint_manager = CheckpointManager(run_folder, args)
        for epoch in range(args.epochs):
            train_result = run_train(nn, optimizer, dataloader, epoch, args, checkpoint_manager, ema)
            if checkpoint_manager and is_main():
                if checkpoint_manager.save_epoch(train_result["average_loss"]):
                    checkpoint_manager.save_checkpoint(nn, optimizer, epoch, -1, is_best=True, prefix="epoch_", ema=ema)
                if checkpoint_manager.stop_early():
                    if is_main():
                        print(f"Early stopping at epoch {epoch}")
                    break
        if is_main() and not args.dev:
            with open(f"{run_folder}/DONE.txt", "w") as _:
                pass
    finally:
        if args.distributed:
            cleanup()
        if is_main() and args.wandb:
            cleanup_wandb()


if __name__ == "__main__":
    main()