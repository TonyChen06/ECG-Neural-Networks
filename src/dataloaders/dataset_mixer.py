from dataloaders.dataset.base_dataset import BaseDataset

from dataloaders.data_representation.signal import Signal
from dataloaders.data_representation.bpe_symbolic import BPESymbolic
from dataloaders.data_representation.multi_view_signal import MultiViewSignal
from dataloaders.task.forecasting import Forecasting
from dataloaders.task.pretrain import Pretrain
from dataloaders.dataset.base_dataset import load_base_dataset

from utils.dir_file import DirFileManager
from utils.gpu_setup import is_main

from configs.constants import BASE_DATASETS, ALLOWED_DATA

class DatasetMixer:
    def __init__(
        self,
        args,
    ):
        self.args = args
        self.dfm = DirFileManager()

    def build_torch_dataset(self, ):
        data_representation = self.build_data_representation()
        task = self.build_task()
        datasets = []

        for data_name in self.args.data:
            if data_name in BASE_DATASETS:
                dataset = load_base_dataset(data_name, self.args)
            datasets.extend(dataset)
        if is_main(): print(f"Length of Dataset: {len(datasets)}")
        return BaseDataset(datasets, data_representation, task, self.args)

    def build_data_representation(self):
        if is_main():
            print(f"Using {self.args.data_representation} representation")
        if self.args.data_representation == "signal":
            return Signal(self.args)
        elif self.args.data_representation == "multi_view_signal":
            return MultiViewSignal(self.args)
        elif self.args.data_representation == "bpe_symbolic":
            vocab, merges = self.dfm.open_tokenizer(self.args.bpe_tokenizer_path)
            return BPESymbolic(vocab, merges, self.args)
        raise ValueError(f"Unknown data representation: {self.args.data_representation}")

    def build_task(self):
        if self.args.task in ["pretrain", "generation", "reconstruction"]:
            return Pretrain(self.args)
        elif self.args.task == "forecasting":
            return Forecasting(self.args)
        raise ValueError(f"Unknown task type: {self.args.task}")
    
    def assert_allowed_datasets(
        self,
    ):
        for d in self.args.data:
            if d not in ALLOWED_DATA:
                raise ValueError(f"Invalid dataset: {d}")