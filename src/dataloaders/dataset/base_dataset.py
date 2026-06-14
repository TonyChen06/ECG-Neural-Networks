from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os
import re
import numpy as np
import random

from configs.constants import DATA_DIR

from utils.dir_file import DirFileManager

# MIMIC-IV preprocessed filenames look like:
#   files_p1000_p10000032_s40689238_40689238_0.npy
# where the SECOND p-field (p10000032) is the patient id.
_PATIENT_ID_RE = re.compile(r"_p\d+_p(\d+)_")


def _parse_patient_id(path):
    m = _PATIENT_ID_RE.search(os.path.basename(path))
    return m.group(1) if m else None


class BaseDataset(Dataset):
    def __init__(self, data, data_representation, task, args):
        self.data = data
        self.args = args
        self.data_representation = data_representation
        self.task = task
        self.dfm = DirFileManager()

        self.patient_pair = (
            args.neural_network == "xecg"
            and getattr(args, "xecg_patient_pair", False)
            and "train" in args.mode
        )
        if self.patient_pair:
            self._build_patient_index()

    def _build_patient_index(self):
        index = {}
        unparsed = 0
        for path in self.data:
            pid = _parse_patient_id(path)
            if pid is None:
                pid = f"__solo_{unparsed}"  # unparseable filename -> its own singleton patient
                unparsed += 1
            index.setdefault(pid, []).append(path)
        self.patient_to_records = index
        self.patients = list(index.keys())

    def __len__(self):
        return len(self.patients) if self.patient_pair else len(self.data)

    def __getitem__(self, index):
        if self.patient_pair:
            return self._getitem_patient_pair(index)
        npy_file = self.dfm.open_npy(self.data[index])
        if self.args.augment:
            npy_file["ecg"] = self.augment_ecg(npy_file["ecg"])

        transformed_data = self.data_representation(npy_file)
        out = self.task(transformed_data)
        return out

    def _getitem_patient_pair(self, index):
        records = self.patient_to_records[self.patients[index]]
        # Sample up to n_global distinct recordings; views cycle over them (modulo)
        # exactly as in bench-xecg's ECGCODEDataset.
        n_sample = min(self.args.xecg_n_global, len(records))
        if len(records) > n_sample:
            records = [records[i] for i in np.random.choice(len(records), n_sample, replace=False)]
        signals = []
        for path in records:
            npy_file = self.dfm.open_npy(path)
            if self.args.augment:
                npy_file["ecg"] = self.augment_ecg(npy_file["ecg"])
            transformed = self.data_representation(npy_file)
            signals.append(np.asarray(transformed["transformed_data"], dtype=np.float32))
        return self.task.xecg_patient_multiview(signals)
    
    def augment_ecg(self, signal):
        if random.random() < 0.25:
            noise_level = 0.05
            noise = np.random.normal(0, noise_level * np.std(signal), signal.shape)
            perturbed_signal = signal + noise

            if random.random() < 0.25:
                wander_amplitude = 0.07 * np.max(np.abs(signal))
                wander = wander_amplitude * np.sin(np.linspace(0, random.randint(1, 5) * np.pi, signal.shape[1]))
                wander = np.tile(wander, (signal.shape[0], 1))
                perturbed_signal += wander

            return perturbed_signal
        return signal
    
def load_base_dataset(data, args):
    saved_path = f"{DATA_DIR}/{data}/preprocessed_{args.segment_len}"
    saved_dir = sorted([os.path.join(saved_path, f) for f in os.listdir(saved_path)])
    if args.task in ["generation", "reconstruction", "forecasting"]:
        train, test = train_test_split(saved_dir, train_size = 0.7, random_state=args.seed)
    elif args.task == "pretrain":
        train = saved_dir
        test = saved_dir
    if "train" in args.mode:
        return train
    else:
        return test