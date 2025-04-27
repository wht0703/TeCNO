from pathlib import Path
from torch.utils.data import Dataset
import pickle
import numpy as np
import pandas as pd


class Cataract1kHelper(Dataset):
    def __init__(self, hparams, data_p, dataset_split=None):
        assert dataset_split != None
        self.data_p = data_p
        assert hparams.data_root != ""
        self.data_root = Path(hparams.data_root)
        self.number_vids = len(self.data_p)
        self.dataset_split = dataset_split
        self.factor_sampling = hparams.factor_sampling

    def __len__(self):
        return self.number_vids

    def __getitem__(self, index):
        vid_id = index
        p = self.data_root / self.data_p[vid_id]
        unpickled_x = pd.read_pickle(p)
        stem = np.asarray(unpickled_x[0],
                          dtype=np.float32)[::self.factor_sampling]
        y_hat = np.asarray(unpickled_x[1],
                           dtype=np.float32)[::self.factor_sampling]
        y = np.asarray(unpickled_x[2])[::self.factor_sampling]
        return stem, y_hat, y


class Cataract1k():
    def __init__(self, hparams):
        self.name = "Cholec80Pickle"
        self.hparams = hparams
        self.class_labels = [
            "Incision",
            "Viscoelastic",
            "Capsulorhexis",
            "Hydrodissection",
            "Phacoemulsification",
            "IrrigationAspiration",
            "CapsulePolishing",
            "LensImplantation",
            "LensPositioning",
            "AnteriorChamberFlushing",
            "ViscoelasticSuction",
            "TonifyingAntibiotics",
            "Idle"
        ]
        self.out_features = self.hparams.out_features
        self.features_per_seconds = hparams.features_per_seconds
        hparams.factor_sampling = (int(1 / hparams.features_subsampling))
        print(
            f"Subsampling features: 25features_ps --> {hparams.features_subsampling}features_ps (factor: {hparams.factor_sampling})"
        )

        self.data_p = {}
        self.data_p["train"] = [(
            f"{self.features_per_seconds:.1f}fps/video_{i}_{self.features_per_seconds:.1f}fps.pkl"
        ) for i in range(0, 40)]
        self.data_p["val"] = [(
            f"{self.features_per_seconds:.1f}fps/video_{i}_{self.features_per_seconds:.1f}fps.pkl"
        ) for i in range(40, 48)]
        self.data_p["test"] = [(
            f"{self.features_per_seconds:.1f}fps/video_{i}_{self.features_per_seconds:.1f}fps.pkl"
        ) for i in range(48, 56)]

        # Datasplit is equal to Endonet and Multi-Task Recurrent ConvNet with correlation loss for surgical vid analysis
        self.weights = {}
        self.weights["train"] = [
            2.2861910721735503,
            1.6481203007518797,
            0.607202216066482,
            0.8426879901583885,
            0.17998489177915722,
            0.42886210674596964,
            1.558589306029579,
            1.4715359828141783,
            1.927541329581428,
            1.687192118226601,
            0.6410856340664483,
            1.0,
            0.3205990756450009
        ]
        self.weights["train_log"] = [2.2861910721735503, 1.6481203007518797, 0.607202216066482, 0.8426879901583885, 0.17998489177915722, 0.42886210674596964, 1.558589306029579, 1.4715359828141783, 1.927541329581428, 1.687192118226601, 0.6410856340664483, 1.0, 0.3205990756450009]

        self.data = {}
        for split in ["train", "val", "test"]:
            self.data[split] = Cataract1kHelper(hparams,
                                                self.data_p[split],
                                                dataset_split=split)

        print(
            f"train size: {len(self.data['train'])} - val size: {len(self.data['val'])} - test size:"
            f" {len(self.data['test'])}")

    @staticmethod
    def add_dataset_specific_args(parser):  # pragma: no cover
        cataract1k_specific_args = parser.add_argument_group(
            title='cataract1k dataset specific args options')
        cataract1k_specific_args.add_argument("--features_per_seconds",
                                                  default=25,
                                                  type=float)
        cataract1k_specific_args.add_argument("--features_subsampling",
                                                  default=5,
                                                  type=float)

        return parser
