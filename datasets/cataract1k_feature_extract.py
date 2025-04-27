import pandas as pd
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from PIL import Image
from albumentations import (
    Compose,
    Resize,
    Normalize,
    ShiftScaleRotate,
)
import torch


class Cataract1kFeatureExtract:
    def __init__(self, hparams):
        self.hparams = hparams
        self.dataset_mode = hparams.dataset_mode
        self.input_height = hparams.input_height
        self.input_width = hparams.input_width
        self.fps_sampling = hparams.fps_sampling
        self.fps_sampling_test = hparams.fps_sampling_test
        self.cataract1k_root_dir = Path(self.hparams.data_root)  # videos splitted in images
        self.image_root = hparams.image_root
        self.transformations = self.__get_transformations()
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
        weights = [
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
        self.class_weights = np.asarray(weights)
        self.label_col = "class"
        self.df = {}
        self.df["all"] = pd.read_pickle(
            self.cataract1k_root_dir / "cataract_split_250px_5fps.pkl")
        assert self.df["all"].isnull().sum().sum(
        ) == 0, "Dataframe contains nan Elements"
        self.df["all"] = self.df["all"].reset_index()

        self.vids_for_training = [i for i in range(0, 40)]
        self.vids_for_val = [i for i in range(40, 48)]
        self.vids_for_test = [i for i in range(48, 56)]

        self.df["train"] = self.df["all"][self.df["all"]["video_idx"].isin(
            self.vids_for_training)]
        self.df["val"] = self.df["all"][self.df["all"]["video_idx"].isin(
            self.vids_for_val)]
        if hparams.test_extract:
            print(
                f"test extract enabled. Test will be used to extract the videos (testset = all)"
            )
            self.vids_for_test = [i for i in range(1, 56)]
            self.df["test"] = self.df["all"]
        else:
            self.df["test"] = self.df["all"][self.df["all"]["video_idx"].isin(
                self.vids_for_test)]

        len_org = {
            "train": len(self.df["train"]),
            "val": len(self.df["val"]),
            "test": len(self.df["test"])
        }

        self.data = {}

        if self.dataset_mode == "img_multilabel":
            for split in ["train", "val"]:
                self.df[split] = self.df[split].reset_index()
                self.data[split] = Dataset_from_Dataframe(
                    self.df[split],
                    self.transformations[split],
                    self.label_col,
                    img_root=f"{self.image_root}/",
                    image_path_col="image_path",
                    add_label_cols=[])
            # here we want to extract all features
            self.df["test"] = self.df["all"].reset_index()
            self.data["test"] = Dataset_from_Dataframe(
                self.df["test"],
                self.transformations["test"],
                self.label_col,
                img_root=f"{self.image_root}/",
                image_path_col="image_path",
                add_label_cols=["video_idx", "image_path"])

    def __get_transformations(self):
        norm_mean = [0.3183, 0.2889, 0.2074]
        norm_std = [0.2991, 0.2494, 0.2164]
        normalize = Normalize(mean=norm_mean, std=norm_std)
        training_augmentation = Compose([
            ShiftScaleRotate(shift_limit=0.1,
                             scale_limit=(-0.2, 0.5),
                             rotate_limit=15,
                             border_mode=0,
                             value=0,
                             p=0.7),
        ])

        data_transformations = {}
        data_transformations["train"] = Compose([
            Resize(height=self.input_height, width=self.input_width),
            training_augmentation,
            normalize,
            ToTensorV2(),
        ])
        data_transformations["val"] = Compose([
            Resize(height=self.input_height, width=self.input_width),
            normalize,
            ToTensorV2(),
        ])
        data_transformations["test"] = data_transformations["val"]
        return data_transformations

    @staticmethod
    def add_dataset_specific_args(parser):  # pragma: no cover
        cataract1k_specific_args = parser.add_argument_group(
            title='cataract1k specific args options')
        cataract1k_specific_args.add_argument("--fps_sampling",
                                            type=float,
                                            default=25)
        cataract1k_specific_args.add_argument("--fps_sampling_test",
                                            type=float,
                                            default=25)
        cataract1k_specific_args.add_argument(
            "--dataset_mode",
            default='video',
            choices=[
                'vid_multilabel', 'img', 'img_multilabel',
                'img_multilabel_feature_extract'
            ])
        cataract1k_specific_args.add_argument("--test_extract",
                                            action="store_true")
        return parser


class Dataset_from_Dataframe(Dataset):
    def __init__(self,
                 df,
                 transform,
                 label_col,
                 img_root="",
                 image_path_col="path",
                 add_label_cols=[]):
        self.df = df
        self.transform = transform
        self.label_col = label_col
        self.image_path_col = image_path_col
        self.img_root = img_root
        self.add_label_cols = add_label_cols

    def __len__(self):
        return len(self.df)

    def load_from_path(self, index):
        img_path_df = self.df.loc[index, self.image_path_col]
        p = f"{self.img_root}/{img_path_df}"
        X = Image.open(p)
        X_array = np.array(X)
        return X_array, p

    def __getitem__(self, index):
        X_array, p = self.load_from_path(index)
        if self.transform:
            X = self.transform(image=X_array)["image"]
        label = torch.tensor(int(self.df[self.label_col][index]))
        add_label = []
        for add_l in self.add_label_cols:
            add_label.append(self.df[add_l][index])
        X = X.type(torch.FloatTensor)
        return X, label, add_label

