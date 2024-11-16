import os

import pandas as pd
from pathlib import Path

from tqdm import tqdm


def write_pkl(root_dir):
    root_dir = Path(root_dir)
    img_base_path = root_dir / "images"
    annot_timephase_path = root_dir / "annotations"
    out_path = root_dir / "dataframes"

    case_list = os.listdir(img_base_path)
    case_list = sorted(case_list, key=lambda x: int(x.split("_")[1]))

    class_labels = [
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

    print(f"Start to create dataframes pickle\n")
    outer = tqdm(total=len(case_list), desc="Creating dataframes from: ", position=0, leave=True)

    cataract_df = pd.DataFrame(columns=[
        "image_path", "class", "time", "video_idx"
    ])

    for i in range(len(case_list)):
        outer.set_description(f"Processing: {case_list[i]}", refresh=True)
        vid_df = pd.DataFrame()
        timestamp_path_for_vid = annot_timephase_path / f"{case_list[i]}/timestamp.csv"
        timestamp_df = pd.read_csv(timestamp_path_for_vid)
        timestamp_df["frame"] = [i for i in range(len(timestamp_df["image_path"]))]
        vid_df["video_idx"] = [i] * len(timestamp_df["image_path"])
        # add image class
        for j, p in enumerate(class_labels):
            timestamp_df["label"] = timestamp_df.label.replace({p: str(j)})

        vid_df = pd.concat([vid_df, timestamp_df], axis=1)
        vid_df = vid_df.rename(columns={
            "label": "class",
            "frame": "time",
        })
        cataract_df = pd.concat([cataract_df, vid_df], ignore_index=True, sort=False)
        outer.update(1)
    print(cataract_df.shape)
    print(cataract_df.columns)
    os.makedirs(out_path, exist_ok=True)
    cataract_df.to_pickle(out_path / "cataract_split_250px_5fps.pkl")
    cataract_df.to_csv(out_path / "cataract_split_250px_5fps.csv", index=False)


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    root_dir = "../../"
    write_pkl(root_dir)
