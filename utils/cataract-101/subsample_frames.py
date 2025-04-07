import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def subsample_frames(video_dir: str = '../../seg_videos_cataract_101', image_dir: str = '../../images_cataract_101',
                     annotations_dir: str = '../../annotations_cataract_101'):
    case_list = os.listdir(video_dir)
    case_list.sort()

    print(f'Start subsampling videos with 5 fps and create corresponding ground truth labels')

    outer = tqdm(total=len(case_list), position=0, leave=True)

    for i in range(len(case_list)):
        if case_list[i] == '.DS_Store':
            outer.update(1)
            continue
        video_path = Path(os.path.join(video_dir, case_list[i]))
        video_list = sorted(video_path.glob('*.mp4'), key=lambda v: int(v.stem.split('_')[3].split('-')[0]))
        img_out_path = Path(os.path.join(image_dir, case_list[i]))
        outer.set_description(f'Processing: {case_list[i]}', refresh=True)
        os.makedirs(img_out_path, exist_ok=True)
        df = pd.DataFrame(columns=['image_path', 'label'])
        for video in video_list:
            if video.stem.split('_')[3].split('-')[0] == '0':
                continue
            cmd = f'ffmpeg -hide_banner -loglevel panic -i {video} -vf "scale=250:250,fps=5" {img_out_path/video.stem}_%06d.png'
            os.system(cmd)
            img_path = [i.relative_to(image_dir) for i in img_out_path.glob(f'{video.stem}_*.png')]
            img_path = sorted(img_path, key=lambda v: int(v.stem.split('_')[-1]))
            labels = [video.stem.split('_')[2]] * len(img_path)
            new_rows = pd.DataFrame()
            new_rows['image_path'] = img_path
            new_rows['label'] = labels
            df = pd.concat([df, new_rows], ignore_index=True, sort=False)
        df.to_csv(os.path.join(annotations_dir, f"{case_list[i]}/timestamp.csv"), index=False)
        outer.update(1)
    print(f"Done\n")

if __name__ == '__main__':
    subsample_frames()
