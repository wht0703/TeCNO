import os

import cv2
import pandas as pd
from tqdm import tqdm


def cal_frames(video_path: str = '../../videos_cataract_101', annotation_path: str = '../../annotations_cataract_101'):
    print(f'Start calculating frames')
    df_video_info = pd.read_csv(os.path.join(annotation_path, 'videos.csv'), sep=';')
    outer = tqdm(total=len(df_video_info), desc='Files', position=0, leave=True)

    for idx, row in df_video_info.iterrows():
        video = os.path.join(video_path, f'case_{row["VideoID"]}.mp4')

        outer.set_description(f'Procerssing case: {os.path.abspath(video)}', refresh=True)
        cap = cv2.VideoCapture(video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        if row['Frames'] == total_frames and row['FPS'] == fps:
            outer.update(1)
            continue
        else:
            row['Frames'] = total_frames
            row['FPS'] = fps
        outer.update(1)

if __name__ == '__main__':
    cal_frames()