import os

import cv2
import pandas as pd
from tqdm import tqdm


def cal_frames(video_path: str = '../../videos', annotation_path: str = '../../annotations'):
    case_list = os.listdir(annotation_path)
    case_list.sort()

    print(f"Start calculating frames")

    outer = tqdm(total=len(case_list), desc='Files', position=0, leave=True)

    for i in range(0, len(case_list)):
        case_folder = os.path.join(annotation_path, case_list[i])
        video_info_csv = os.path.join(case_folder, case_list[i] + '_video.csv')
        video = os.path.join(video_path, case_list[i] + '.mp4')

        outer.set_description(f"Processing case {os.path.abspath(video)}", refresh=True)
        cap = cv2.VideoCapture(video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        video_info = pd.read_csv(video_info_csv)
        video_info['total_frames'] = total_frames
        video_info.to_csv(video_info_csv, index=False)
        outer.update(1)

if __name__ == '__main__':
    cal_frames()