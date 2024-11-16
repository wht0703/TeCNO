import os

import pandas as pd


def label_data(video_path: str = '../../videos', annotation_path: str = '../../annotations'):
    case_list = os.listdir(annotation_path)
    case_list.sort()

    for i in range(0, len(case_list)):
        case_folder = os.path.join(annotation_path, case_list[i])
        video_info_csv = os.path.join(case_folder, case_list[i] + '_video.csv')
        video_annotation_csv = os.path.join(case_folder, case_list[i] + '_annotations_phases.csv')
        video = os.path.join(video_path, case_list[i] + '.mp4')

        video_info = pd.read_csv(video_info_csv)
        video_annotation = pd.read_csv(video_annotation_csv)

        print(f'case: {case_list[i]}')
        print(f'video: {video}')

        last_end_sec = 0
        label_df = pd.DataFrame(columns=['Phase'])
        for i, row in video_annotation.iterrows():
            # Add idle state at the beginning of the operation
            if int(row['sec']) != last_end_sec:
                new_phases = ['idle'] * (int(row['sec']) - last_end_sec - 1) * 5





if __name__ == '__main__':
    label_data()