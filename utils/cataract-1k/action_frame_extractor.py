import os

import pandas as pd
from tqdm import tqdm


def csv_read_annotations(csv_file_path: str):
    action_frames = []
    action_labels = []
    df = pd.read_csv(csv_file_path)
    for i, row in df.iterrows():
        action_labels.append(row['comment'])
        action_frames.append([row['frame'], row['endFrame']])
    return action_frames, action_labels

def csv_read_total_frames(csv_file_path: str):
    df = pd.read_csv(csv_file_path)
    return df['total_frames'][0]

def convert_frames_to_videos(video_path: str, frame_start: int, frame_end: int, case_folder: str, name: str):
    cmd = f"ffmpeg -hide_banner -loglevel panic -i {video_path} -vf \"select=\'between(n, {frame_start}, {frame_end})\',setpts=PTS-STARTPTS\" {case_folder}/{name}"
    os.system(cmd)

def extract_action_frames(video_path: str = '../../videos', annotation_path: str = '../../annotations', seg_video_path: str = '../../seg_videos',):
    case_list = os.listdir(annotation_path)
    case_list.sort()

    print(f'Start extracting action frames')
    outer  = tqdm(total=3, desc='Extracting action frames from: ', position=0, leave=True)

    for i in range(0, 3):
        case_folder = os.path.join(annotation_path, case_list[i])
        video_info_csv = os.path.join(case_folder, f"{case_list[i]}_video.csv")
        video_annotation_csv = os.path.join(case_folder, f"{case_list[i]}_annotations_phases.csv")
        video = os.path.join(video_path, f"{case_list[i]}.mp4")
        seg_video = os.path.join(seg_video_path, case_list[i])

        os.makedirs(seg_video, exist_ok=True)

        outer.set_description(f"Processing case {case_list[i]}.mp4", refresh=True)

        annotations, labels = csv_read_annotations(video_annotation_csv)
        total_frames = csv_read_total_frames(video_info_csv)

        last_frame = 0
        for j in range(len(labels)):
            labels[j] = labels[j].replace(' ', '')
            labels[j] = labels[j].replace('/', '')
            labels[j] = labels[j].replace('_', '')
            if last_frame != annotations[j][0]:
                frame_start = last_frame
                frame_end = annotations[j][0] - 1
                video_name = f"{seg_video}/{case_list[i]}_Idle_{frame_start}-{frame_end}.mp4"
                convert_frames_to_videos(video, frame_start, frame_end, case_folder, video_name)
            frame_start = annotations[j][0]
            frame_end = annotations[j][1]
            last_frame = frame_end + 1
            video_name = f"{seg_video}/{case_list[i]}_{labels[j]}_{frame_start}-{frame_end}.mp4"
            convert_frames_to_videos(video, frame_start, frame_end, case_folder, video_name)
            if j == len(labels) - 1 and last_frame <= total_frames - 1:
                frame_end = total_frames - 1
                video_name = f"{seg_video}/{case_list[i]}_Idle_{last_frame}-{frame_end}.mp4"
                convert_frames_to_videos(video, last_frame, frame_end, case_folder, video_name)
        outer.update(1)

if __name__ == '__main__':
    extract_action_frames()