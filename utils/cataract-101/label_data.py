import os

import pandas as pd
from tqdm import tqdm

stage_map = {
    1: 'Incision',
    2: 'ViscousAgentInjection',
    3: 'Rhexis',
    4: 'Hydrodissection',
    5: 'Phacoemulsificiation',
    6: 'IrrigationAndAspiration',
    7: 'CapsulePolishing',
    8: 'LensImplantSettingUp',
    9: 'ViscousAgentRemoval',
    10: 'TonifyingAndAntibiotics'
}

def label_date(annotation_dir: str = '../../annotations_cataract_101'):
    print(f'Start annotating cataract-101 dataset')
    df_annotation = pd.read_csv(os.path.join(annotation_dir, 'annotations.csv'), sep=';')
    df_video_info = pd.read_csv(os.path.join(annotation_dir, 'videos.csv'), sep=';')

    outer = tqdm(total=len(df_video_info), position=0, leave=True)

    rows = []
    for idx, row in df_annotation.iterrows():
        outer.set_description(f'Annotating case {row["VideoID"]}', refresh=True)

        new_row = {'caseId': row['VideoID'], 'comment': stage_map[row['Phase']], 'frame': row['FrameNo']}
        if idx != len(df_annotation) - 1 and df_annotation.iloc[idx + 1]['VideoID'] == row['VideoID']:
            current_stage_end_frame = df_annotation.iloc[idx + 1]['FrameNo'] - 1
            new_row['endFrame'] = current_stage_end_frame
            rows.append(new_row)
        else:
            current_stage_end_frame = df_video_info[df_video_info['VideoID'] == row['VideoID']]['Frames'].item()
            new_row['endFrame'] = current_stage_end_frame - 1
            rows.append(new_row)
            case_annotation_path = os.path.join(annotation_dir, f'case_{row["VideoID"]}')
            os.makedirs(case_annotation_path, exist_ok=True)
            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(case_annotation_path, f'case_{row["VideoID"]}_annotations_phases.csv'), index=False)
            rows = []
            outer.update(1)

if __name__ == '__main__':
    label_date()