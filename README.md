# Table of Contents

<img src="assets/tecno_logo.png"
alt="logo tecno"
width=150px
style="margin-right:50px"
align="right" />

1. [**About**](#about)
2. [**Project Structure**](#project-structure)
3. [**Getting started**](#getting-started)
    1. [**Prerequisites**](#prerequisites)
    2. [**Usage**](#usage)

# About

This Project is an adapted version of the original project TeCNO ([**TeCNO**](https://github.com/tobiascz/TeCNO)),
applying
the model to surgical stage recognition in cataract surgeries.

TeCNO performs hierarchical prediction refinement with causal, dilated convolutions for surgical phase recognition and
outperforms various state-of-the-art LSTM approaches!

Link to paper: [**TeCNO Paper**](https://arxiv.org/abs/2003.10751)

<p align="center">
     <img src="assets/abstract_tecno.png"
          alt="logo tecno"
          width=1000px />
</p>

# Project Structure

- `model/`contains implementations of CNN-based feature extractors and the multi-stage temporal convolutional network (
  MS-TCN).


- `datasets/` provides dataset classes and loading utilities.


- `modules/` implements training, validation, and testing pipelines for both feature extractors and the MS-TCN.


- `utils/` utility scripts for data conversion and preprocessing, including routines to transform given dataset formats
  into the format required for model training.

# Getting started

Follow these steps to get the code running on your local machine!

## Prerequisites

The implementation has been tested with Python 3.10.

To install all required packages, run the following command, or install the dependencies listed in `requirements.txt`
manually.

```
pip install -r requirements.txt
```

## Usage

We are using the publicly available [Cataract1k](https://github.com/Negin-Ghamsarian/Cataract-1K)
and [Cataract-101](https://ftp.itec.aau.at/datasets/ovid/cat-101/).
Since the original project is based on laparoscopic surgery videos dataset `Cholec80`, we first need to convert these
datasets into the model’s required format.
To proceed, please ensure that your directory structure matches the following layout.

For `Cataract1k`:

```
./project_root
├── annotations/   # contains raw annotation data
│   ├── case_4687
│   ├── case_4693
│   ...
│   └── case_5357
│   ...
└── videos/       # contains raw surgey videos
```

For `Cataract-101`:

```
./project_root
├── annotations_cataract_101/   # contains raw annotation data
│   ├── annotations.csv
│   └── videos.csv
│   ...
└── videos_cataract_101/       # contains raw surgey videos
```

### Stage 1 - Separate Video into Multiple Phase Segments

Both cataract‐surgery video datasets provide frame‐wise phase annotations. To preserve these annotations after sampling,
we first split each video into multiple phase segments.

For `Catract1k`:

**Note:** We assume that all intervals between two consecutive phase annotations are idle phases, which are not included
in the original dataset annotations.

```
cd ./utils/cataract-1k
python cal_frames.py
python action_frame_extractor.py
```

For `Cataract-101`:

```
cd ./utils/cataract-101
python label_data.py
python action_frame_extrator.py
```

After executing the instructions above, you should see the newly created folders `seg_videos` and
`seg_videos_cataract_101`, respectively.

### Stage 2 - Subsample Video and Generate Final Model Input

The raw Cataract‑1K and Cataract‑101 videos run at 60 fps and 25 fps, respectively. We then subsample them to 5 fps and
extract each frame as an image.

For `Cataract1k`:

```
cd ./utils/cataract-1k
python subsample_frames.py
python create_dataframes.py
```

For `Cataract-101`:

```
cd ./utils/cataract-101
python subsample_frames.py
python create_dataframes.py
```

After executing the instructions above, you should see the following newly created directories

```
./project_root
├── images/        # contains video frames (images_cataract_101 for Cataract-101 dataset)
│   ├── case_4678
│   ├── case_4693
│   ...
│   └── case_5357
│   ...
└── dataframes/   # contains data loading input files (dataframes_cataract-101 for Cataract-101 dataset) 
    ├── cataract_split_250px_5fps.csv # for debugging
    └── cataract_split_250px_5fps.pkl # actual input for data loading
```

### Adapting New Dataset to model

To successfully apply this model on new dataset, you should write your own script to generate final data frames served
as actual
input for data loading. The final DataFrame (e.g. saved as a CSV/Pickle File) should follow this format:

| image_path         | class                    | time                | video_idx   |
|--------------------|--------------------------|---------------------|-------------|
| path to the images | ground truth class index | video time in frame | video index |

After that, define new dataset classes by following the structures in
`datasets/cataract101.py` and `datasets/cataract101_feature_extract.py`.
To generate the weights for the loss function, use `utils/tecno/cal_median_frequency_weights.py`.

### Stage 3 - Training

We fine‑tune the CNN‑based feature extractor and then train the MS‑TCN using the extracted features. For training, we
split the dataset into training, validation, and test sets with a ratio of 7:1.5:1.5, respectively.
This is defined in the dataset classes and can be easily adjusted by modifying the corresponding lines.

#### Train feature extractor

Ensure that `train.py` only calls `trainer.fit(model)` and specifies `mode='max'` in both callback definitions, since we
use validation accuracy for early stopping and checkpointing.

For `Cataract1k`:

```
python train.py -c modules/cnn/config/config_feature_extract.yml
```

For `Cataract-101`:

```
python train.py -c modules/cnn/config/config_feature_extract_cataract_101.yaml
```

This will train your feature extractor and, in the **test** step, extract features for every frame in each video and
save them as `.pkl` files. To enable this, set `trainer.test` to use the appropriate model checkpoint, and set
`test-extract: True` in the configuration file.

#### Train MS-TCN

Ensure that `train.py` only calls `trainer.fit(model)`, and that both early‑callback definitions use `mode='min'`, since
we use validation loss as the criterion for early stopping and checkpointing. Adapt the `data_root` in configuration
file.

For `Cataract1k`:

```
python train.py -c modules/mstcn/config/config_tcn.yml
```

For `Cataract-101`:

```
python train.py -c modules/mstcn/config/config_tcn_cataract_101.yaml
```

### Model Performance Evaluation

All pretrained model check points can be found [**model checkpoints**](https://tumde-my.sharepoint.com/:u:/g/personal/haotong_wang_tum_de/EXcmL8WDS9VFm5fuJx67O0MB-1Sq8zakUjagYA2uBi3ZNw?e=Adf9xc)

#### Evaluate Feature Extractor

During the **test** step, video‑level performance metrics for each test video are computed in parallel with feature
extraction and saved as `.txt` files.

`test_cnn.ipynb`: These files are used to calculate the averages and standard deviations of both video‑level and
stage‑level performance metrics, and to visualize the stage‑level metrics in box plots.

`fearture_extractor_cam.ipynb`: generates CAM based on an single input video frame image.

#### Evaluate MS-TCN

`test_tcn.ipynb`: computes the averages and standard deviations of both video‑level and stage‑level performance
metrics, and visualizes the stage‑level metrics in box plots. Additionally, it computes and visualizes the confusion
matrix for each test video, and provides functionality to display the final inference results.

**Note**:

- To use the provided notebooks, please adjust the checkpoint path, data path, model hyperparameter class
  inputs and test set indices
  accordingly.
- Several evaluation results could be found in `./evalutation`. For Cataract-1k, the reported results are based on a
  feature extractor fine-tuned for 7 epochs and an MS-TCN trained for 24 epochs. For Cataract-101, the reported results
  are based on a feature extractor fine-tuned for 5 epochs and an MS-TCN trained for 23 epochs. To achieve the similar training outcome,
  you may need to adjust checkpointing or early stopping callback in `train.py` accordingly.

## Known Issues

- All performance results presented are based on our current workflow, in which we first split videos into distinct
  stage segments and then subsample them. However, in real‑world scenarios—where phase boundaries must be determined a
  posteriori—the model trained in this way will not work. This issue requires further investigation. One possible
  solution is to perform the subsampling in feature space. Please adjust the code in `datasets/cholec80.py` and
  `datasets/cataract101.py` accordingly, and use the `features_per_second` and `features_subsampling` parameters to
  control the subsampling.


- The default metrics calculation for MS-TCN in `utils/metric-helper` seems to be flawed. A quick work‑around is to
  assign the output class number manually:
  ```
  preds_stage = to_onehot(preds[s].argmax(dim=1), num_classes=num_output_stages) # num_output_stages should be an integer value 
  target_onehot = to_onehot(target, num_classes=num_output_stages)
  ```
  This will stop the code from throwing errors, but to correctly analyze model performance, please use the notebooks
  mentioned in the previous section.