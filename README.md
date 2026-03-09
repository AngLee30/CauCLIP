<h1 align="center">CauCLIP</h1>

<p align="center">
<b>CauCLIP: Bridging the Sim-to-Real Gap in Surgical Video Understanding
Via Causality-Inspired Vision-Language Modeling</b>
</p>

<p align="center">
<b>ICASSP 2026</b>
</p>

<p align="center">
Yuxin He</a>,
An Li</a>,
Cheng Xue*</a>
</p>

<p align="center">
Southeast University
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.06619">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b" />
  </a>
</p>

## 1. Introduction

This repository provides a standardized implementation of our proposed framework, **CauCLIP**, for **surgical phase recognition** under domain shift.

## 2. Data Preparation

Extract frames from each video in the [**SurgVisDom** (MICCAI 2020 sub-challenge)](https://surgvisdom.grand-challenge.org/) dataset:

**For each video in the training set:**

* Create a folder named after the video (without the file extension).
* Extract all frames from the video and place them inside the corresponding folder.
* Name each frame using the format:
```
frame_{:05d}.png
```
* Frame indexing starts from `00001`.
* Each video maintains its own independent frame numbering.

**Example**

```
train_root_dir/
├── ND_0001/
│   ├── frame_00001.png
│   ├── frame_00002.png
│   ├── ...
├── ND_0002/
│   ├── frame_00001.png
│   ├── frame_00002.png
│   ├── ...
├── ...
```

**For each video in the validation set:**

* Split the frames into groups of 128 frames.
* Each group should be stored in a separate folder.
* The folder name should follow the format:
```
{video_name}_{group_id}
```
where `group_id` starts from `000` and increases sequentially.

Inside each group folder:

* Frames should also be named using the format:
```
frame_{:05d}.png
```

**Example**

```
val_root_dir/
├── test_video_0000_000/
│   ├── frame_00001.png
│   ├── ...
│   ├── frame_00128.png
├── test_video_0000_001/
│   ├── frame_00001.png
│   ├── ...
│   ├── frame_00128.png
├── ...
```

## 3. Configuration

Edit the configuration file:

```
./configs/train.yaml
```

Set the following fields:

* `train_root_dir`: root directory containing the **training frames**
* `val_root_dir`: root directory containing the **validation frames**

Other hyperparameters in the configuration file can also be modified if needed.

## 4. Install Dependencies

Install **PyTorch** (recommended version):

```
PyTorch ≈ 1.11.0 + cu113
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

## 5. Training

Run the training script:

```bash
python train.py
```

