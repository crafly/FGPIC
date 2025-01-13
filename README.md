# FEW-SHOT OBJECT DETECTION BASED ON PROTOTYPE NETWORK FEATURE ENHANCEMENT

This repo is based on [MMFewShot](https://github.com/open-mmlab/mmfewshot).

Our FGPIC model is a modification based on the FPD model (https://github.com/wangchen1801/FPD).
The modified method is stored in FPD-twice/FGPIC/query_support.py.
To demonstrate that the method in the FGPIC model is more effective than the original method in the FPD model, we have included the original method from the FPD model for verification (FPD-twice/fpd). Additionally, we provide the .pth file obtained by training the FPD model in the current environment.

## Quick Start
```bash
# creat a conda environment
conda create -n FGPIC python=3.8
conda activate FGPIC
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio cudatoolkit=11.3 -c pytorch -c conda-forge

# dependencies
pip install openmim
mim install mmcv-full==1.6.0
mim install mmcls==0.25.0
mim install mmdet==2.24.0
#cd mmdet-2.24.0
#pip install .

pip install -r requirements.txt

# install mmfewshot
pip install git+https://github.com/open-mmlab/mmfewshot.git
# or manually download the code, then
# cd mmfewshot
# pip install .
```

## Prepare Datasets
Please refer to [mmfewshot/data](https://github.com/open-mmlab/mmfewshot/blob/main/tools/data/README.md)
for the data preparation steps.

## Results on VOC Dataset

* Few Shot Fine-tuning

| Config | Split | Shot | Novel AP50 |
|:---:|:---:|:---:|:---:|
|[config](configs/fpd/voc/split1/fpd_r101_c4_2xb4_voc-split1_10shot-fine-tuning.py)|1|10|75.1|
|[config](configs/fpd/voc/split2/fpd_r101_c4_2xb4_voc-split2_10shot-fine-tuning.py)|2|10|48.0|
|[config](configs/fpd/voc/split3/fpd_r101_c4_2xb4_voc-split3_10shot-fine-tuning.py)|3|10|71.5|

## Results on COCO Dataset

* Few Shot Fine-tuning

| Config | Shot | Novel mAP (nAP) |
|:---:|:---:|:---:|
|[config](configs/fpd/coco/fpd_r101_c4_2xb4_coco_30shot-fine-tuning.py)|30|23.1|

## Evaluation

```bash
# single-gpu test
python test.py ${CONFIG} ${CHECKPOINT} --eval mAP|bbox
```


## Training
```bash
# single-gpu training
python train.py ${CONFIG}
```
