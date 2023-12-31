# Distribution-Based-frame-supervised-Language-driven-Action-Localization

This is the implementation for the paper "Probability Distribution Based Frame-supervised Language-driven Action Localization" (ACM MM2023). [Arxiv Preprint](https://arxiv.org/)

This repository is based on the repository of the paper ["Video Moment Retrieval from Text Queries via Single Frame Annotation"](https://github.com/r-cui/ViGA).

## Prerequisites
- pytorch=1.10.0
- python=3.7
- numpy
- scipy
- pyyaml
- tqdm

You can also run the following commands to prepare the conda environmnet.
```
# preparing environment
bash conda.sh
conda activate DBFS
```

## Preparation

#### Annotations
The frame-annotations we used are available in the `data/charadessta/annotations` and `data/tacos/annotations` folder.

#### Features 
We use I3D features for charadessta and C3D features for tacos. I3D features for charadessta can be downloaded from [link](https://github.com/JonghwanMun/LGI4temporalgrounding). C3D features for tacos can be downloaded from [link](https://github.com/microsoft/2D-TAN) and be extracted as individual files. Then save them to the `data/charadessta/features` and `data/tacos/features` folder seperately. 

Please also download [glove](https://nlp.stanford.edu/data/glove.840B.300d.zip) to the `data/glove` folder.

#### Model
Our trained model are provided in [link](https://drive.google.com/drive/folders/10UIZM2OWx5UzbLjHZWmGi6cZphgj1rU8?usp=sharing). Please download them to the `ckpt/` folder.

## Quick Start
Run the following commands for evaluation: 
```
# Evaluate charades
python -m src.experiment.eval --exp ckpt/charades

# Evaluate tacos
python -m src.experiment.eval --exp ckpt/tacos

```

## Citation
Shuo Yang, Zirui Shang, and Xinxiao Wu. 2023. Probability Distribution Based Frame-supervised Language-driven Action Localization. In Proceedings of the 31st ACM International Conference on Multimedia (MM ’23), October 29–November 3, 2023, Ottawa, ON, Canada. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3581783.3612512
