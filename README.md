[![arXiv](https://img.shields.io/badge/arXiv-2402.05713-b31b1b.svg)](https://arxiv.org/abs/2402.05713) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Anytime, Anywhere, Anyone: Investigating the Feasibility of Segment Anything Model for Crowd-Sourcing Medical Image Annotations
### Pranav Kulkarni*, Adway Kanhere*, Dharmam Savani*, Andrew Chan, Devina Chatterjee, Paul H. Yi, Vishwa S. Parekh

\* Authors contributed equally to this work.

![concept figure](./assets/fig.png)

Curating annotations for medical image segmentation is a labor-intensive and time-consuming task that requires domain expertise, resulting in "narrowly" focused deep learning (DL) models with limited translational utility. Recently, foundation models like the Segment Anything Model (SAM) have revolutionized semantic segmentation with exceptional zero-shot generalizability across various domains, including medical imaging, and hold a lot of promise for streamlining the annotation process. However, SAM has yet to be evaluated in a crowd-sourced setting to curate annotations for training 3D DL segmentation models. In this work, we explore the potential of SAM for crowd-sourcing "sparse" annotations from non-experts to generate "dense" segmentation masks for training 3D nnU-Net models, a state-of-the-art DL segmentation model. Our results indicate that while SAM-generated annotations exhibit high mean Dice scores compared to ground-truth annotations, nnU-Net models trained on SAM-generated annotations perform significantly worse than nnU-Net models trained on ground-truth annotations ($p<0.001$, all).

Check out our preprint [here](https://arxiv.org/abs/2402.05713)!

# Datasets

## Medical Segmentation Decathlon (MSD)

### Liver Segmentation

### Spleen Segmentation

## Beyond The Cranial Vault (BTCV)
