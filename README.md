# PATMAT Person Aware Tuning of MAsk Aware Transformer for FAce Inpainting    (ICCV 2023)

<!-- > Generative models such as StyleGAN2 and Stable Diffusion have achieved state-of-the-art performance in computer vision tasks such as image synthesis, inpainting, and de-noising. However, 
current generative models for face inpainting often fail to preserve fine facial details and the identity of the person, despite creating aesthetically convincing image structures and textures.
In this work, we propose Person Aware Tuning (PAT) of Mask-Aware Transformer (MAT) for face inpainting, which addresses this issue. Our proposed method, PATMAT, effectively preserves identity by
incorporating reference images of a subject and fine-tuning a MAT architecture trained on faces. By using ~40 reference images, PATMAT creates anchor points in MAT's style module, and tunes the model
using the fixed anchors to adapt the model to a new face identity. Moreover, PATMAT's use of multiple images per anchor during training allows the model to use fewer reference images than competing methods.
We demonstrate that PATMAT outperforms state-of-the-art models in terms of image quality, the preservation of person-specific details, and the identity of the subject. Our results suggest that PATMAT can be a promising approach for improving the quality of personalized face inpainting. -->

<a href="https://arxiv.org/abs/2304.06107"><img src="https://img.shields.io/badge/arXiv-2008.00951-b31b1b.svg"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>  

<p align="center">
<img src="docs/patmat_teaser"/>  
<br>
Pivotal Tuning Inversion (PTI) enables employing off-the-shelf latent based
semantic editing techniques on real images using StyleGAN. 
PTI excels in identity preserving edits, portrayed through recognizable figures —
Serena Williams and Robert Downey Jr. (top), and in handling faces which
are clearly out-of-domain, e.g., due to heavy makeup (bottom).
</br>
</p>

## Description   
Our twofold steps; PAT and MAT build extensively on Pivot Tuning (PTI) paper + code and MAT paper + code. 
## Getting Started

### Prerequisites
- Linux or macOS
- NVIDIA GPU + CUDA CuDNN (Not mandatory bur recommended)
- Python 3

### Installation
- Dependencies:  
	1. lpips
	2. wandb
	3. pytorch
	4. torchvision
	5. matplotlib
	6. dlib
- All dependencies can be installed using *pip install* and the package name

## Pretrained Models
Please download the pretrained models from the following links.

### Auxiliary Models
various auxiliary models needed for PAT inversion task.  
This includes the StyleGAN generator and pre-trained models used for loss computation.
| Path | Description
| :--- | :----------
|[FFHQ StyleGAN](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl) | StyleGAN2-ada model trained on FFHQ with 1024x1024 output resolution.
|[Dlib alignment](https://drive.google.com/file/d/1HKmjg6iXsWr4aFPuU0gBXPGR83wqMzq7/view?usp=sharing) | Dlib alignment used for images preproccessing.
|[FFHQ e4e encoder](https://drive.google.com/file/d/1ALC5CLA89Ouw40TwvxcwebhzWXM5YSCm/view?usp=sharing) | Pretrained e4e encoder. Used for StyleCLIP editing.
| Glinnt360k can be downloaded from this link: https://drive.google.com/file/d/1pRDYnndOUemVrZaFV6ZGpH3eQowQpQlL/view?usp=sharing |

Note: The StyleGAN model is used directly from the official [stylegan2-ada-pytorch implementation](https://github.com/NVlabs/stylegan2-ada-pytorch).
For StyleCLIP pretrained mappers, please see [StyleCLIP's official routes](https://github.com/orpatashnik/StyleCLIP/blob/main/utils.py)


By default, we assume that all auxiliary models are downloaded and saved to the directory `pretrained_models`. 
However, you may use your own paths by changing the necessary values in `configs/path_configs.py`. 
