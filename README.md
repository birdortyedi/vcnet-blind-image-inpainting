
# [Re] - VCNet: A Robust Approach to Blind Image Inpainting

This repository is the re-production implementation of [VCNet: A Robust Approach to Blind Image Inpainting](https://arxiv.org/pdf/2003.06816.pdf) by [Yi Wang](https://shepnerd.github.io/), [Ying-Cong Chen](https://yingcong.github.io/), Xin Tao and [Jiaya Jia](http://jiaya.me/) in the scope of [ML Reproducibility Challenge 2020](https://paperswithcode.com/rc2020). 

<!---
Authored by [Furkan Kınlı](https://birdortyedi.github.io/), Barış Özcan, [Furkan Kıraç](http://fkirac.net/).
--->

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

To download datasets, please visit the corresponding web pages:
[FFHQ](https://github.com/NVlabs/ffhq-dataset) - [CelebAMaskHQ](https://github.com/switchablenorms/CelebAMask-HQ) - 
[Places](http://places2.csail.mit.edu/download.html) - [ImageNet](http://image-net.org/download) - 
[graffiti](https://github.com/pavelkraleu/graffiti-dataset) - [raindrop](https://github.com/rui1996/DeRaindrop). 

After downloading, extract each of them to corresponding folder that you created as:
```
mkdir datasets
cd datasets
mkdir <dataset_name>
```


## Training

To train VCNet from the scratch in the paper, run this command:

```train
python main.py --base_cfg config.yml
```

## Evaluation

To evaluate VCNet on FFHQ with a particular test mode, run:

```eval
python main.py --base_cfg config.yml -t -w vcnet.pth --dataset "FFHQ" --test_mode 1
```

Modes:
*   1: Contaminant image
*   2: Random brush strokes with noise
*   3: Random brush strokes with colors
*   4: Real occlusions
*   5: Graffiti
*   6: Facades (*problematic*)
*   7: Words (*problematic*)
*   8: Face swapping (*problematic*)

## Pre-trained Models

You can download pretrained models here:

- [FFHQ-CelebAMaskHQ-ImageNet](https://www.dropbox.com/s/e2f0dsoxqe693z0/VCNet_FFHQ_200000step_4bs_0.0002lr_1gpu_16run.pth?dl=0)
  
- [Places-ImageNet](https://www.dropbox.com/s/f03iblb3epayt6c/VCNet_Places_300000step_4bs_0.0002lr_1gpu_17run.pth?dl=0)

## Qualitative Results

From FFHQ-CelebAMaskHQ model 
(from left to right: the original - the contaminant - broken - gt mask - smoothed mask - predicted mask - output):

![][ffhq-celeba-1]
![][ffhq-celeba-2]
![][ffhq-celeba-3]
![][ffhq-celeba-4]

From FFHQ-ImageNet model 
(from left to right: the original - the contaminant - broken - gt mask - smoothed mask - predicted mask - output):

![][ffhq-in-1]
![][ffhq-in-2]
![][ffhq-in-3]
![][ffhq-in-4]

From Places-ImageNet model 
(from left to right: the original - the contaminant - broken - gt mask - smoothed mask - predicted mask - output):

![][places-1]
![][places-2]
![][places-3]
![][places-4]


## Quantitative Results

#### FFHQ

| Models               |       BCE       |      PSNR      |      SSIM      |
| -------------------  |---------------- | -------------- | -------------- |
| Contextual Attention |     1.297       |      16.56     |      0.5509    |
| GMC                  |     0.766       |      20.06     |      0.6675    |
| Partial Conv.        |     0.400       |      20.19     |      0.6795    |
| Gated Conv.          |     0.660       |      17.16     |      0.5915    |
| VCN (**original**)   |     0.400       |      20.94     |      0.6999    |
| VCN (**ours**)       |     0.439       |      24.76     |      0.7026    |

#### Places2

| Models               |       BCE       |      PSNR      |      SSIM      |
| -------------------  |---------------- | -------------- | -------------- |
| Contextual Attention |     0.574       |      18.12     |      0.6018    |
| GMC                  |     0.312       |      20.38     |      0.6956    |
| Partial Conv.        |     0.273       |      19.73     |      0.6682    |
| Gated Conv.          |     0.504       |      18.42     |      0.6423    |
| VCN (**original**)   |     0.253       |      20.54     |      0.6988    |
| VCN (**ours**)       |     0.437       |      21.53     |      0.7070    |

## Contacts

<!--Please feel free to open an issue or to send an e-mail to ```furkan.kinli@ozyegin.edu.tr```-->

[ffhq-celeba-1]: outputs/ffhq-celeba/examples_175544.png
[ffhq-celeba-2]: outputs/ffhq-celeba/examples_176544.png
[ffhq-celeba-3]: outputs/ffhq-celeba/examples_178744.png
[ffhq-celeba-4]: outputs/ffhq-celeba/examples_178944.png
[ffhq-in-1]: outputs/ffhq-imagenet/examples_153144.png
[ffhq-in-2]: outputs/ffhq-imagenet/examples_173544.png
[ffhq-in-3]: outputs/ffhq-imagenet/examples_177944.png
[ffhq-in-4]: outputs/ffhq-imagenet/examples_181344.png
[places-1]: outputs/places/examples_85199.png
[places-2]: outputs/places/examples_85599.png
[places-3]: outputs/places/examples_87999.png
[places-4]: outputs/places/examples_91399.png