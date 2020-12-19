
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
*   6: Facades
*   7: Words
*   8: Face swapping

## Pre-trained Models

<!---You can download pretrained models here:--->
Models will be available soon!

- [FFHQ-CelebAMaskHQ-ImageNet](https://www.dropbox.com/s/e2f0dsoxqe693z0/VCNet_FFHQ_200000step_4bs_0.0002lr_1gpu_16run.pth?dl=0) 
- [Places-ImageNet](https://www.dropbox.com/s/f03iblb3epayt6c/VCNet_Places_300000step_4bs_0.0002lr_1gpu_17run.pth?dl=0)

## Qualitative Results

Some intermediate results of VCNet on FFHQ-CelebAMaskHQ-ImageNet model 
(from left to right: the original - the contaminant - broken - gt mask - smoothed mask - predicted mask - output):

Iteration 75k:
![][res75k1]
![][res75k2]
![][res75k3]

Iteration 90k:
![][res90k1]
![][res90k2]
![][res90k3] 

Iteration 105k:
![][res105k1]
![][res105k2]
![][res105k3] 

## Quantitative Results

#### FFHQ

| Models               |       BCE       |      PSNR      |      SSIM      |
| -------------------  |---------------- | -------------- | -------------- |
| Contextual Attention |     1.297       |      16.56     |      0.5509    |
| GMC                  |     0.766       |      20.06     |      0.6675    |
| Partial Conv.        |     0.400       |      20.19     |      0.6795    |
| Gated Conv.          |     0.660       |      17.16     |      0.5915    |
| VCN (**original**)   |     0.400       |      20.94     |      0.6999    |
| VCN (**ours**)       |     0.438       |      22.29     |      0.6658    |

#### Places2

| Models               |       BCE       |      PSNR      |      SSIM      |
| -------------------  |---------------- | -------------- | -------------- |
| Contextual Attention |     0.574       |      18.12     |      0.6018    |
| GMC                  |     0.312       |      20.38     |      0.6956    |
| Partial Conv.        |     0.273       |      19.73     |      0.6682    |
| Gated Conv.          |     0.504       |      18.42     |      0.6423    |
| VCN (**original**)   |     0.253       |      20.54     |      0.6988    |
| VCN (**ours**)       |     0.439       |      20.29     |      0.6870    |

## Contacts

Please feel free to open an issue or to send an e-mail to ```furkan.kinli@ozyegin.edu.tr```

[res75k1]: outputs/examples_74744.png
[res75k2]: outputs/examples_75144.png
[res75k3]: outputs/examples_75544.png
[res90k1]: outputs/examples_90944.png
[res90k2]: outputs/examples_91144.png
[res90k3]: outputs/examples_91344.png
[res105k1]: outputs/examples_104344.png
[res105k2]: outputs/examples_104744.png
[res105k3]: outputs/examples_106144.png