# ICCV2019_MirrorNet

## Where Is My Mirror? (ICCV2019)
Xin Yang\*, [Haiyang Mei](https://mhaiyang.github.io/)\*, Ke Xu, Xiaopeng Wei, Baocai Yin, [Rynson W.H. Lau](http://www.cs.cityu.edu.hk/~rynson/)<sup>&dagger;</sup> (* Joint first authors, <sup>&dagger;</sup> Rynson Lau is the corresponding author)

[[Project Page](https://mhaiyang.github.io/ICCV2019_MirrorNet/index.html)][[Arxiv](https://arxiv.org/pdf/1908.09101.pdf)]

### Abstract
Mirrors are everywhere in our daily lives. Existing computer vision systems do not consider mirrors, and hence may get confused by the reflected content inside a mirror, resulting in a severe performance degradation. However, separating the real content outside a mirror from the reflected content inside it is non-trivial. The key challenge lies in that mirrors typically reflect contents similar to their surroundings, making it very difficult to differentiate the two. In this paper, we present a novel method to accurately segment mirrors from an input image. To the best of our knowledge, this is the first work to address the mirror segmentation problem with a computational approach. We make the following contributions. First, we construct a large-scale mirror dataset that contains mirror images with the corresponding manually annotated masks. This dataset covers a variety of daily life scenes, and will be made publicly available for future research. Second, we propose a novel network, called MirrorNet, for mirror segmentation, by modeling both semantical and low-level color/texture discontinuities between the contents inside and outside of the mirrors. Third, we conduct extensive experiments to evaluate the proposed method, and show that it outperforms the carefully chosen baselines from the state-of-the-art detection and segmentation methods.

### Citation
If you use this code or our dataset (including test set), please cite:

```
@InProceedings{Yang_2019_ICCV, 
    author = {Yang, Xin and Mei, Haiyang and Xu, Ke and Wei, Xiaopeng and Yin, Baocai and Rynson, W.H. Lau}, 
    title = {Where Is My Mirror?}, 
    booktitle = {IEEE International Conference on Computer Vision (ICCV)}, 
    year = {2019}
}
```

### Dataset
See [Peoject Page](https://mhaiyang.github.io/ICCV2019_MirrorNet/index.html)

### Requirements
* PyTorch == 0.4.1
* TorchVision == 0.2.1
* CUDA 9.0  cudnn 7
* Setup
```
sudo pip3 install -r requirements.txt
git clone https://github.com/Mhaiyang/dss_crf.git
sudo python setup.py install
```

### Train & Test
##### Train:
Coming Soon！

##### Test:
Coming Soon！

### Main Results

##### Quantitative Results

| method | CRF | IOU | Acc | F | MAE | BER |  
|----------------|------|------|------|-----------------|-----------------| ----------  
| Statistics | - | 30.76 | 0.595 | 0.436 | 0.360 | 32.94 |
| PSPNet | - | 63.18 | 0.750 | 0.746 | 0.117 | 15.82 |            |
| ICNet | - |
| Mask RCNN | - |
| DSS | - |
| PiCANet | - |
| RAS | - |
| R<sup>3</sup>Net w/o C | - |
| R<sup>3</sup>Net | - |
| DSC | - |
| BDRAR w/o C | - |
| BDRAR | - |
| MirrorNet w/o C | - |
| MirrorNet | - |

##### Qualitative Results

### License
Please see `license.txt` 

### Contact
E-Mail: mhy666@mail.dlut.edu.cn
