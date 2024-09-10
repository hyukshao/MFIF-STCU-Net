# Multi-focus image fusion based on transformer and depth information learning
Code for our papaer Multi-focus image fusion based on transformer and depth information learning

Dataset
----
We selected image pairs from the publicly available image dataset of ADEChallengeData2016 for preprocessing. Firstly, we cropped them into 224 * 224 pixel image blocks and removed image blocks with single information, leaving behind image blocks with more information. As a result, we obtained 37422 image blocks with rich and clear information.

Training
----
run main.py

Notice: We run the training process on an Nvidia 3090Ti GPUs, if you want to use less GPU, the batch size should be changed.

And the training are conducted with generated dataset (2 source images and 1 guidance map). 

Testing
----
run run_inference.py

Currently, the code can be directly used for the tseting on Lytro dataset. Please change the directory if you want to run the test on your own image pairs. Two source images should be named xxxA.jpg/png/bmp and xxxB.jpg/png/bmp in pair and put into the same directory.

A single GPU is enough to carry out it.

## Pre-trained models & Results
----
You are very welcome to conduct results comparison with our method. The fusion results on Lytro dataset are shown in Data/result. The Pre-trained model is shown in pretrianed_Model/model_best.pth.tar.


## Citing Multi-focus image fusion based on transformer and depth information learning
If you find this work useful for your research, please cite our [paper]():
@article{SHAO2024109629,
title = {Multi-focus image fusion based on transformer and depth information learning},
journal = {Computers and Electrical Engineering},
volume = {119},
pages = {109629},
year = {2024},
issn = {0045-7906},
doi = {https://doi.org/10.1016/j.compeleceng.2024.109629},
url = {https://www.sciencedirect.com/science/article/pii/S0045790624005561},
author = {Xinfeng Shao and Xin Jin and Qian Jiang and Shengfa Miao and Puming Wang and Xing Chu},
keywords = {Multi-focus image fusion, Transformer, U-net, Deep learning, Depth information},
abstract = {According to the imaging principle of the camera, the focusing and defocusing parts of the image are often determined by the depth information in the real scene. Only objects within a certain depth can present a clear appearance in the captured image, while objects outside the depth often become blurry. Thus, a single camera cannot clearly present enough visual information for automatic driving systems, but multi-sensor image fusion can produce comprehensive information for vehicles to improve its ability of traffic environment perception. In this work, we first use the depth estimation model and the α-matte model to create a simulated multi-focus image dataset based on the focus characteristics. Second, we combine Transformer and convolution neural networks to respectively extract global and local information in image processing tasks. Thus, a novel deep learning network architecture for multi-focus image fusion is proposed in this work. Our network named STCU-Net is based on U-Net and designed with Transformer and convolution neural networks. Qualitative and quantitative evaluations have confirmed the superiority of the proposed method compared to state-of-the-art methods. The code is available at https://github.com/hyukshao/MFIF-STCU-Net.}
}
```


