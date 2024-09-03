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
```


