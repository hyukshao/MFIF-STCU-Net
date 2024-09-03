import argparse
import os
from path import Path
import torch
import torch.backends.cudnn as cudnn
# import torch.nn.functional as F
import torchvision.transforms.functional as F
import models
from tqdm import tqdm
import torchvision.transforms as transforms
import fusion_transforms
# from scipy.ndimage import imread
from imageio import imread
import numpy as np
from matplotlib import pyplot
from PIL import Image
from torchstat import stat
from fvcore.nn import FlopCountAnalysis, parameter_count_table

os.environ['CUDA_VISIBLE_DEVICES']='0'

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))
parser = argparse.ArgumentParser(description='PyTorch CBAfusionNet inference on a folder of img pairs',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', metavar='DIR', default='Data/lytro_224',
                    help='path to images folder, image names must match \'[name]A.[ext]\' and \'[name]B.[ext]\'')
# parser.add_argument('--data', metavar='DIR', default='Data/lytro',
#                     help='path to images folder, image names must match \'[name]A.[ext]\' and \'[name]B.[ext]\'')
parser.add_argument('--pretrained', metavar='PTH', default='model_best.pth.tar',
                    help='path to pre-trained model')
parser.add_argument('--output', metavar='DIR', default="result",
                    help='path to output folder. If not set, will be created in data folder')
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    global args, save_path
    args = parser.parse_args()
    data_dir = Path(args.data)
    print("=> fetching img pairs in '{}'".format(args.data))
    if args.output is None:
        save_path = data_dir/'ResultsNew'
    else:
        save_path = Path(args.output)
    print('=> will save everything to {}'.format(save_path))
    save_path.makedirs_p()

    # Data loading code
    input_transform = transforms.Compose([
        fusion_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        # transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])

    img_pairs = []
    for ext in args.img_exts:
        test_files = data_dir.files('*A.{}'.format(ext))
        for file in test_files:
            img_pair = file.parent / (file[15:24] + 'B.{}'.format(ext))

            if img_pair.isfile():
                img_pairs.append([file, img_pair])
    # print("img_pairs:",img_pairs)

    print('{} samples found'.format(len(img_pairs)))
    # create model
    network_data = torch.load(args.pretrained)
    print("=> using pre-trained model '{}'".format(network_data['arch']))
    model = models.__dict__[network_data['arch']](network_data).to(device)
    print('Total paprams: %.2fMB' % (sum(p.numel() for p in model.parameters()) / 1e6))
    model.eval()
    cudnn.benchmark = True

    for (img1_file, img2_file) in tqdm(img_pairs):
        img1 = input_transform(imread(img1_file))
        img2 = input_transform(imread(img2_file))
        input_var = torch.cat([img1, img2]).unsqueeze(0)
        input_var = input_var.to(device)
        output = model(input_var)


        for fusion_output in output:

            result_fusion = fusion_output[1:4, :]
            result_fusion = tensor2rgb(result_fusion)
            # print("result_fusion.shape:", result_fusion.shape, type(result_fusion))
            to_save = np.rint(result_fusion * 255)
            to_save[to_save<0] = 0
            to_save[to_save>255] = 255
            to_save = to_save.astype(np.uint8).transpose(1, 2, 0)
            pyplot.imsave(save_path / '{}{}.png'.format(img1_file[15:24], 'Fusion'), to_save)


def tensor2rgb(img_tensor):

    map_np = img_tensor.detach().cpu().numpy()

    return map_np


if __name__ == '__main__':
    main()
