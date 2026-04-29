# TODO:Evaluate test set accuracy
import argparse
import logging
import os
import time
import warnings
import torch
from torch.utils.data import DataLoader
import json
from utils.dataset import BasicDataset
from utils.eval import eval_net
from utils.logger import logger
from model.Build_models import MLoRA_SAM2, FT_SAM2, Adapter_SAM2, BitFit_SAM2

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# faster, but more memory
# from torch.backends import cudnn
# cudnn.benchmark = True
start_time = time.time()

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate test set', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in_chans', type=str2tuple, default="3, 1, 3", help='Channels of input images')
    parser.add_argument('--json_path', type=str, default=r"./json/test.json", help='.json path')
    parser.add_argument('--base_num', type=int, default=1024, help='Crop to the size of the predicted images')
    parser.add_argument('--background', default=False, help='Whether to consider background classes')
    parser.add_argument('--net_type', type=str, default='Adapter_SAM2', help='Net type')
    parser.add_argument('--batchsize', type=int, default=2, help='Number of batch size')
    parser.add_argument('--gpu_id', '-g', metavar='G', type=int, default=0, help='Number of gpu')
    return parser.parse_args()


def eval_dataset(n_img, json_path, class_json, runs, net, device, batch_size, img_shape, bg):
    test_dataset = BasicDataset(n_img, json_path, img_shape, query='test', if_transform=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    logging.info('Test set:')
    mIoU, mF1, OA, confusion, nc = eval_net(net, test_loader, device, class_json, n_img, bg)
    if bg:
        confusion.plot(runs + "Confusion matrix", n_classes=nc)


def create_path(path):
    try:
        os.mkdir(path)
        print(f'Created directory: {path}')
    except OSError:
        pass


def str2tuple(string):
    try:
        string = string.strip().strip('()[]')
        numbers = [int(x.strip()) for x in string.split(',')]
        return tuple(numbers)
    except:
        raise argparse.ArgumentTypeError('Must be a tuple of integers, e.g., "3, 1" or "(3, 1)"')


if __name__ == '__main__':
    args = get_args()
    in_chans = args.in_chans
    n_img = len(in_chans) if isinstance(in_chans, tuple) else 1
    img_size = args.base_num
    json_path = args.json_path
    net_type = args.net_type
    batchsize = args.batchsize
    gpu_id = args.gpu_id
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    assert os.path.exists(json_path), f"cannot find {json_path} file!"
    json_dict = json.load(open(json_path, 'r'))
    runs = json_dict['runs']
    create_path(runs)
    class_json = json_dict['classes']
    n_classes = len(json.load(open(class_json, 'r')))
    ckpt = json_dict['ckpt']
    logger('test_' + net_type, runs)

    # Creat model
    if net_type == 'FT_SAM2':
        net = FT_SAM2(n_classes=n_classes, in_chans=in_chans, topk=1, model_type="s", img_size=img_size)
    elif net_type == 'MLoRA_SAM2':
        net = MLoRA_SAM2(n_classes=n_classes, in_chans=in_chans, topk=1, model_type="s", img_size=img_size)
    elif net_type == 'Adapter_SAM2':
        net = Adapter_SAM2(n_classes=n_classes, in_chans=in_chans, topk=1, model_type="s", img_size=img_size)
    elif net_type == 'BitFit_SAM2':
        net = BitFit_SAM2(n_classes=n_classes, in_chans=in_chans, topk=1, model_type="s", img_size=img_size)
    else:
        raise NotImplementedError(f"Model type:'{net_type}' does not exist, please check the 'net_type' arg!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)

    net.load_state_dict(torch.load(ckpt + f"{net_type}_best_mIoU.pth", map_location=device))
    logging.info(f'Loading model from {ckpt + net_type}_best.pth\n')

    eval_dataset(n_img, json_path, class_json, runs, net, device, batchsize, img_size, args.background)
    logging.info(f'The entire {net_type} model test time is: {round((time.time() - start_time) / 3600, 2)} hours.')
