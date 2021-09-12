import os
import time
import argparse

import paddle

from dataset import TopDownMpiiDataset
from transforms import LoadImageFromFile, TopDownRandomFlip, TopDownAffine, TopDownGetRandomScaleRotation, \
    TopDownGenerateTarget, Compose, Collect, NormalizeTensor
from top_down import TopDown
from timer import TimeAverager, calculate_eta
from utils import load_pretrained_model


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')
    # params of training
    parser.add_argument(
        '--image_size',
        dest='image_size',
        help='image size for sample',
        type=int,
        default=256)

    parser.add_argument(
        '--pretrained_model',
        dest='pretrained_model',
        help='The directory for pretrained model',
        type=str,
        default='./output/best_model/model.pdparams')

    parser.add_argument(
        '--dataset_root',
        dest='dataset_root',
        help='The path of dataset root',
        type=str,
        default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    val_tranforms = [
        LoadImageFromFile(),
        TopDownAffine(),
        NormalizeTensor(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        Collect(keys=['img'],
                meta_keys=['image_file', 'center', 'scale', 'rotation', 'flip_pairs'])
    ]
    val_dataset = TopDownMpiiDataset(ann_file=os.path.join(args.dataset_root, 'annotations/mpii_val.json'),
                                     img_prefix=os.path.join(args.dataset_root, 'images'),
                                     pipeline=val_tranforms, image_size=args.image_size, test_mode=True)

    batch_size = 32
    val_loader = paddle.io.DataLoader(val_dataset,
                                      batch_size=batch_size, shuffle=False, drop_last=False, return_list=True)

    model = TopDown()
    load_pretrained_model(model, args.pretrained_model)
    model.eval()
    results = []
    for data in val_loader:
        with paddle.no_grad():
            result = model(return_loss=False, **data)
        results.append(result)

    work_dir = './output'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    eval_config = {'interval': 10, 'metric': 'PCKh', 'save_best': 'PCKh'}
    results = val_dataset.evaluate(results, work_dir, **eval_config)
    print(f'[EVAL] ', end='')
    for k, v in sorted(results.items()):
        print(f'{k}={v} ', end='')
    print('')
