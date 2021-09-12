import os
import time
import argparse

import paddle

from dataset import TopDownMpiiDataset
from transforms import LoadImageFromFile, TopDownRandomFlip, TopDownAffine, TopDownGetRandomScaleRotation, \
    TopDownGenerateTarget, Compose, Collect, NormalizeTensor
from top_down import TopDown
from timer import TimeAverager, calculate_eta

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
        '--dataset_root',
        dest='dataset_root',
        help='The path of dataset root',
        type=str,
        default=None)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    tranforms = [
        LoadImageFromFile(),
        TopDownRandomFlip(flip_prob=0.5),
        TopDownGetRandomScaleRotation(rot_factor=40, scale_factor=0.5),
        TopDownAffine(),
        NormalizeTensor(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        TopDownGenerateTarget(sigma=2),
        Collect(keys=['img', 'target', 'target_weight'],
                meta_keys=[
                    'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
                    'rotation', 'flip_pairs'
                ])
    ]
    dataset = TopDownMpiiDataset(ann_file=os.path.join(args.dataset_root, 'annotations/mpii_train.json'),
                                 img_prefix=os.path.join(args.dataset_root, 'images'),
                                 pipeline=tranforms, image_size=args.image_size)

    val_tranforms=[
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

    batch_size = 64
    train_loader = paddle.io.DataLoader(
        dataset,
        num_workers=0,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        return_list=True,
    )

    val_loader = paddle.io.DataLoader(val_dataset,
                                      batch_size=batch_size // 2,shuffle=False,drop_last=False,return_list=True)

    model = TopDown()

    iters_per_epoch = len(train_loader)
    learning_rate = paddle.optimizer.lr.MultiStepDecay(
        learning_rate=5e-4, milestones=[170 * iters_per_epoch, 200 * iters_per_epoch], gamma=0.1)
    lr = paddle.optimizer.lr.LinearWarmup(
        learning_rate=learning_rate,
        warmup_steps=500,
        start_lr=0,
        end_lr=5e-4)
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=lr)

    avg_loss = 0.0
    avg_pose_acc = 0.0
    max_epochs = 210
    epoch = 0
    best_mean = 0
    log_iters = 10
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    
    iters = iters_per_epoch * max_epochs
    iter = 0
    batch_start = time.time()
    while epoch < max_epochs:

        model.train()
        for batch_id, data in enumerate(train_loader):
            reader_cost_averager.record(time.time() - batch_start)
            iter += 1
            output = model.train_step(data, optimizer)
            loss = output['loss']
            loss.backward()
            optimizer.step()
            model.clear_gradients()
            avg_loss += loss.numpy()[0]
            log_vars = output['log_vars']
            avg_pose_acc += log_vars["acc_pose"]
            lr.step()
            batch_cost_averager.record(
                time.time() - batch_start, num_samples=batch_size)
            if (iter) % log_iters == 0:
                avg_loss /= log_iters
                avg_pose_acc /= log_iters
                remain_iters = iters - iter
                avg_train_batch_cost = batch_cost_averager.get_average()
                avg_train_reader_cost = reader_cost_averager.get_average()
                eta = calculate_eta(remain_iters, avg_train_batch_cost)
        
                print(
                    "[TRAIN] epoch={}, batch_id={}, loss={:.6f}, lr={:.6f},acc_pose={:.3f} ETA {}"
                        .format(epoch, batch_id + 1,
                                avg_loss, optimizer.get_lr(), avg_pose_acc, eta))
                avg_loss = 0.0
                avg_pose_acc = 0.0
                reader_cost_averager.reset()
                batch_cost_averager.reset()
        
            # print(f'epoch:{epoch} batch_id:{batch_id} lr:{optimizer.get_lr()} loss:{log_vars["mse_loss"]} acc_pose:{log_vars["acc_pose"]}')
            batch_start = time.time()

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
        print(f'[EVAL] epoch={epoch}:',end='')
        for k, v in sorted(results.items()):
            print(f'{k}={v} ', end='')
        print('')
        if results['Mean@0.1'] > best_mean:
            best_mean = results['Mean@0.1']
            current_save_dir = os.path.join(work_dir, 'best_model')
            if not os.path.exists(current_save_dir):
                os.makedirs(current_save_dir)
            paddle.save(model.state_dict(),
                        os.path.join(current_save_dir, 'model.pdparams'))
            paddle.save(optimizer.state_dict(),
                        os.path.join(current_save_dir, 'model.pdopt'))


        epoch += 1
