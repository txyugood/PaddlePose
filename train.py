import os

import paddle

from dataset import TopDownMpiiDataset
from transforms import LoadImageFromFile, TopDownRandomFlip, TopDownAffine, TopDownGetRandomScaleRotation, \
    TopDownGenerateTarget, Compose, Collect, NormalizeTensor
from top_down import TopDown

if __name__ == '__main__':
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
    dataset = TopDownMpiiDataset(ann_file='/Users/alex/baidu/mmpose/data/mpii/annotations/mpii_train.json',
                                 img_prefix='/Users/alex/baidu/mmpose/data/mpii/images',
                                 pipeline=tranforms)

    val_tranforms=[
        LoadImageFromFile(),
        TopDownAffine(),
        NormalizeTensor(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        Collect(keys=['img'],
                meta_keys=['image_file', 'center', 'scale', 'rotation', 'flip_pairs'])
    ]
    val_dataset = TopDownMpiiDataset(ann_file='/Users/alex/baidu/mmpose/data/mpii/annotations/mpii_val.json',
                                 img_prefix='/Users/alex/baidu/mmpose/data/mpii/images',
                                 pipeline=val_tranforms, test_mode=True)


    train_loader = paddle.io.DataLoader(
        dataset,
        num_workers=0,
        batch_size=2,
        shuffle=True,
        drop_last=True,
        return_list=True,
    )

    val_loader = paddle.io.DataLoader(val_dataset,
                                      batch_size=2,shuffle=False,drop_last=False,return_list=True)

    model = TopDown()
    learning_rate = paddle.optimizer.lr.MultiStepDecay(
        learning_rate=5e-4, milestones=[170, 200], gamma=0.1)
    lr = paddle.optimizer.lr.LinearWarmup(
        learning_rate=learning_rate,
        warmup_steps=500,
        start_lr=0,
        end_lr=5e-4)
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=lr)
    max_epochs = 210
    epoch = 0
    best_mean = 0
    while epoch < max_epochs:
        for batch_id, data in enumerate(train_loader):
            model.train()
            output = model.train_step(data, optimizer)
            loss = output['loss']
            loss.backward()
            optimizer.step()
            model.clear_gradients()
            lr.step()
            log_vars = output['log_vars']
            print(f'epoch:{epoch} batch_id:{batch_id} lr:{optimizer.get_lr()} loss:{log_vars["mse_loss"]} acc_pose:{log_vars["acc_pose"]}')
        i = 0
        results = []
        for data in val_loader:
            if i > 2:
                break
            model.eval()
            with paddle.no_grad():
                result = model(return_loss=False, **data)
            results.append(result)
            i += 1
        work_dir = './output'
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        eval_config = {'interval': 10, 'metric': 'PCKh', 'save_best': 'PCKh'}
        results = val_dataset.evaluate(results, work_dir, **eval_config)
        print(f'{epoch}:',end='')
        for k, v in sorted(results.items()):
            print(f'{k}: {v} ', end='')
        print('\n')
        if results['PCKh@0.1'] > best_mean:
            best_mean = results['PCKh@0.1']
            current_save_dir = os.path.join(work_dir, 'best_model')
            if not os.path.exists(current_save_dir):
                os.makedirs(current_save_dir)
            paddle.save(model.state_dict(),
                        os.path.join(current_save_dir, 'model.pdparams'))
            paddle.save(optimizer.state_dict(),
                        os.path.join(current_save_dir, 'model.pdopt'))


        epoch += 1
