import os
import sys

sys.path.append("../..")
import functools
import numpy as np
import torch.nn as nn
import asamseg.utils as myut
import pytorch_lightning as pl
import asamseg.transforms as my_transforms

from monai.data import partition_dataset
from asamseg.data_modules import MicroDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from monai.transforms import Compose, LoadImaged, AddChanneld, ToTensord

from pytorch_lightning.loggers import TensorBoardLogger

root = '../..'  # 项目根目录

seed = 2202
folds = 1
gpu_num = 0
device = myut.try_gpu(gpu_num)
project_name = 'segmentation'
architecture_name = 'my_unet'
model_type = myut.get_model_type(architecture_name)
experiment_name = architecture_name + '_res_cam'

dataset_dir = os.path.join(root, 'data', 'MicroDataset')
data_csv_file = os.path.join(dataset_dir, 'csv_file', 'all.csv')
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')
gap_maps_dir = os.path.join(dataset_dir, 'gap_maps')
image_postfix = 'png'
label_postfix = 'png'
gap_maps_postfix = 'png'
train_batch_size = 10
eval_batch_size = 1
num_workers = 8
learning_rate = 3e-3
# learning_rate = 6e-3
learning_rate = learning_rate
total_epochs = 10000
num_trainset = 50
total_iterations = (num_trainset // train_batch_size) * total_epochs

class_list = ('bg', 'fg')

loss_func = nn.CrossEntropyLoss(reduction='none')

save_dir = './runs'

os.makedirs(os.path.join(save_dir, 'wandb'), exist_ok=True)

all_data = myut.get_data(
    data_csv_file,
    (images_dir, image_postfix),
    (labels_dir, label_postfix),
    (gap_maps_dir, gap_maps_postfix),
)
# partition_seed = [64752, 4173, 11037, 25524, 46996]
partition_seed = [4173]
train_data_list = []
val_data_list = []
test_data_list = []
for fold in range(folds):
    train_data, val_data, test_data = partition_dataset(all_data, ratios=[0.7, 0.1, 0.2], seed=partition_seed[fold],
                                                        shuffle=True)
    train_data_list.append(train_data)
    val_data_list.append(val_data)
    test_data_list.append(test_data)

pl.seed_everything(seed, workers=True)

val_result_list = []

print("train_data_list", train_data_list)
print("val_data_list", val_data_list)
print("test_data_list", test_data_list)
for fold in range(folds):
    train_mean, train_std = myut.cal_mean_std(
        [item['image'] for item in train_data_list[fold]]
    )

    train_transforms = Compose([
        LoadImaged(keys=['image', 'label'], dtype=np.uint8),
        my_transforms.Augumentation(),
        # my_transforms.InstanceToBinaryLabel(),
        my_transforms.ConvertLabelAndGenerateGapV2(),
        my_transforms.Normalize(mean=train_mean, std=train_std),
        AddChanneld(keys=['image']),
        ToTensord(keys=['image', 'label', 'gap_map']),
    ])
    train_transforms.set_random_state(seed=seed);
    eval_transforms = Compose([
        LoadImaged(keys=['image', 'label', 'gap_map'], dtype=np.uint8),
        # my_transforms.Augumentation(),
        my_transforms.InstanceToBinaryLabel(),
        #         my_transforms.ConvertLabelAndGenerateGapV2(),
        my_transforms.Normalize(mean=train_mean, std=train_std),
        AddChanneld(keys=['image']),
        ToTensord(keys=['image', 'label', 'gap_map']),
    ])
    eval_transforms.set_random_state(seed=seed);

    data_module = MicroDataModule(
        train_data_list[fold], train_transforms, train_batch_size, val_data_list[fold], eval_transforms,
        eval_batch_size, num_workers
    )

    model = model_type(
        in_channels=1,
        out_channels=len(class_list),
        extra_gap_weight=0.,
        # use_res_block=True,
        learning_rate=learning_rate,
        loss_func=loss_func,
        total_iterations=total_iterations,

    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min'
    )
    wandb_logger = WandbLogger(
        name=f'{experiment_name}', project=project_name, save_dir=save_dir
    )
    #     trainer = pl.Trainer(
    #         precision=16, gpus=[gpu_num], deterministic=True, logger=wandb_logger, progress_bar_refresh_rate=0,
    #         callbacks=[checkpoint_callback], check_val_every_n_epoch=10, max_epochs=total_epochs
    #     )
    # wandb_logger = TensorBoardLogger('logs', name='my_model')
    trainer = pl.Trainer(
        precision=16, gpus=[gpu_num], deterministic=True, logger=wandb_logger, progress_bar_refresh_rate=0,
        callbacks=[checkpoint_callback], check_val_every_n_epoch=10, max_epochs=total_epochs
    )
    trainer.fit(model, datamodule=data_module)

    val_dataset = myut.get_dataset(val_data_list[fold], eval_transforms)
    val_loader = myut.get_dataloader(
        dataset=val_dataset,
        shuffle=False,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        drop_last=False
    )

    model = model_type.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        in_channels=1,
        out_channels=len(class_list),
        extra_gap_weight=0.,
        # use_res_block=True,
        learning_rate=learning_rate,
        loss_func=loss_func,
        total_iterations=total_iterations,

    )
    model.freeze()
    model.to(device);

    val_result = myut.evaluate(model, val_loader, class_list, device, use_sliding_window=True)
    val_result_list.append(val_result)
    wandb_logger.finalize('success')

eval_result = functools.reduce(lambda x, y: x + y, val_result_list) / folds

result = myut.get_readable_eval_result(
    val_result_list + [eval_result],
    [f'fold_{i + 1}' for i in range(folds)] + ['mean']
)

print('result:', result)

print('end:')
