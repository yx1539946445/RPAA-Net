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
from monai.networks.utils import eval_mode
from monai.transforms import Compose, LoadImaged, AddChanneld, ToTensord

root = '../..'  # 项目根目录

seed = 2202
folds = 5
gpu_num = 0
device = myut.try_gpu(gpu_num)
project_name = 'segmentation'
# architecture_name = 'my_unet'
# architecture_name = 'my_ccnet'
# architecture_name = 'my_segnet'
# architecture_name = 'my_pspnet'
# architecture_name = 'my_spnet'
architecture_name = 'my_cenet'
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
partition_seed = [64752, 4173, 11037, 25524, 46996]
#
# partition_seed = [4173]

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

checkpoint_list = [
    # # # 权重实验 ccnet  + 不加权重      aclf9oum
    # 'epoch=8559-step=42799.ckpt',  # 1
    # 'epoch=9459-step=47299.ckpt',  # 2
    # 'epoch=9189-step=45949.ckpt',  # 3
    # 'epoch=8269-step=41349.ckpt',  # 4
    # 'epoch=8649-step=43249.ckpt',  # 5

    # # # 权重实验 10 unet + 加CA注意力  20rf40um
    # 'epoch=9219-step=46099.ckpt',  # 1
    # 'epoch=9299-step=46499.ckpt',  # 2
    # 'epoch=9669-step=48349.ckpt',  # 3
    # 'epoch=8649-step=43249.ckpt',  # 4
    # 'epoch=9749-step=48749.ckpt',  # 5

    # # 权重实验 pspnet  + 不加权重  pxfdbdeg
    # 'epoch=9309-step=46549.ckpt',  # 1
    # 'epoch=7999-step=39999.ckpt',  # 2
    # 'epoch=8719-step=43599.ckpt',  # 3
    # 'epoch=7819-step=39099.ckpt',  # 4
    # 'epoch=9749-step=48749.ckpt',  # 5

    # # # 权重实验 segnet  + 不加权重  segnet
    # 'epoch=9579-step=47899.ckpt',  # 1
    # 'epoch=9279-step=46399.ckpt',  # 2
    # 'epoch=9409-step=47049.ckpt',  # 3
    # 'epoch=8359-step=41799.ckpt',  # 4
    # 'epoch=8289-step=41449.ckpt',  # 5

    # 权重实验 spnet  + 不加权重    spnet
    # 'epoch=7669-step=38349.ckpt',  # 1
    # 'epoch=9389-step=46949.ckpt',  # 2
    # 'epoch=7519-step=37599.ckpt',  # 3
    # 'epoch=8389-step=41949.ckpt',  # 4
    # 'epoch=8819-step=44099.ckpt',  # 5

    # 权重实验 cenet  + 不加权重    1vb7iiwp
    'epoch=19-step=99.ckpt',  # 1
    # 'epoch=9389-step=46949.ckpt',  # 2
    # 'epoch=7519-step=37599.ckpt',  # 3
    # 'epoch=8389-step=41949.ckpt',  # 4
    # 'epoch=8819-step=44099.ckpt',  # 5
]

val_result_list = []
timer = myut.Timer()

fold = 0
train_mean, train_std = myut.cal_mean_std(
    [item['image'] for item in train_data_list[fold]]
)

eval_transforms = Compose([
    LoadImaged(keys=['image', 'label'], dtype=np.uint8),
    my_transforms.CenterCropForVisualize(),
    my_transforms.InstanceToBinaryLabel(),
    my_transforms.Normalize(mean=train_mean, std=train_std),
    AddChanneld(keys=['image']),
    ToTensord(keys=['image', 'label']),
])
eval_transforms.set_random_state(seed=seed);

val_dataset = myut.get_dataset(val_data_list[fold], eval_transforms)
val_loader = myut.get_dataloader(
    dataset=val_dataset,
    shuffle=False,
    batch_size=eval_batch_size,
    num_workers=num_workers,
    drop_last=False
)

model = model_type.load_from_checkpoint(
    os.path.join(save_dir, project_name, '1vb7iiwp/checkpoints', checkpoint_list[fold]),
    in_channels=1,
    out_channels=len(class_list),
    extra_gap_weight=0.,
    learning_rate=learning_rate,
    loss_func=loss_func,
    total_iterations=total_iterations
)
model.freeze()
model.to(device);

for i in range(100):
    with eval_mode(model):
        for batch in val_loader:
            images, labels = myut.get_batch_data(batch, ('image', 'label'))
            images, labels = images.float().to(device), labels.long().to(device)
            #             print(images.size())
            timer.start()
            preds = model(images)
            timer.stop()
            break

print(f'{timer.avg():.5f} sec')
