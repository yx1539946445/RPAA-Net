import os
import sys

sys.path.append("../..")

import torch
import torch.nn as nn
import asamseg.utils as myut
import torchvision.models as models
from ptflops import get_model_complexity_info

train_batch_size = 10
learning_rate = 3e-3
total_epochs = 10000
num_trainset = 50
total_iterations = (num_trainset // train_batch_size) * total_epochs

class_list = ('bg', 'fg')

loss_func = nn.CrossEntropyLoss(reduction='none')

model_type = myut.get_model_type('my_ccnet')
with torch.cuda.device(0):
    net = model_type(
        in_channels=1,
        out_channels=len(class_list),
        extra_gap_weight=0.,
        # use_res_block=True,
        learning_rate=learning_rate,
        loss_func=loss_func,
        total_iterations=total_iterations,
    )
    macs, params = get_model_complexity_info(net, (1, 256, 256), as_strings=False,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity(GMACs): ', round(macs / 10 ** 9, 2)))
    print('{:<30}  {:<8}'.format('Computational complexity(GFLOPs): ', round(2 * macs / 10 ** 9, 2)))
    print('{:<30}  {:<8}'.format('Number of parameters(MB): ', round(params / 10 ** 6, 2)))
