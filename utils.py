import os
import cv2
import math
import time
import torch
import random
import skimage
import torchmetrics
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import asamseg.evaluation as ev
import asamseg.stats_utils as su
import skimage.color as color
import skimage.measure as measure
import asamseg.models as my_models
import skimage.morphology as morph
from PIL import Image
from torch import optim
from torchvision import transforms
from typing import List, Optional, Sequence, Tuple
from kornia.utils import tensor_to_image
from kornia.color import grayscale_to_rgb
from kornia.augmentation import Denormalize
from kornia.color.gray import grayscale_to_rgb
from monai.transforms import LoadImage
from monai.inferers import SlidingWindowInferer, SimpleInferer
from monai.networks.utils import predict_segmentation, eval_mode
from monai.data import CacheDataset, DataLoader, list_data_collate
import numpy
import gala.evaluate as gala_ev

# 避免除零
_SMOOTH = 1e-06


class Accumulator():
    def __init__(self):
        self.total = 0
        self.count = 0

    def addData(self, val, n=1):
        self.total += val * n
        self.count += n

    def mean(self):
        return self.total / (self.count + _SMOOTH)


class Timer():
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()


def get_weight_bias(model, type_list):
    weight, bias = [], []
    for m in model.modules():
        if isinstance(m, type_list):
            weight += [m.weight]
            if not m.bias is None:
                bias += [m.bias]
    return weight, bias


def get_parameters(model, type_list):
    parameters = []
    for m in model.modules():
        if isinstance(m, type_list):
            parameters += m.parameters()
    return parameters


def configure_optimizers(model, learning_rate):
    lr = learning_rate
    weight_p, bias_p = get_weight_bias(model, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear))
    normalization_p = get_parameters(model, (nn.BatchNorm2d, nn.GroupNorm,))
    #     optimizer = optim.Adam([
    #         {'params': weight_p, 'weight_decay': 0.0005, 'lr': lr},
    #         {'params': bias_p, 'weight_decay': 0, 'lr': lr},
    #         {'params': normalization_p, 'weight_decay': 0, 'lr': lr}
    #     ])
    optimizer = optim.SGD([
        {'params': weight_p, 'weight_decay': 0.0005, 'lr': lr},
        {'params': bias_p, 'weight_decay': 0, 'lr': lr},
        {'params': normalization_p, 'weight_decay': 0, 'lr': lr}
    ], momentum=0.99)

    return optimizer


def get_batch_data(batch, keys=('image', 'label', 'gap_map')):
    return (batch[key] for key in keys)


def cal_batch_loss(model, batch, loss_func, num_classes=2, use_sliding_window=False):
    '''
    Calculate loss of one batch.
    '''
    images, labels, gap_maps = get_batch_data(batch, ('image', 'label', 'gap_map'))
    images, labels, gap_maps = images.float(), labels.long(), gap_maps.long()
    inferer = get_inferer(use_sliding_window=use_sliding_window)
    preds = inferer(inputs=images, network=model)
    loss = loss_func(preds.contiguous(), labels.contiguous())
    return loss


def cal_batch_loss_aux(model, batch, loss_func, num_classes=2, use_sliding_window=False):
    '''
    Calculate loss of one batch.
    '''

    images, labels, gap_maps = get_batch_data(batch, ('image', 'label', 'gap_map'))
    images, labels, gap_maps = images.float(), labels.long(), gap_maps.long()
    inferer = get_inferer(use_sliding_window=use_sliding_window)
    aux, preds = inferer(inputs=images, network=model)
    loss = loss_func(preds.contiguous(), labels.contiguous())
    aux_loss = loss_func(aux.contiguous(), labels.contiguous())
    loss = loss + aux_loss * 0.4
    return loss.mean()


def cal_batch_loss_aux_gap(model, batch, loss_func, extra_gap_weight=0., num_classes=2, use_sliding_window=False):
    '''
    Calculate loss of one batch.
    '''
    images, labels, gap_maps = get_batch_data(batch, ('image', 'label', 'gap_map'))
    images, labels, gap_maps = images.float(), labels.long(), gap_maps.long()
    inferer = get_inferer(use_sliding_window=use_sliding_window)
    aux, preds = inferer(inputs=images, network=model)
    loss = loss_func(preds.contiguous(), labels.contiguous())
    aux_loss = loss_func(aux.contiguous(), labels.contiguous())
    loss = loss + aux_loss * 0.4 + loss * gap_maps * extra_gap_weight
    return loss.mean()


def cal_batch_loss_gap(model, batch, loss_func, extra_gap_weight=0., n_classes=2, use_sliding_window=False):
    '''
    Calculate loss of one batch.
    Args:
        loss_func: reduction must be none.
    '''
    images, labels, gap_maps = get_batch_data(batch, ('image', 'label', 'gap_map'))
    # gap_maps_list = []
    # for i in range(gap_maps.shape[0]):
    #     gap_map = gap_maps.cpu()[i]
    #     # gap_maps_numpy = np.array(gap_maps_cpu)
    #     # gap_maps_numpy = gap_maps_numpy.data.numpy().transpose(1,2,0)
    #     # image = cv2.imread(gap_maps_numpy)
    #     gap_map = np.uint8(gap_map * 255)
    #     gap_map = binary_label_to_readable_image(gap_map)
    #     # gap_map = gap_map.transpose(1, 2, 0)
    #     # gray = cv2.cvtColor(gap_map, cv2.COLOR_BGR2GRAY)
    #     ret, threshold = cv2.threshold(gap_map, 127, 255, cv2.THRESH_BINARY)
    #     contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     gap_maps_list.append(len(contours))
    # b, h, w = gap_maps.shape
    # gap_maps_cpu = gap_maps.cpu()
    # gap_maps_numpy = np.array(gap_maps_cpu)
    # gap_maps_numpy = gap_maps_numpy.data.numpy()
    images, labels, gap_maps = images.float(), labels.long(), gap_maps.long()
    inferer = get_inferer(use_sliding_window=use_sliding_window)
    preds = inferer(inputs=images, network=model)
    loss_main = loss_func(preds.contiguous(), labels.contiguous())  # 0.6631
    # zero_gaps = torch.ones([b, h, w])
    # zero_gaps = zero_gaps.data.numpy()

    # zero_gaps_0 = torch.zeros([b, h, w]).cuda(0)
    # zero_gaps_1 = torch.ones([b, h, w]).cuda(0)
    # zero_gaps = torch.stack([zero_gaps_1, zero_gaps_0], dim=0).transpose(0, 1)

    # loss_aux = loss_func(zero_gaps.contiguous(), gap_maps.contiguous())  # 0.3334
    # # loss_aux = F.nll_loss(zero_gaps.contiguous(), gap_maps.contiguous())
    # loss = loss + loss * extra_gap_weight * gap_maps
    # print("loss_main", loss_main.mean())
    # # loss_aux = gap_maps * loss_main * np.mean(gap_maps_list)
    # loss_aux = loss_func(zero_gaps.contiguous(), gap_maps.contiguous()) * gap_maps * loss_main * np.mean(gap_maps_list)  # 0.3334
    # print("loss_aux", loss_aux.mean())
    # loss = loss_main + loss_aux
    loss = loss_main + loss_main * gap_maps * extra_gap_weight
    return loss.mean()


def get_losses_weights(losses: [list, numpy.ndarray, torch.Tensor]):
    if type(losses) != torch.Tensor:
        losses = torch.tensor(losses)
    weights = torch.div(losses, torch.sum(losses)) * losses.shape[0]
    return weights


def cal_mean_std(image_list, rgb=False):
    '''
    Calculate the mean and standard deviation of the image data set.
    '''
    mean = torch.tensor([0, 0, 0] if rgb else [0], dtype=torch.float)
    std = torch.tensor([0, 0, 0] if rgb else [0], dtype=torch.float)
    pixel_count = 0
    to_tensor = transforms.ToTensor()
    loader = LoadImage(dtype=np.uint8, image_only=True)

    for i in range(len(image_list)):
        image = loader(image_list[i])
        image = to_tensor(image)
        mean += image.sum(dim=(1, 2))
        pixel_count += image.size()[1] * image.size()[2]

    mean /= pixel_count

    for i in range(len(image_list)):
        image = loader(image_list[i])
        image = to_tensor(image)
        std += ((image - mean.unsqueeze(1).unsqueeze(1)) ** 2).sum(dim=(1, 2))

    std = torch.sqrt(std / pixel_count)

    return mean, std


def instance_to_mask(instance):
    '''
    Args:
        instance: PIL or numpy image
    '''
    instance = np.array(instance, dtype=np.uint32)
    instance = 256 * (256 * instance[:, :, 0] + instance[:, :, 1]) + instance[:, :, 2]
    object_list = np.unique(instance[instance != 0])
    current_num = 1
    for obj in object_list:
        instance[instance == obj] = current_num
        current_num += 1
    return instance.astype(np.uint8)


def mask_small_object(mask, area_thresh, pixels=[0, 1, 2], weights=[0.5, 1.0, 10.0]):
    """
    :param mask: 输入mask掩码
    :param area_thresh: 小物体的阈值，面积小于该值认为是小物体
    :param pixels: 背景 小物体 大物体的mask值
    :param weight: 背景，前景大物体，前景小物体的权重
    :return: 权重图
    """

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))

    for i in range(len(area)):
        if area[i] < area_thresh:
            cv2.fillPoly(mask, [contours[i]], (2,))
    for i in range(len(pixels)):
        mask[mask == [pixels[i]]] = weights[i]
    return mask


def instance_mask_to_binary_label(instance_mask):
    '''
    Args:
        instance_mask: PIL or numpy image
    '''
    instance_mask = np.array(instance_mask)
    instance_mask[instance_mask != 0] = 1
    return instance_mask


def binary_label_to_readable_image(label):
    '''
    Args:
        label: PIL or numpy image
    '''
    label = np.array(label)
    label[label == 1] = 255
    return label.astype(np.uint8)


def label_to_rgb(label):
    rgb_image = np.empty(label.shape + (3,), dtype=np.float64)
    rgb_image[label == 0] = np.array([0, 0, 0], dtype=np.float64)
    rgb_image[label == 1] = np.array([1, 1, 1], dtype=np.float64)
    return rgb_image


def _file_params_is_invalid(file_params):
    return type(file_params) != tuple or \
           len(file_params) != 2 or \
           type(file_params[0]) != str


def _gen_file_path(file_params, filename):
    return os.path.join(file_params[0], filename + '.' + file_params[1])


def get_data(
        data_csv: str,
        image_params: Tuple[str],
        label_params: Tuple[str],
        gap_map_params: Optional[Tuple[str]] = None
) -> List:
    '''
    Args:
        image_params:   (images_dir, image_postfix)
        label_params:   (labels_dir, label_postfix)
        gap_map_params: (gap_maps_dir, gap_map_postfix)
    '''
    if _file_params_is_invalid(image_params):
        raise ValueError('get_data(): image_params is invalid!')
    if _file_params_is_invalid(label_params):
        raise ValueError('get_data(): label_params is invalid!')
    if gap_map_params is not None and _file_params_is_invalid(gap_map_params):
        raise ValueError('get_data(): gap_map_params is invalid!')

    data_list = []
    data_csv = pd.read_csv(data_csv, dtype=str)
    for i in range(len(data_csv)):
        filename = data_csv.iloc[i, 0]
        data_item = {}
        data_item['image'] = _gen_file_path(image_params, filename)
        data_item['label'] = _gen_file_path(label_params, filename)
        if gap_map_params is not None:
            data_item['gap_map'] = _gen_file_path(gap_map_params, filename)
        data_list.append(data_item)
    return data_list


def get_dataset(data, transform, num_workers=None):
    return CacheDataset(
        data,
        transform,
        cache_num=9223372036854775807,
        cache_rate=1.0,
        num_workers=num_workers,
        progress=True
    )


def get_dataloader(dataset, shuffle, batch_size, num_workers, drop_last):
    return DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=drop_last,
        collate_fn=list_data_collate
    )


def plot_image_list(image_list, title_list, figsize=(16, 12), cmap=None):
    plt.figure(figsize=figsize)
    for i, (image, title) in enumerate(zip(image_list, title_list)):
        plt.subplot(int(math.ceil(len(image_list) / 3)), 3, i + 1)
        plt.title(title)
        plt.axis('off')
        plt.imshow(image, cmap=cmap)


def show_result(
        model, dataloader, device, train_mean, train_std, num=-1,
        use_sliding_window=False, save_dir=None, file_start_num=0,
        label_instance=True,
        need_softmax_argmax=True,
):
    '''
    Args:
        file_start_num: start number of the saved file
    '''
    colors = [
        'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure',
        'beige', 'bisque', 'blanchedalmond', 'blue', 'blueviolet',
        'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate',
        'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan',
        'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen',
        'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
        'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue',
        'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet',
        'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue',
        'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro',
        'ghostwhite', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow',
        'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory',
        'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon',
        'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray',
        'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen',
        'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue',
        'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon',
        'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple',
        'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise',
        'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin',
        'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered',
        'orchid', 'palegoldenrod', 'palegreen', 'palevioletred', 'papayawhip',
        'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown',
        'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell',
        'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow',
        'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise',
        'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen',
    ]
    random.shuffle(colors)
    inferer = get_inferer(use_sliding_window=use_sliding_window)
    denormalize = Denormalize(mean=train_mean, std=train_std)
    with eval_mode(model):
        for i, batch in enumerate(dataloader):
            if num == -1 or num == i:
                break
            images, labels = get_batch_data(batch, ('image', 'label'))
            images = images.to(device)
            with torch.no_grad():
                predicts = inferer(inputs=images, network=model)
                if need_softmax_argmax:
                    predicts = predict_segmentation(predicts, mutually_exclusive=True)
                else:
                    predicts = predicts.squeeze(dim=1)
                batch_size = predicts.size()[0]
                for j, (image, label, predict) in enumerate(zip(images, labels, predicts)):
                    image = denormalize(image)
                    image = grayscale_to_rgb(image)
                    image = tensor_to_image(image)

                    ground_truth = tensor_to_image(label)
                    predict = tensor_to_image(predict)
                    if label_instance:
                        ground_truth = color.label2rgb(ground_truth, bg_label=0, colors=colors)

                        predict = measure.label(predict)  # 标记不同连通域
                        predict = color.label2rgb(predict, bg_label=0, colors=colors)
                    else:
                        ground_truth = label_to_rgb(ground_truth)
                        predict = label_to_rgb(predict)

                    if save_dir is not None:
                        image_dir = os.path.join(save_dir, 'image')
                        ground_truth_dir = os.path.join(save_dir, 'ground_truth')
                        predict_dir = os.path.join(save_dir, 'predict')
                        os.makedirs(image_dir, exist_ok=True)
                        os.makedirs(ground_truth_dir, exist_ok=True)
                        os.makedirs(predict_dir, exist_ok=True)

                        file_num = file_start_num + i * batch_size + j + 1
                        tmp_image = Image.fromarray((image * 255).astype(np.uint8))
                        tmp_ground_truth = Image.fromarray((ground_truth * 255).astype(np.uint8))
                        tmp_predict = Image.fromarray((predict * 255).astype(np.uint8))

                        tmp_image.save(os.path.join(image_dir, f'image_{file_num:02}.png'))
                        tmp_ground_truth.save(os.path.join(ground_truth_dir, f'ground_truth_{file_num:02}.png'))
                        tmp_predict.save(os.path.join(predict_dir, f'predict_{file_num:02}.png'))

                    plot_image_list(
                        image_list=(image, ground_truth, predict),
                        title_list=('original image', 'groud truth', 'prediction'),
                        figsize=(36, 24),
                        # cmap=plt.cm.gray
                    )


def _get_heatmap(feature):
    heatmap = None
    heatmap = cv2.normalize(feature, heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = heatmap[:, :, ::-1]
    return heatmap


def visualize_feature_relevance(
        model,
        dataloader,
        device,
        train_mean,
        train_std,
        cam_position=2,
        class_num=1,
        num=-1,
        use_sliding_window=False,
        save_dir=None,
        file_start_num=0,
        need_softmax_argmax=True
):
    if class_num < 0 or class_num > 1:
        raise ValueError('visualize_feature_relevance(): class_num is invalid!')

    inferer = get_inferer(use_sliding_window=use_sliding_window)
    with eval_mode(model):
        for i, batch in enumerate(dataloader):
            if num == -1 or num == i:
                break
            images, labels = get_batch_data(batch, ('image', 'label'))
            images, labels = images.float().to(device), labels.long().to(device)

            denormalize = Denormalize(mean=train_mean, std=train_std)
            image = denormalize(images[0])
            image = tensor_to_image(image)
            ground_truth = binary_label_to_readable_image(labels[0].detach().cpu().numpy())
            plot_image_list(
                image_list=(image, ground_truth),
                title_list=('original image', 'groud truth'),
                figsize=(36, 24),
                cmap=plt.cm.gray
            )

            with torch.no_grad():
                predicts, rel_list, _, feature_before_cam_list, _ = inferer(inputs=images, network=model)
                if need_softmax_argmax:
                    predicts = predict_segmentation(predicts, mutually_exclusive=True)
                else:
                    predicts = predicts.squeeze(dim=1)
                batch_size = predicts.size()[0]
                for j, (image, label, predict) in enumerate(zip(images, labels, predicts)):
                    image = denormalize(image)
                    image = grayscale_to_rgb(image)
                    image = tensor_to_image(image)

                    ground_truth = tensor_to_image(label)
                    ground_truth = label_to_rgb(ground_truth)
                    predict = tensor_to_image(predict)
                    predict = label_to_rgb(predict)

                    h, w = images.size()[2:]
                    min_channel = rel_list[cam_position][class_num].argmin()
                    max_channel = rel_list[cam_position][class_num].argmax()
                    feature_before = feature_before_cam_list[cam_position]
                    feature_before = F.interpolate(feature_before.float(), size=(h, w), mode='bilinear')
                    feature_before = _scale_feature_map(feature_before, 0, 1)
                    feature_before = feature_before.detach().cpu()

                    min_channel_feature_before = (feature_before[0][min_channel] * 255).numpy().astype(np.uint8)
                    max_channel_feature_before = (feature_before[0][max_channel] * 255).numpy().astype(np.uint8)
                    min_channel_feature_before = _get_heatmap(min_channel_feature_before)
                    max_channel_feature_before = _get_heatmap(max_channel_feature_before)

                    if save_dir is not None:
                        image_dir = os.path.join(save_dir, 'image')
                        ground_truth_dir = os.path.join(save_dir, 'ground_truth')
                        predict_dir = os.path.join(save_dir, 'predict')
                        min_channel_heatmap_dir = os.path.join(save_dir, 'min_channel_heatmap')
                        max_channel_heatmap_dir = os.path.join(save_dir, 'max_channel_heatmap')
                        os.makedirs(image_dir, exist_ok=True)
                        os.makedirs(ground_truth_dir, exist_ok=True)
                        os.makedirs(predict_dir, exist_ok=True)
                        os.makedirs(min_channel_heatmap_dir, exist_ok=True)
                        os.makedirs(max_channel_heatmap_dir, exist_ok=True)

                        file_num = file_start_num + i * batch_size + j
                        tmp_image = Image.fromarray((image * 255).astype(np.uint8))
                        tmp_ground_truth = Image.fromarray((ground_truth * 255).astype(np.uint8))
                        tmp_predict = Image.fromarray((predict * 255).astype(np.uint8))
                        tmp_min_channel_heatmap = Image.fromarray(min_channel_feature_before.astype(np.uint8))
                        tmp_max_channel_heatmap = Image.fromarray(max_channel_feature_before.astype(np.uint8))

                        tmp_image.save(os.path.join(image_dir, f'image_{file_num:02}.png'))
                        tmp_ground_truth.save(os.path.join(ground_truth_dir, f'ground_truth_{file_num:02}.png'))
                        tmp_predict.save(os.path.join(predict_dir, f'predict_{file_num:02}.png'))
                        tmp_min_channel_heatmap.save(
                            os.path.join(min_channel_heatmap_dir, f'min_channel_heatmap_{file_num:02}.png'))
                        tmp_max_channel_heatmap.save(
                            os.path.join(max_channel_heatmap_dir, f'max_channel_heatmap_{file_num:02}.png'))

                    plot_image_list(
                        image_list=(min_channel_feature_before, max_channel_feature_before),
                        title_list=('min channel', 'max channel'),
                        figsize=(36, 24),
                    )


def _scale_feature_map(feature_map, a, b):
    max_v = feature_map.max()
    min_v = feature_map.min()
    k = (b - a) / (max_v - min_v + _SMOOTH)
    return a + k * (feature_map - min_v)


def try_gpu(num=0):
    if 0 <= num and num < torch.cuda.device_count():
        return torch.device(f'cuda:{num}')
    return torch.device('cpu')


def evaluate(model, dataloader, class_list, device, use_sliding_window=False, need_softmax_argmax=True):
    inferer = get_inferer(use_sliding_window=use_sliding_window)
    classification_metrics_calculator = ev.ClassificationMetricsCalculator(num_classes=len(class_list))
    confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=len(class_list)).to(device)
    objDice = ev.ObjectDice().to(device)
    aji = Accumulator()
    merger_error_ = Accumulator()
    split_error_ = Accumulator()
    vi_ = Accumulator()
    with eval_mode(model):
        for batch in dataloader:
            images, labels = get_batch_data(batch, ('image', 'label'))
            images, labels = images.float().to(device), labels.long().to(device)
            preds = inferer(inputs=images, network=model)
            if need_softmax_argmax:
                preds = preds.softmax(dim=1).argmax(dim=1)
            else:
                preds = preds.squeeze(dim=1)

            confusion_matrix(preds, labels)
            for i in range(preds.size()[0]):
                pred = preds[i].detach().cpu().numpy()
                label = labels[i].detach().cpu().numpy()
                merger_error, split_error, vi = get_vi(pred, label)
                merger_error_.addData(merger_error)
                split_error_.addData(split_error)
                vi_.addData(vi)
                objDice(
                    torch.from_numpy(measure.label(pred)),
                    torch.from_numpy(measure.label(label))
                )
                pred = morph.label(pred, connectivity=2)
                label = morph.label(label, connectivity=2)
                pred = su.remap_label(pred, by_size=False)
                label = su.remap_label(label, by_size=False)

                aji.addData(su.get_fast_aji(label, pred))
    d = {}
    d['AJI'] = aji.mean()
    d['VI'] = vi_.mean()
    d['ME'] = merger_error_.mean()
    d['SE'] = split_error_.mean()
    instance_level_metrics = pd.Series(d)
    metrics = classification_metrics_calculator.get_metrics(confusion_matrix.compute())
    # return metrics.append(instance_level_metrics).append(objDice.compute())
    metrics = pd.concat([metrics, instance_level_metrics])
    metrics = pd.concat([metrics, objDice.compute()])
    return metrics


def get_vi(pred: np.ndarray, mask: np.ndarray, bg_value: int = 0, method: int = 1) -> Tuple:
    """
    Referenced by:
    Marina Meilă (2007), Comparing clusterings—an information based distance,
    Journal of Multivariate Analysis, Volume 98, Issue 5, Pages 873-895, ISSN 0047-259X, DOI:10.1016/j.jmva.2006.11.013.
    :param method: 0: skimage implementation and 1: gala implementation (https://github.com/janelia-flyem/gala.)
    :return Tuple = (VI, merger_error, split_error)
    """
    vi, merger_error, split_error = 0.0, 0.0, 0.0

    label_pred, num_pred = measure.label(pred, connectivity=2, background=bg_value, return_num=True)
    label_mask, num_mask = measure.label(mask, connectivity=2, background=bg_value, return_num=True)
    if method == 0:
        # scikit-image
        split_error, merger_error = skimage.variation_of_information(label_mask, label_pred)
    elif method == 1:
        # gala
        merger_error, split_error = gala_ev.split_vi(label_pred, label_mask)
    vi = merger_error + split_error
    if math.isnan(vi):
        return 10, 5, 5
    return merger_error, split_error, vi


def get_readable_eval_result(
        segmentation_metrics_list: Sequence[pd.Series],
        columns: Sequence[str]
) -> pd.DataFrame:
    d = {}
    for c, segmentation_metrics in zip(columns, segmentation_metrics_list):
        seg_metrics = segmentation_metrics.copy()
        for k in seg_metrics.index:
            if type(seg_metrics[k]) == np.ndarray:
                seg_metrics[k] = [round(value * 100, 2) for value in seg_metrics[k]]
            else:
                seg_metrics[k] = round(seg_metrics[k] * 100, 2)
        d[c] = seg_metrics
    return pd.DataFrame(d)


def get_model_type(model_name: str):
    model_dict = {
        'my_unet': my_models.my_unet,
        'my_pspnet': my_models.my_pspnet,
        'my_segnet': my_models.my_segnet,
        'my_ccnet': my_models.my_ccnet,
        'my_spnet': my_models.my_spnet,
        'unet': my_models.unet,
        'my_msdrnet': my_models.my_msdrnet,
    }
    if type(model_name) != str or model_name not in model_dict.keys():
        raise ValueError('get_model(): invalid model_name!')
    return model_dict[model_name]


def poly_learning_rate(initial_learning_rate, current_iteration, total_iteration, power=0.9):
    initial_learning_rate *= (1 - float(current_iteration) / total_iteration) ** power
    return initial_learning_rate


def get_instance_list(mask, kernel_size=5):
    '''
    Get each instance and the dilated instance from the mask
    '''
    instance_list = []
    dilated_instance_list = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    for i in range(1, mask.max() + 1):
        instance_i = np.zeros_like(mask)
        instance_i[mask == i] = 1
        instance_list.append(instance_i)
        dilated_instance_i = cv2.dilate(instance_i, kernel=kernel, iterations=1)
        dilated_instance_list.append(dilated_instance_i)
    return instance_list, dilated_instance_list


def get_gap_map(dilated_instance_list, mask):
    num_instances = len(dilated_instance_list)
    gap_map = np.zeros_like(mask)
    for i in range(num_instances):
        for j in range(i + 1, num_instances):
            gap_map = np.logical_or(gap_map, np.logical_and(dilated_instance_list[i], dilated_instance_list[j]))
    return gap_map


def clean_gap_map(gap_map, mask):
    '''
    Eliminate unimportant parts in gap_map, these parts do not cover the object,
    so it means that the distance between the objects is large, we do not pay attention to these parts
    '''
    gap_instance_map = skimage.measure.label(gap_map)
    for i in range(1, gap_instance_map.max() + 1):
        gap_i = np.zeros_like(gap_instance_map)
        gap_i[gap_instance_map == i] = 1
        num_overlap_pixel = np.logical_and(mask, gap_i).sum()
        if num_overlap_pixel == 0:
            gap_instance_map[gap_instance_map == i] = 0
    return instance_mask_to_binary_label(gap_instance_map)


def get_inferer(use_sliding_window=False):
    return SlidingWindowInferer(roi_size=(256, 256), sw_batch_size=16) \
        if use_sliding_window else SimpleInferer()


def get_adjusted_feature_map_s2b(small_feature_map, big_feature_map):
    '''
    Adjust the size of small_feature_map to be consistent with big_feature_map
    assumption：small_feature_map.size[2:] <= big_feature_map.size[2:]
    '''
    if len(big_feature_map.size()) != 4 or len(small_feature_map.size()) != 4:
        raise ValueError('get_adjusted_feature_map_s2b(): invalid parameters!')
    big_height, big_width = big_feature_map.size()[2:]
    small_height, small_width = small_feature_map.size()[2:]
    if small_height > big_height or small_width > big_width:
        raise ValueError('get_adjusted_feature_map_s2b(): invalid parameters!')
    right_pad = int(small_width != big_width)
    down_pad = int(small_height != big_height)
    small_feature_map = F.pad(
        small_feature_map, (0, right_pad, 0, down_pad), "replicate"
    )
    return small_feature_map


def get_adjusted_feature_map_b2s(big_feature_map, small_feature_map):
    '''
    Adjust the size of big_feature_map to be consistent with small_feature_map
    assumption：small_feature_map.size[2:] <= big_feature_map.size[2:]
    '''
    if len(small_feature_map.size()) != 4 or len(big_feature_map.size()) != 4:
        raise ValueError('get_adjusted_feature_map_b2s(): invalid parameters!')
    small_height, small_width = small_feature_map.size()[2:]
    big_height, big_width = big_feature_map.size()[2:]
    if big_width < small_width or big_height < small_height:
        raise ValueError('get_adjusted_feature_map_b2s(): invalid parameters!')
    return big_feature_map[:, :, :small_height, :small_width]


def get_adjusted_feature_map_channels(big_feature_map, small_feature_map):
    '''
    Adjust the size of big_feature_map to be consistent with small_feature_map
    assumption：small_feature_map.size[2:] <= big_feature_map.size[2:]
    '''
    if len(small_feature_map.size()) != 4 or len(big_feature_map.size()) != 4:
        raise ValueError('get_adjusted_feature_map_b2s(): invalid parameters!')
    small_height = small_feature_map.size()[1]
    big_height = big_feature_map.size()[1]
    if big_height < small_height:
        raise ValueError('get_adjusted_feature_map_b2s(): invalid parameters!')
    return big_feature_map[:, :small_height, :, :]


def get_adjusted_feature_map_small_big(small_feature_map, big_feature_map):
    '''
     Adjust the size of small_feature_map to be consistent with big_feature_map
     assumption：small_feature_map.size[2:] <= big_feature_map.size[2:]
     '''
    if len(big_feature_map.size()) != 4 or len(small_feature_map.size()) != 4:
        raise ValueError('get_adjusted_feature_map_s2b(): invalid parameters!')
    big_height, big_width = big_feature_map.size()[2:]
    small_height, small_width = small_feature_map.size()[2:]
    if small_height > big_height or small_width > big_width:
        raise ValueError('get_adjusted_feature_map_s2b(): invalid parameters!')
    right_pad = big_width - small_width
    down_pad = big_height - small_height
    small_feature_map = F.pad(
        small_feature_map, (0, right_pad, 0, down_pad), "replicate"
    )
    return small_feature_map


def get_adjusted_feature_map_b2s_pre(big_feature_map, small_feature_map):
    '''
    Adjust the size of big_feature_map to be consistent with small_feature_map
    assumption：small_feature_map.size[2:] <= big_feature_map.size[2:]
    '''
    if len(small_feature_map.size()) != 4 or len(big_feature_map.size()) != 4:
        raise ValueError('get_adjusted_feature_map_b2s(): invalid parameters!')
    small_height, small_width = 256, 256
    return big_feature_map[:, :, :small_height, :small_width]
