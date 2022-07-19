import numpy as np
import cv2
from model import MRIModel
import torch


def cancer(path):
    if np.max(cv2.imread(path)) > 0:
        return 1
    else:
        return 0


def get_model(model_name, backbone, in_channels, num_classes):
    model = None
    if 'FPN' in model_name or 'Unet' in model_name or 'LinkNet' in model_name or 'UnetPlusPlus' in model_name:
        model = MRIModel(model_name, backbone, in_channels=in_channels, out_classes=num_classes)
    else:
        raise ValueError(f'Undefined model name: {model_name}')
    return model


def save_model(trainer, filename):
    trainer.save_checkpoint(filename)


def dataset_dict_from_df(df, dataset_dict):
    for i in range(len(df['mask'])):
        mask_img = np.array(cv2.imread(df['mask'][i], 0) / 255.)
        mask_img = np.expand_dims(mask_img, axis=0)
        dataset_dict[i] = dict(image=cv2.imread(df['mask'][i].replace('_mask', '')), mask=mask_img)
    return dataset_dict
