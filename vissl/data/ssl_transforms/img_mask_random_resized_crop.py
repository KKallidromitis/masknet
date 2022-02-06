# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict

import torch
import torchvision
import torchvision.transforms.functional as TF
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform


@register_transform("MaskRandomResizedCrop")
class MaskRandomResizedCrop(ClassyTransform):
    """
    Apply random crop and resize to a PIL Image and Mask.
    """
    def __init__(self, size=224):
        super().__init__()
        self.size = size
        self.totensor = torchvision.transforms.ToTensor()
        self.topil = torchvision.transforms.ToPILImage()
        
    def __call__(self, image,mask):
        """
        Args:
            image (PIL Image or Tensor): Image to be cropped and resized.
            mask (Tensor): Mask to be cropped and resized.
        Returns:
            PIL Image or Tensor: Randomly cropped/resized image.
            Mask Tensor: Randomly cropped/resized mask.
        """
        
        #import ipdb;ipdb.set_trace()
        i, j, h, w = torchvision.transforms.RandomResizedCrop.get_params(image,scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0))
        image = TF.resize(TF.crop(image, i, j, h, w),(self.size,self.size),interpolation=3)#InterpolationMode.BICUBIC
        image = self.topil(torch.clip(self.totensor(image),min=0, max=255))
        
        mask = TF.resize(TF.crop(mask, i, j, h, w),(self.size,self.size),interpolation=0) #InterpolationMode.NEAREST  
        return image,mask

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MaskRandomResizedCrop":
        """
        Instantiates MaskRandomResizedCrop from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            MaskRandomResizedCrop instance.
        """
        
        size = config.get("size", 224)
        logging.info(f"MaskRandomResizedCrop | Using size: {size}")
        return cls(size=size)