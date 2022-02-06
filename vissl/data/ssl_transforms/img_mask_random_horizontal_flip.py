# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict

import torch
import torchvision.transforms.functional as TF
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform


@register_transform("MaskRandomHorizontalFlip")
class MaskRandomHorizontalFlip(ClassyTransform):
    """
    Apply horizontal flip to a PIL Image and Mask.
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, image,mask):
        """
        Args:
            image (PIL Image or Tensor): Image to be flipped.
            mask (Tensor): Mask to be flipped.
        Returns:
            PIL Image or Tensor: Randomly flipped image.
            Mask Tensor: Randomly flipped mask.
        """
        
        if torch.rand(1) < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            return image,mask
        return image,mask

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MaskRandomHorizontalFlip":
        """
        Instantiates MaskRandomHorizontalFlip from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            MaskRandomHorizontalFlip instance.
        """
        
        p = config.get("p", 0.5)
        logging.info(f"MaskRandomHorizontalFlip | Using p: {p}")
        return cls(p=p)
