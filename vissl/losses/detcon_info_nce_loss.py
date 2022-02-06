# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import pprint

import numpy as np
import torch
from classy_vision.generic.distributed_util import get_cuda_device_index, get_world_size,get_rank
from classy_vision.losses import ClassyLoss, register_loss
from torch import nn
from vissl.config import AttrDict
from vissl.utils.distributed_utils import gather_from_all


@register_loss("detcon_info_nce_loss")
class DetconInfoNCELoss(ClassyLoss):

    def __init__(self, loss_config: AttrDict, device: str = "gpu"):
        super(DetconInfoNCELoss, self).__init__()

        self.loss_config = loss_config
        # loss constants
        self.temperature = self.loss_config.temperature
        self.buffer_params = self.loss_config.buffer_params
        self.info_criterion = DetconInfoNCECriterion(
            self.buffer_params, self.temperature
        )

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        return cls(loss_config)

    def forward(self, output,mask_ids, target):
        loss = self.info_criterion(output,mask_ids)
        return loss

    def __repr__(self):
        repr_dict = {"name": self._get_name(), "info_average": self.info_criterion}
        return pprint.pformat(repr_dict, indent=2)


class DetconInfoNCECriterion(nn.Module):

    def __init__(self, buffer_params, temperature: float):
        super(DetconInfoNCECriterion, self).__init__()
        self.use_gpu = get_cuda_device_index() > -1
        self.temperature = temperature
        self.max_val = 1e9
        self.num_rois = 16
        self.buffer_params = buffer_params
        self.dist_rank = get_rank()
        logging.info(f"Creating Detcon Info-NCE loss")
        
    def make_same_obj(self,ind_0, ind_1):
        b = ind_0.shape[0]
        same_obj = torch.eq(ind_0.reshape([b, self.num_rois, 1]),
                             ind_1.reshape([b, 1, self.num_rois]))
        return same_obj.float().unsqueeze(2)
    
    def manual_cross_entropy(self,labels, logits, weight):
        ce = - weight * torch.sum(labels * torch.nn.functional.softmax(logits,dim = 1), dim=-1)
        return torch.mean(ce)

    def forward(self, embedding: torch.Tensor, mask_ids: torch.Tensor):
        batch_size = int(len(embedding)/2)
        #import ipdb;ipdb.set_trace()
        
        embeddings_buffer,mask_ids_buffer = self.gather_embeddings_ids(embedding,mask_ids)
        target1 = embeddings_buffer[:batch_size]
        target2 = embeddings_buffer[batch_size:]
        tind1 = mask_ids_buffer[:batch_size]
        tind2 = mask_ids_buffer[batch_size:]

        same_obj_aa = self.make_same_obj(tind1, tind1).to('cuda')
        same_obj_ab = self.make_same_obj(tind1, tind2).to('cuda')
        same_obj_ba = self.make_same_obj(tind2, tind1).to('cuda')
        same_obj_bb = self.make_same_obj(tind2, tind2).to('cuda')

        target1 = torch.nn.functional.normalize(target1)
        target2 = torch.nn.functional.normalize(target2)

        labels_local = torch.nn.functional.one_hot(torch.tensor(np.arange(batch_size))
                                                   ,batch_size).unsqueeze(1).unsqueeze(3).to('cuda')
        labels_ext = torch.nn.functional.one_hot(torch.tensor(np.arange(batch_size)),
                                                 batch_size * 2).unsqueeze(1).unsqueeze(3).to('cuda')

        logits_aa = torch.einsum("abk,uvk->abuv", target1, target1) / self.temperature
        logits_bb = torch.einsum("abk,uvk->abuv", target2, target2) / self.temperature
        logits_ab = torch.einsum("abk,uvk->abuv", target1, target2) / self.temperature
        logits_ba = torch.einsum("abk,uvk->abuv", target2, target1) / self.temperature

        labels_aa = labels_local * same_obj_aa
        labels_ab = labels_local * same_obj_ab
        labels_ba = labels_local * same_obj_ba
        labels_bb = labels_local * same_obj_bb

        logits_aa = logits_aa - self.max_val * labels_local * same_obj_aa
        logits_bb = logits_bb - self.max_val * labels_local * same_obj_bb
        labels_aa = 0. * labels_aa
        labels_bb = 0. * labels_bb

        labels_abaa = torch.cat([labels_ab, labels_aa], axis=2)
        labels_babb = torch.cat([labels_ba, labels_bb], axis=2)

        labels_0 = torch.reshape(labels_abaa, [batch_size, self.num_rois, -1])
        labels_1 = torch.reshape(labels_babb, [batch_size, self.num_rois, -1])

        num_positives_0 = torch.sum(labels_0, axis=-1, keepdims=True)
        num_positives_1 = torch.sum(labels_1, axis=-1, keepdims=True)

        labels_0 = labels_0 / torch.max(num_positives_0, torch.ones(num_positives_0.shape).to('cuda'))
        labels_1 = labels_1 / torch.max(num_positives_1, torch.ones(num_positives_1.shape).to('cuda'))

        obj_area_0 = torch.sum(self.make_same_obj(tind1, tind1), axis=[2, 3])
        obj_area_1 = torch.sum(self.make_same_obj(tind2, tind2), axis=[2, 3])

        weights_0 = torch.greater(num_positives_0[..., 0], 1e-3).float()
        weights_0 = weights_0 / obj_area_0
        weights_1 = torch.greater(num_positives_1[..., 0], 1e-3).float()
        weights_1 = weights_1 / obj_area_1

        logits_abaa = torch.cat([logits_ab, logits_aa], axis=2)
        logits_babb = torch.cat([logits_ba, logits_bb], axis=2)

        logits_abaa = torch.reshape(logits_abaa, [batch_size, self.num_rois, -1])
        logits_babb = torch.reshape(logits_babb, [batch_size, self.num_rois, -1])

        loss_a = self.manual_cross_entropy(labels_0, logits_abaa, weights_0)
        loss_b = self.manual_cross_entropy(labels_1, logits_babb, weights_1)
        loss = loss_a + loss_b
        
        return loss

    @staticmethod
    def gather_embeddings_ids(embedding: torch.Tensor,mask_ids: torch.Tensor):
        """
        Do a gather over all embeddings, so we can compute the loss.
        Final shape is like: (batch_size * num_gpus) x embedding_dim
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            embedding_gathered = gather_from_all(embedding)
            mask_ids_gathered = gather_from_all(mask_ids)
        else:
            embedding_gathered = embedding
            mask_ids_gathered = mask_ids
        return embedding_gathered, mask_ids_gathered
