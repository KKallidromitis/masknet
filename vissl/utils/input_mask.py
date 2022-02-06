import numpy as np
import torch
import skimage
import pickle
import torchvision

def create_patch_mask(image,segments=[3,2]):
    totensor = torchvision.transforms.ToTensor()
    """
    Input is a PIL Image or Tensor with [CxWxH]
    """
    try:
        image = torchvision.transforms.ToTensor()(image)
    except:
        pass
    dims=list(np.floor_divide(image.shape[1:],segments))
    
    mask=torch.hstack([torch.cat([torch.zeros(dims[0],dims[1])+i+(j*(segments[0])) 
                                  for i in range(segments[0])]) for j in range(segments[1])])
    
    mods = list(np.mod(image.shape[1:],segments))
    if mods[0]!=0:
        mask = torch.cat([mask,torch.stack([mask[-1,:] for i in range(mods[0])])])
    if mods[1]!=0:
        mask = torch.hstack([mask,torch.stack([mask[:,-1] for i in range(mods[1])]).T])
        
    return mask.int()

def create_fh_mask(image, scale=1000, min_size=1000):
    mask = skimage.segmentation.felzenszwalb(image, scale=scale, min_size=min_size)
    #mask = mask.astype(np.dtype('<u1'))
    return torch.tensor(mask).int()

def convert_binary_mask(mask,max_mask_id=256,pool_size=7):
    batch_size = mask.shape[0]
    mask_ids = torch.arange(max_mask_id).reshape(1,max_mask_id, 1, 1).float()
    binary_mask = torch.eq(mask_ids, mask).float()
    binary_mask = torch.nn.AdaptiveAvgPool2d((pool_size,pool_size))(binary_mask)
    binary_mask = torch.reshape(binary_mask,(batch_size,max_mask_id,pool_size*pool_size)).permute(0,2,1)
    binary_mask = torch.argmax(binary_mask, axis=-1)
    binary_mask = torch.eye(max_mask_id)[binary_mask]
    binary_mask = binary_mask.permute(0, 2, 1)
    return binary_mask

def sample_masks(binary_mask,n_masks=16):
    batch_size=binary_mask.shape[0]
    mask_exists = torch.greater(binary_mask.sum(-1), 1e-3)
    sel_masks = mask_exists.float() + 0.00000000001
    sel_masks = sel_masks / sel_masks.sum(1, keepdims=True)
    sel_masks = torch.log(sel_masks)
    
    dist = torch.distributions.categorical.Categorical(logits=sel_masks)
    mask_ids = dist.sample([n_masks]).T
    
    sample_mask = torch.stack([binary_mask[b][mask_ids[b]] for b in range(batch_size)])
    
    return sample_mask,mask_ids

def create_maskedfeats(masks,feats):
    bs, emb, emb_x, emb_y  = feats.shape
    masks_area = masks.sum(axis=-1, keepdims=True)
    smpl_masks = masks / torch.clamp(masks_area, 1)

    embedding_local = torch.reshape(feats,[bs, emb_x*emb_y, emb])
    smpl_embedding = torch.matmul(smpl_masks.float(), embedding_local)
    return [smpl_embedding]