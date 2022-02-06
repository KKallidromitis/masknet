from joblib import Parallel, delayed
from skimage.segmentation import felzenszwalb
import numpy as np
import pickle
import torchvision
import torch
import os
import time
import logging
from torchvision import datasets

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    
    https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """

    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
    
    
class Preload_Masks():
    def __init__(self,dataset_dir,output_dir,mask_type=['fh','patch'],batch_num=100,experiment_name='',
                 num_threads=os.cpu_count(),scale=1000,min_size=1000,segments=[3,3]):
        
        self.output_dir=output_dir
        self.mask_type=mask_type
        self.scale = scale
        self.min_size = min_size
        self.segments = segments
        self.experiment_name = experiment_name
        self.num_threads = num_threads
        self.batch_num = batch_num
        self.totensor = torchvision.transforms.ToTensor()
        self.image_dataset = ImageFolderWithPaths(dataset_dir,transform=self.totensor)
        self.ds_length = len(self.image_dataset)
        self.size = len(self.image_dataset)//self.batch_num
        self.save_path = os.path.join(self.output_dir,self.experiment_name)
        
    def create_patch_mask(self,image,segments):
        dims=list(np.floor_divide(image.shape[1:],segments))

        mask=torch.hstack([torch.cat([torch.zeros(dims[0],dims[1])+i+(j*(segments[0])) 
                                      for i in range(segments[0])]) for j in range(segments[1])])

        mods = list(np.mod(image.shape[1:],segments))
        if mods[0]!=0:
            mask = torch.cat([mask,torch.stack([mask[-1,:] for i in range(mods[0])])])
        if mods[1]!=0:
            mask = torch.hstack([mask,torch.stack([mask[:,-1] for i in range(mods[1])]).T])

        return mask.int()

    def create_fh_mask(self,image, scale, min_size):
        mask = felzenszwalb(image.permute(1,2,0), scale=scale, min_size=min_size)
        return torch.tensor(mask).int()

    def select_mask(self,obj):
        image,label,path = obj
        mask_fh,mask_patch = [],[]
        
        if 'fh' in self.mask_type:
            mask_fh = self.create_fh_mask(image, scale=self.scale, min_size=self.min_size)
        if 'patch' in self.mask_type:  
            mask_patch = self.create_patch_mask(image,segments=self.segments)
            
        return [mask_fh,mask_patch,path]
    
    def pkl_save(self,file,name):
        with open(name, 'wb') as handle:
            pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def save_masks(self,mask_fh,mask_patch,base_name):

        if 'fh' in self.mask_type:
            self.pkl_save(mask_fh,base_name+'_fh.pkl')

        if 'patch' in self.mask_type:
            self.pkl_save(mask_patch,base_name+'_patch.pkl')
        return

    def save_dicts(self,fh_dict,patch_dict):

        if 'fh' in self.mask_type:
            self.pkl_save(fh_dict,self.save_path+'/img_to_fh.pkl')

        if 'patch' in self.mask_type:
            self.pkl_save(patch_dict,self.save_path+'/img_to_patch.pkl')
        return
    
    def save_dir(self,fh_dir,patch_dir):

        if 'fh' in self.mask_type:
            self.pkl_save(fh_dir,self.save_path+'/fh_dir.pkl')

        if 'patch' in self.mask_type:
            self.pkl_save(patch_dir,self.save_path+'/patch_dir.pkl')
        return
    
    def forward(self):
        print('Dataset Length: %d  Size of Batches: %d'%(self.ds_length,self.size))
        img_paths,fh_paths,patch_paths,idx = [],[],[],[]
        fh_dir,patch_dir = [],[]
        try:
            os.mkdir(os.path.join(output_dir,experiment_name))
        except:
            if not os.path.exists(self.output_dir):
                os.makedirs(os.path.join(self.output_dir,self.experiment_name))

        for batch in range(self.batch_num):
            start = time.time()
            ds_range = range(batch*self.size,(batch+1)*self.size)
            if batch==self.batch_num-1:
                ds_range = range(batch*self.size,self.ds_length)

            mask_fh,mask_patch,path = zip(*Parallel(n_jobs=self.num_threads,prefer="threads")
                                       (delayed(self.select_mask)(self.image_dataset[i]) for i in ds_range))

            end = time.time()
            print(f"Finished processing batch %d / %d, Time taken %f min"%(batch+1,self.batch_num,(end - start)/60))

            base_name = os.path.join(self.output_dir,self.experiment_name,str(batch))
            self.save_masks(mask_fh,mask_patch,base_name)

            fh_dir.append(base_name+'_fh.pkl')
            patch_dir.append(base_name+'_patch.pkl')
            
            img_paths.extend(path)
            fh_paths.extend([batch for p in range(len(path))])
            patch_paths.extend([batch for p in range(len(path))])
            idx.extend(list(range(len(path))))

        fh_dict,patch_dict = {},{}

        for i in range(self.ds_length):
            fh_dict[img_paths[i]] = [fh_paths[i],idx[i]]
            patch_dict[img_paths[i]] = [patch_paths[i],idx[i]]

        self.save_dicts(fh_dict,patch_dict)
        self.save_dir(fh_dir,patch_dir)
        return 

if __name__=="__main__":
    
    mask_loader = Preload_Masks(dataset_dir = '/home/kkallidromitis/masknet/data/sample/images/',
                                output_dir = '/home/kkallidromitis/masknet/data/sample/masks/',
                                mask_type = ['fh','patch'],
                                batch_num=12,
                                experiment_name = 'train',
                               )

    mask_loader.forward()