# <<< import external stuff <<<
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, utils

import os

import numpy as np

import PIL
from PIL import Image

import time
# --- import external stuff ---

class TabulatedSeries(torch.utils.data.Dataset):
    '''
    Create a dataset. Data are indexed in a text file containing images paths.
    init arguments are as follows:
    - table_path        ->  path referencing to the text file containing examples' paths
    - transform         ->  transforms to be applied to images
    - rotation          ->  toggles continuous rotation of the image
    - reflection        ->  toggles reflection of the images
    - rotation_90       ->  toggles 90° rotations of the input
    '''
    
    def __init__(
        self,
        table_path      : list[str],
        transform                       = None,
        rotation        : bool          = True,
        reflection      : bool          = True,
        rotation_90     : bool          = False,
        cropkey         : bool          = True,
        crop_lim        : tuple[float]  = (0.25,0.75),
        bootstrap_loader: bool          = False,
        twin_image      : bool          = False,
        full_loading    : bool          = False
        ):
        
        super(TabulatedSeries, self).__init__()
        
        self.table_path     = table_path
        self.transform      = transform
        
        self.rotation       = rotation
        self.rotation_90    = rotation_90
        self.reflection     = reflection
        
        self.cropkey        = cropkey
        self.crop_lim       = crop_lim
        
        self.bootstrap_loader = bootstrap_loader
        
        self.twin_image     = twin_image
        self.full_loading   = full_loading
        
        with open(self.table_path,'r') as table_file:
            self.table  = table_file.readlines()
            self.length = len(self.table)

        if self.full_loading:
            print('Loading full dataset (one-time overhead)...')
            start_clock = time.time()
            self.full_dataset = []
            for line in self.table:
                paths = line.split()
                self.full_dataset.append(
                    [self.transform( import_image(Image.open(paths[0])) ).unsqueeze(0), 
                     self.transform( import_image(Image.open(paths[1])) ).unsqueeze(0) ]
                )
            print(f'time required for full dataset retrieval is {time.time()-start_clock}')
            print('DONE!')

            
        if self.bootstrap_loader:
            # In this case we need to modify the table so that we have a bootstrap dataset
            #table_old = self.table
            #self.table = []
            #for ii in range(self.length):
                #index = torch.randint( 0, self.length, (1,) )
                #self.table.append( table_old[index] )
            
            table_old   = self.table
            self.table  = []
            id_set      = set()
            for line in table_old:
                id_set.add( line.split()[-1] )
            id_list = list(id_set)
            del id_set # <- free memory
            for ii in range( len(id_list) ):
                index = torch.randint( 0, len(id_list), (1,) )
                lines2append = [
                    line for line in table_old if line.split()[-1]==id_list[index]
                    ]
                for line in lines2append:
                    self.table.append( line )
            print(f'Table length is {len(self.table)}; __len__ is {self.length}')
            self.length = len(self.table)
            #print(len(table_old))
                
        
        
    def __len__(self): return self.length

    
    def pick_image(self, idx):
        
        if not self.full_loading:
            table_line = self.table[idx]
            paths = table_line.split()
            #paths = table_line.split()[:-1] # <- the [:-1] is to remove id

            out_list = []
            for path in paths:
                image = import_image(Image.open(path))
                out_list.append(self.transform(image).unsqueeze(0))

        else: # we do not need to retrieve data into memory
            out_list = self.full_dataset[idx]
            
        if self.rotation:
            rotation_angle = 360*torch.rand(1).item()
            for ii in range(len(out_list)):
                out_list[ii] = out_list[ii].rotate(rotation_angle, PIL.Image.NEAREST, fillcolor=(0,0,0))
        elif self.rotation_90:
            coin = torch.rand(1).item()
            if 0.0 <= coin < 0.25:
                rotation_key = 1#PIL.Image.ROTATE_90
            elif 0.25 <= coin < 0.5:
                rotation_key = 2#PIL.Image.ROTATE_180
            elif 0.5 <= coin < 0.75:
                rotation_key = 3#PIL.Image.ROTATE_270
            if coin < 0.75:
                for ii in range(len(out_list)):
                    out_list[ii] = torch.rot90(out_list[ii], k=rotation_key, dims=(-1,-2))#.transpose(rotation_key)
                    
                    
        if self.reflection:
            hor_flip = torch.rand(1) >= 0.5
            ver_flip = torch.rand(1) >= 0.5
            for ii in range(len(out_list)):
                if hor_flip:
                    out_list[ii] = torch.flip(out_list[ii], (-1,))#.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                if ver_flip:
                    out_list[ii] = torch.flip(out_list[ii], (-2,))#.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        
        #if self.transform:
        #    for ii in range(len(out_list)):
        #        out_list[ii] = self.transform(out_list[ii])
                
        #for ii in range(len(out_list)):
        #    out_list[ii] = out_list[ii].unsqueeze(0)
            
            
        out_tensor = torch.cat(out_list, dim=0)
                
        return out_tensor
    
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        image1 = self.pick_image(idx)
        
        if self.twin_image:
            
            good_image = False
            
            while not good_image:
                                
                image2 = self.pick_image( torch.randint(self.length,(1,)).item() )
                
                vertical = torch.randint(image1.shape[-1], (1,)).item()
                horizontal = torch.randint(image1.shape[-2],(1,)).item()
                
                image2 = torch.roll( image2, (vertical, horizontal), dims=(-1,-2) )
                
                image = image1 + image2
                
                if torch.max(image)-1 <= 1e-3:
                    good_image = True
        
        else:
            image = image1
        
        
        return image
        
    
def give_dataloaders(args):
    '''
    This function returns train, validation or testing dataloaders, depending on what variable is present in args. Return type is a dictionary.
    '''
    
    num_workers = args.nproc
    
    set_names = ['train_set', 'valid_set', 'test_set']
    
    has_sets = False
    
    for set_name in set_names:
        if hasattr(args, set_name):
            has_sets = True
        
    if not has_sets:
        raise RuntimeError('No dataset was detected in args. Check arguments are parsed correctly.')
    
    transform = transforms.Compose(
            [
                transforms.Grayscale( num_output_channels=1 ),
                transforms.Resize( args.size ),
                transforms.ToTensor(),
            ]
        )
            
    dataloaders = {}
    
    master_path = args.paths['master']
            
    for set_name in set_names:
        
        if hasattr(args, set_name):
            
            set_path = getattr(args, set_name)
            
            bootstrap = args.bootstrap if hasattr(args, 'bootstrap') else False
            twin_image  = args.twin_image if hasattr(args, 'twin_image') else False
            
            dataset = TabulatedSeries(
                table_path          = set_path,
                transform           = transform,
                rotation            = args.rotation,
                rotation_90         = args.rotation90,
                cropkey             = args.crop,
                crop_lim            = args.croplims,
                bootstrap_loader    = bootstrap,
                twin_image          = twin_image,
                full_loading        = args.full_loading
                )
            
            if bootstrap:
                with open(f'{master_path}/{set_name}_bootstrap.txt', 'w+') as bootstrap_file:
                    for line in dataset.table:
                        bootstrap_file.write(line)
            else:
                with open(f'{master_path}/{set_name}.txt', 'w+') as check_set_file:
                    for line in dataset.table:
                        check_set_file.write(line)
                        
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size      = args.batch,
                shuffle         = False if set_name == 'test_set' else True, # <- in the case of testing and validation, we want a fixed order for data
                num_workers     = num_workers,
                pin_memory      = True
                )
            
            dataloaders[set_name] = dataloader
    
    return dataloaders


def import_image(im, min_size=1, fill_color=(0, 0, 0, 0)):
    '''
    Utility function. Takes care of image importing operations. Also, pads the image if it is not square
    '''
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im


