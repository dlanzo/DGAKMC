# <<< importing external stuff <<<
import os
import sys

import torch
import torch.nn as nn
from torchvision import utils, datasets, transforms

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import PIL
from PIL import Image

from warnings import warn

from multiprocessing import Process
# --- importing external stuff ---

# <<< import numba <<<
try:
    from numba import njit, prange
except ImportError:
    print('It seems that "numba" is not installed on this machine or there are problems in importing it. Falling back on non-jitted versions of scripts. Some operations will be slower. Consider installing it (e.g. by "pip install numba").')
    
    def njit(fun): # <- alternative definition of njit
        return fun
# --- import numba ---

def build_train_logs_dir_tree(args, mkdir=True):
    '''
    This function builds the output folder structure and adds appropriate paths to dictionary in args
    '''
    
    master = f'train_logs/{args.id}' # <- name of master folder
    
    if os.path.isdir(master) and mkdir:
        # in this case we have a naming conflict: id will be replaced with id_N with N=number of folders with the same id
        print(f'Naming conflict found with id "{args.id}".')
        num_existing_folders = len([subfolder for subfolder in os.listdir('train_logs') if args.id in subfolder])
        args.id = f'{args.id}_{num_existing_folders}'
        
        master = f'train_logs/{args.id}'
        print(f'Naming conflict handled. Training logs will be saved in {master}')
    
    if mkdir:
        os.mkdir(master)
    
    model_path = f'{master}/model'
    
    if args.graphics:
        lossplot_path = f'{master}/lossplot.png'
        if args.gifs:
            gifs_path = f'{master}/gifs'
        else:
            gifs_path = None
    else:
        lossplot_path = None
        gifs_path = None
        
    trainloss_path = f'{master}/train_loss.txt'
    validloss_path = f'{master}/valid_loss.txt'
    
    args.paths = {
        'master'    :   master,
        'model'     :   model_path,
        'gif'       :   gifs_path,
        'lossplot'  :   lossplot_path,
        'trainloss' :   trainloss_path,
        'validloss' :   validloss_path
        }
    
    for name in ['model', 'gif']:
        if args.paths[name] is not None and mkdir:
            os.mkdir(args.paths[name])
    
    return args


def build_test_dir_tree(args):
    '''
    This function builds the output folder for testing procedure, given the id in args
    '''
    master = f'test_outputs/{args.id}' # <- name of the master folder

    if os.path.isdir(master):
            
        print(f'Naming conflict found with id "{args.id}".')
        
        num_existing_folders = len([subfolder for subfolder in os.listdir('test_outputs') if args.id in subfolder])
        
        args.id = f'{args.id}_{num_existing_folders}'
        master = f'test_outputs/{args.id}'
        
        print(f'Naming conflict handled. Testing outputs will be saved in {master}')
            
    os.mkdir(master)
    
    if args.graphics:
        png_path    = f'{master}/png'
        if args.gifs:
            gifs_path = f'{master}/gifs'
        else:
            gifs_path = None
    
    else:
        warn('Graphics was disabled for testing procedure. Little information for many calculations will be produced.')
        png_path    = None
        gifs_path   = None
        
    area_path       = f'{master}/area'
    progloss_path   = f'{master}/progloss'
    
    AR_path     = f'{master}/AR' if args.AR else None
    
    args.paths = {
        'master'    :   master,
        'gif'       :   gifs_path,
        'area'      :   area_path,
        'progloss'  :   progloss_path,
        'AR'        :   AR_path
        }
    
    for name in ['gif','area','progloss','AR']:
        if args.paths[name] is not None:
            os.mkdir(args.paths[name])
    
    return args


def build_predict_dir_tree(args):
    '''
    This function builds the directory structure for the output of a prediction run. It returns paths in addition to args.paths
    '''
    master  = f'out/{args.id}'
    if os.path.isdir(master):
        print(f'Naming conflict found with id "{args.id}".')
        num_existing_folders = len([subfolder for subfolder in os.listdir('out') if args.id in subfolder])
        args.id = f'{args.id}_{num_existing_folders}'
        
        master = f'out/{args.id}'
        print(f'Naming conflict handled. Prediction outputs will be saved in {master}')
        
    os.mkdir(master)
    
    AR_path         = f'{master}/AR'   if args.AR                else None
    gif_path        = f'{master}/gifs' if args.gifs and args.graphics   else None
    
    phi_0_path  = f'{master}/initial_condition.png'
    area_path   = f'{master}/area'
    
    init_geo_source_path = \
        f'{master}/init_geo.py' if args.gengeo else None
    
    args.paths = {
        'AR'                :   AR_path,
        'phi_0'             :   phi_0_path,
        'geo_source'        :   init_geo_source_path,
        'gifs'              :   gif_path,
        'area'              :   area_path
        }
    
    for name in ['png', 'gifs']:
        if args.paths[name] is not None:
            os.mkdir(args.paths[name])
        
    return args


def save_args(args):
    '''
    This function saves the argument list in the log master folder
    '''
    with open(f'{args.paths["master"]}/args.txt', 'w+') as args_file:
        for arg in dir(args):
            if arg[0] != '_':
                args_file.write(f'{arg} \t : \t {vars(args)[arg]} \n')
                

def print_model_info(model):
    '''
    This function prints some infos about the model
    '''
    if not hasattr(model, 'model_list'):
        # model is a simple model: just print the number of parameters
        params_nums = sum( [p.numel() for p in model.parameters()] )
        print()
        print('<<< model infos <<<')
        print(f'The number of parameters in the model is: {params_nums}')
        print('--- model infos ---')
        print()
    elif hasattr(model, 'model_list'):
        # model is a committee model; print some additional infos
        params_nums = sum( [p.numel() for p in model.model_list[0].parameters()] )
        num_models  = len(model)
        print()
        print('<<< model infos <<<')
        print(f'model is a CommitteeModel combining inferences from {num_models} models.')
        print(f'Each model has a number of parameters: {params_nums}, totalling {num_models*params_nums} parameters')
        print()
    else:
        raise RuntimeError('It seems that model is neither a torch.nn.Module nor a CommitteeModel.')
    
    
def import_model(model, path, mode='dictionary'):
    '''
    This function loads a .pt file model in provided model.
    '''
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'No model found at path "{path}".'
            )
    
    print('<<< Loading model ... <<<')
    if mode == 'dictionary':
        model.load_state_dict(
            torch.load(
                path,
                map_location = 'cpu'
                )
            )
    elif mode == 'serialized':
        model = torch.load(
            path,
            map_location = 'cpu'
            )
    else:
        raise (f"--- Load mode {mode} is not valid. ---")
    print('--- Loading done! ---')
    
    return model
    
    
def log_epoch_start_info(epoch, paths):
    '''
    This function prints some infos at the begining of the epoch
    '''
    print()
    print(f'<<< Epoch {epoch} starting... <<<')
    print(f'master folder for outputs is {paths["master"]}')
    print()
    

def make_lossplot(gen_losses, discr_losses, GAN_method, paths):
    '''
    This function outputs the training/validation loss as a function of the number of epochs
    '''
    matplotlib.use('Agg')
    
    if GAN_method == 'WGAN':
        print('Outputting W distance loss plot...', end='')
    else:
        print('Outputting G&D losses...', end='')
        plt.plot(
            np.arange(len(gen_losses)),
            np.array(gen_losses), label='G loss'
            )
        if GAN_method == 'SGAN':
            plt.plot(
                np.arange(len(gen_losses)),
                0*np.array(gen_losses) + np.log(2), label='log(2)'
                )
        elif GAN_method == 'LSGAN':
            plt.plot(
                np.arange(len(gen_losses)),
                0*np.array(gen_losses) + 0.25, label='1/4'
                )
    
    plt.plot(
        np.arange(len(discr_losses)),
        np.array(discr_losses), label='D loss'
        )
    
    plt.legend()
    
    plt.savefig(paths['lossplot'])
    plt.close()
    
    print('done!')
    

def out_png(im_pred, im_target, path, cmap, var=None):
    '''
    This function outputs image for visual inspection of the quality of the model
    '''
    
    matplotlib.use('Agg')
    
    with torch.no_grad():
        
        if var is None:
            f, axarr = plt.subplots( 1,3 )
            axarr[0].imshow(
                im_pred, cmap = cmap, vmin=0, vmax=1
                )
            axarr[1].imshow(
                im_target, cmap = cmap, vmin=0, vmax=1
                )
            axarr[2].imshow(
                torch.abs(im_target-im_pred), cmap = cmap, vmin=0, vmax=1)
            
            axarr[0].title.set_text('Predicted')
            axarr[1].title.set_text('True')
            axarr[2].title.set_text('Error')
        
        else:
            f, axarr = plt.subplots( 2,2 )
            axarr[0,0].imshow(
                im_pred, cmap=cmap, vmin=0, vmax=1
                )
            axarr[0,1].imshow(
                im_target, cmap=cmap, vmin=0, vmax=1
                )
            axarr[1,0].imshow(
                torch.abs(im_target-im_pred), cmap=cmap, vmin=0, vmax=1
                )
            axarr[1,1].imshow(
                var, cmap=cmap, vmin=0, vmax=1
                )
            
            axarr[0,0].title.set_text('Predicted')
            axarr[0,1].title.set_text('True')
            axarr[1,0].title.set_text('Error')
            axarr[1,1].title.set_text('Committee variance')
    
    plt.tight_layout()
    plt.savefig(path)
    
    plt.close()
    del f
    del axarr
    

def out_gifs(y_pred, path, vlims=(0,1)):
    '''
    This function outputs gifs in the relative folder
    '''
    matplotlib.use('Agg')
    
    imagelist = []
    
    vmin, vmax = vlims
    vdelta = vmax - vmin
    
    for ii in range(y_pred.shape[1]):
        imagelist.append( 255*torch.clamp((y_pred[0,ii,:,:,:]-vmin)/vdelta, vmin, vmax).permute(1,2,0).squeeze(-1).numpy() )
        
    imagelist = [Image.fromarray(img) for img in imagelist]
    imagelist[0].save(
        path,
        save_all=True,
        append_images=imagelist[1:],
        duration=100,
        loop=20
        )
    
    del imagelist


def save_model(model, path, mode='dictionary'):
    '''
    This function saves model in specified path
    '''
    if mode == 'dictionary':
        torch.save(
            model.state_dict(),
            path
        )
    elif mode == 'serialized':
        torch.save(
            model,
            path
        )
    else:
        raise ValueError(f"Save mode {mode} is not valid.")


def print_grads(model, label):
    with torch.no_grad():
                            
        grad_max = 0.0
        params = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
        num_params = len(params)
        
        total_norm = 0.0
        
        for p in params:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
            if torch.abs(param_norm) >= grad_max:
                grad_max = param_norm.item()
            total_norm = total_norm ** 0.5
            
        print('<<< Gradient stats <<<')
        print(f'This is the {label} gradient norm: {total_norm}')
        print(f'This is the average {label} gradient norm: {total_norm/num_params}')
        print(f'This is the maximum {label} gradient norm: {grad_max}')
        print('>>> Gradient stats >>>')


def make_singlestep(sequence):
    '''
    This function takes a tensor of shape
    batch x num_steps x num_channels x 1 x 1
    as legacy from KMC GAN, and converts to a sequence
    batch*num_steps x num_channels
    '''
    
    #sequence        = sequence.squeeze() # B x num_steps
    init_seq    = []
    end_seq     = []
    for bb in range(sequence.shape[0]):
        for kk in range(sequence.shape[1]-1):
            init_seq.append( sequence[bb,kk,:,:,:].unsqueeze(0) )
            end_seq.append( sequence[bb,kk+1,:,:,:].unsqueeze(0) )
            
    return torch.cat(init_seq), torch.cat(end_seq)


def log_info(GAN_method, Lip_constraint, GP, W_distance, y_pred, target_data, errG, errD):

    print()
    print('<>'*30)
    
    if GAN_method == 'WGAN':
        print(f'W distance estimate is ----> {(W_distance.item())}')
    else:
        print(f'G loss is ----> {errG.item()}')
        print(f'D loss is ----> {errD.item()}')
    
    if Lip_constraint == 'GP':
        print(f'GP term is ---> {GP.item()}')
    
    print(f'Max phi value is: {y_pred.max()}/{target_data.max()}')
    print(f'Min phi value is: {y_pred.min()}/{target_data.min()}')
    print()
    print(f'Avg phi value is: {y_pred.mean()}/{target_data.mean()}')
    print('<>'*30)
    print()
   

def save_checkpoint(
    epoch           : int,
    generator       : nn.Module,
    discriminator   : nn.Module,
    optimizer_G     : torch.optim,
    optimizer_D     : torch.optim,
    loss_G          : torch.Tensor,
    loss_D          : torch.Tensor,
    save_mode       : str,
    path            : str
    ) -> None:

    # This function saves the training checkpoint
    
    if save_mode == 'serialized':
        torch.save(
            {
                'epoch'         : epoch,
                'generator'     : generator,
                'discriminator' : discriminator,
                'optimizer_G'   : optimizer_G,
                'optimizer_D'   : optimizer_D,
                'loss_G'        : loss_G,
                'loss_D'        : loss_D,
            },
            path
            )

    elif save_mode == 'dictionary':
        torch.save(
            {
                'epoch'         : epoch,
                'generator'     : generator.state_dict(),
                'discriminator' : discriminator.state_dict(),
                'optimizer_G'   : optimizer_G.state_dict(),
                'optimizer_D'   : optimizer_D.state_dict(),
                'loss_G'        : loss_G,
                'loss_D'        : loss_D,
            },
            path
            )

    else:
        raise ValueError(f'Saving mode {save_mode} not recognized.')
