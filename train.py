# <<< import external stuff <<<
import torch
import torch.nn as nn
from torchvision import utils, datasets, transforms
from torchview import draw_graph
from torch.fft import rfft2, irfft2

import matplotlib.pyplot as plt
import matplotlib

import os
import sys

import numpy as np

import PIL
from PIL import Image

import time
import copy
# --- import external stuff ---

# <<< import my stuff <<<
import src.convolutions as convolutions
from src.classes import *
from src.utils import *
from src.dataloaders import give_dataloaders
from src.parser import TrainingParser
# --- import my stuff ---


# <<< training function <<<
def train(models, loss_fn, optimizers, loaders, args):
    '''
    This function trains the model given selected loss function
    '''

    # define kernel for prior convolution
    conv = convolutions.gaussconv_5.to(args.device)
    
    # unpack tuples
    model, discriminator        = models
    optimizer, optimizer_D      = optimizers
    train_loader, valid_loader  = loaders

    len_train_loader = len(train_loader)
    len_valid_loader = len(valid_loader)
 

    real_label  = 1.
    fake_label  = 0.
    gen_label   = 1.
    

    discr_losses    = []
    gen_losses      = []
    

    def noise_function(x):
        # this is here as it depends on variables known at runtime
        noise = args.mani_sigma*torch.randn(x.shape, device=x.device)
        noise = noise - torch.mean(noise, dim=(-1,-2), keepdim=True) # project out non-conservative component
        return x + noise
    

    for epoch in range(args.epochs):
        
        future = 0 if args.nogifs else 100
        
        start_epoch = time.time()
        
        log_epoch_start_info(epoch, args.paths)
        
        optimizer.zero_grad()
        
        epoch_gen_losses = []
        epoch_discr_losses = []
        
        # these are here in case we need to perform subtraining at some point...
        train_generator     = True
        train_discriminator = True

        # <<< training loop <<<
        for j, series_pre_gen in enumerate(train_loader):

            # <<< ramping learning rate to prevent hard exit from reloaded mode <<<
            if epoch == 0 and (args.reload_model or args.reload_discr) and (not args.reload_checkpoint):
                local_lr = j/len(train_loader) * args.lr 
                for group in optimizer_D.param_groups:
                    group['lr'] = local_lr
                for group in optimizer.param_groups:
                    group['lr'] = local_lr
                
                print(f'Ramping lr... [{local_lr}/{args.lr}]')

            elif epoch == 1 and j == 0 and (args.reload_model or args.reload_discr) and (not args.reload_checkpoint):
                for group in optimizer_D.param_groups:
                    group['lr'] = args.lr
                for group in optimizer.param_groups:
                    group['lr'] = args.lr

            elif epoch == 0 and j == 0:
                # This will take effect even if there is a reload checkpoint (so that you can change the lr)
                for group in optimizer_D.param_groups:
                    group['lr'] = args.lr
                for group in optimizer.param_groups:
                    group['lr'] = args.lr

                print('Learning rate reset!')

            # === ramping learning rate to prevent hard exit from reloaded mode ===

            if args.debug and j == 5:
                print('BREAKING because of debug mode...')
                break
                
            # <<< update discriminator <<<
            
            start_discr_update = time.time()
                        
            for kk, series_pre  in enumerate(train_loader):
                '''
                Train the generator for k iterations (in theory it should be until convergence)
                '''
                input_data, target_data = make_singlestep(series_pre)
                input_data, target_data = input_data.to(args.device), target_data.to(args.device)
                
                # box convolution to smooth out phi values
                input_data = conv(input_data).detach()
                target_data = conv(target_data).detach()

                # adding noise
                input_data  = noise_function(input_data)
                target_data = noise_function(target_data)

                # make predictions
                y_pred = model(input_data) # prediction
                y_pred = noise_function(y_pred) # add noise to predicted evolution too
                
                # better safe than sorry
                discriminator.train()
                discriminator.zero_grad()
                
                label = torch.full((target_data.shape[0],1), real_label, dtype=torch.float, device=args.device)
                discr_input_true = torch.cat( (input_data, target_data), dim=1 ) # concatenating couple for discriminator
                output = discriminator( discr_input_true )
                        
                errD_real = loss_fn(output.squeeze(), label.squeeze())
                
                
                label = torch.full((target_data.shape[0],1), fake_label, dtype=torch.float, device=args.device)
                
                discr_input_fake = torch.cat( (input_data, y_pred.detach()), dim=1 )
                
                output = discriminator( discr_input_fake )
                
                errD_fake = loss_fn(output.squeeze(), label.squeeze())
                
                # <<< gradient penalty term... <<<
                if args.Lip_constraint == 'GP':
                    coin = torch.rand(args.batch, 1, 1, 1, device=args.device)
                    x_ = coin*discr_input_true + (1-coin)*discr_input_fake
                    x_.requires_grad = True
                    D_x = discriminator(x_)
                    D_grad = torch.autograd.grad( D_x, x_, grad_outputs=torch.ones(D_x.shape, device=args.device))[0]
                    D_grad_norm = torch.sum(D_grad**2, dim=(-1,-2,-3))**0.5
                    GP = (( D_grad_norm  - 1.0)**2).mean()
                else:
                    GP = 0.0
                # === gradient penalty term... ===
                
                if args.GAN_method == 'WGAN':
                    W_distance = (errD_real - errD_fake)
                    errD = -W_distance + 10*GP
                else:
                    W_distance = None
                    errD = 0.5*(errD_real + errD_fake)
                
                errD.backward()
                
                # calculating and plotting gradient values
                if args.show_grads:
                    print_grads(discriminator, 'D')
                
                if (j%args.superbatch == 0 or j==len_train_loader-1) and train_discriminator:
                    print(f'---> Discriminator updated for the {kk}th time')
                    optimizer_D.step()
                    optimizer_D.zero_grad()
                
                errG = torch.tensor(0.0)
                
                if epoch == 0 and j == 0:
                    if kk == 10*args.k_gen_training: break
                elif kk == args.k_gen_training: break
                
            end_discr_update = time.time()
            print(f'Discriminator updates took {end_discr_update-start_discr_update:3} s')
            # --- update discriminator ---
            
            # <<< graphic output <<<
            if args.graphics and j==0:
                
                with torch.no_grad():
                    
                    series_pre = conv(series_pre[0,...].to(args.device)).unsqueeze(0)
                    x = series_pre[0,0,:,:,:].unsqueeze(0).to(args.device)

                    y_pred = [ x.cpu() ]
                    for _ in range(future):
                        x = model( noise_function(x) )
                        y_pred.append( x.cpu() )
                    
                    y_pred_cpu = torch.cat(y_pred)
                    target_data_cpu = series_pre.cpu()
                    
                    make_lossplot(gen_losses, discr_losses, args.GAN_method, args.paths)

                    if not args.nogifs:
                        out_gifs(
                            y_pred  = y_pred_cpu.unsqueeze(0),
                            path    = f'{args.paths["gif"]}/epoch_{epoch}_example_{j}_pred.gif'
                            )
                        
                        out_gifs(
                            y_pred  = target_data_cpu,
                            path    = f'{args.paths["gif"]}/epoch_{epoch}_example_{j}_true.gif'
                            )
                    
            # --- graphic output ---
            
            
            # cutting data in couples
            input_data, target_data = make_singlestep(series_pre_gen)
            # sending data to device
            input_data, target_data = input_data.to(args.device), target_data.to(args.device)
            
            if j%args.logfreq == 0: # <- print sub-epoch infos
                print(f'Passing example[{j}/{len_train_loader-1}] in epoch {epoch}')
            
            # box convolution to smooth out phi values
            input_data = conv(input_data).detach()
            target_data = conv(target_data).detach()

            # adding noise
            input_data  = noise_function(input_data)
            target_data = noise_function(target_data)

            # <<< update generator <<<
            
            if train_generator:
            
                errG = 0.0
                    
                model.zero_grad()
                optimizer.zero_grad()
                
                label = torch.full((target_data.shape[0],1), gen_label, dtype=torch.float, device=args.device)
                
                y_pred = model(input_data) # prediction
                y_pred = noise_function(y_pred) # add noise to predicted evolution too
                
                discr_input = torch.cat( (input_data, y_pred), dim=1 )
                
                output = discriminator( discr_input )
                
                if args.GAN_method == 'WGAN':
                    errG = -loss_fn(output.squeeze(), label.squeeze())
                else:
                    errG = loss_fn(output.squeeze(), label.squeeze())
                errG.backward()
                
                # calculating and plotting gradient values
                if args.show_grads:
                    print_grads(model, 'G')
                       
                if j%args.superbatch == 0 or j==len_train_loader-1:
                    print()
                    print('---> Generator updated')
                    print()
                    optimizer.step()
                    
                    optimizer.zero_grad()
            # --- update generator
            
            loss4print = errG.item()
            if args.GAN_method == 'WGAN':
                loss4print_discr = W_distance.item()
            else:
                loss4print_discr = errD.item()
            
            epoch_gen_losses.append( loss4print )
            epoch_discr_losses.append( loss4print_discr )
            

            if j%args.logfreq == 0:
                log_info(args.GAN_method, args.Lip_constraint, GP, W_distance, y_pred, target_data, errG, errD)
                
            gen_losses.append( epoch_gen_losses[-1] )
            discr_losses.append( np.abs(epoch_discr_losses[-1]) )
            
            with open( f'{args.paths["trainloss"]}', 'a+') as train_loss_file:
                train_loss_file.write(f'{epoch_gen_losses[-1]} {epoch_discr_losses[-1]}\n')
                
            if j%args.savefreq == 0:
                if args.save_checkpoints:
                    save_checkpoint(
                        epoch           = epoch,
                        generator       = model,
                        discriminator   = discriminator,
                        optimizer_G     = optimizer,
                        optimizer_D     = optimizer_D,
                        loss_G          = errG,
                        loss_D          = errD,
                        save_mode       = args.save_mode,
                        path            = f'{args.paths["model"]}/checkpoint_epoch_{epoch}_{j}.pt'
                        )
                else:
                    # in principle this need not be exclusive... but I will not waste memory by double saving stuff!
                    save_model(
                        model   = model,
                        path    = f'{args.paths["model"]}/generator_epoch_{epoch}_{j}.pt',
                        mode    = args.save_mode
                        )
                    save_model(
                        model   = discriminator,
                        path    = f'{args.paths["model"]}/discriminator_epoch_{epoch}_{j}.pt',
                        mode    = args.save_mode
                        )
        # --- training loop ---
        
        # <<< epoch end logging <<<
        end_epoch = time.time()
        epoch_time = end_epoch-start_epoch
        # --- epoch end logging ---

# <<< main function <<<
def main():
    '''
    Main function: istantiation of models and dataloaders and launcing of training function
    '''
    #Parse arguments
    parser  = TrainingParser()
    args    = parser.parse_args()

    # Print conservative dynamics information
    print(f'Generator is conservative: {args.conservative}')
    time.sleep(0.2)
    
    # crate folder structure
    args = build_train_logs_dir_tree(args)
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Instantiate dataloaders
    dataloaders = give_dataloaders(args)
    train_loader = dataloaders["train_set"]
    valid_loader = dataloaders["valid_set"]


    # <<< instantiate models <<<
    model = KMCGeneratorZeroOrder(
        args.noise_shape,
        args.kernel_size,
        args.hidden,
        args.channels,
        conservative    = args.conservative)

    if args.greedy_mode:
        container = KMCGeneratorContainer() # initialize an empty container

    if args.Lip_constraint == 'GP' or 'none':
        discriminator = KMCDiscriminator( args.size, args.channels, 5 )
    elif args.Lip_constraint == 'SN':
        discriminator = KMCDiscriminatorSpectralNorm( args.size, args.channels, 5 )
    else:
        raise NotImplementedError(f'Lipschitz constraint "{Lip_constraint}" has not been implemented yet.')

    print('GENERATOR INFORMATION:')
    print_model_info(model)
    
    print('DISCRIMINATOR INFORMATION:')
    print_model_info(discriminator)
    # --- instantiate models ---
    
    # <<< Reload operation <<<

    if args.reload_checkpoint != '':
        print(f'RELOADING CHECKPOINT {args.reload_checkpoint}')
        checkpoint = torch.load(args.reload_checkpoint)

    # Non-greedy mode
    if not args.greedy_mode:
        if args.reload_checkpoint:
            if args.load_mode == 'serialized':
                model = checkpoint["generator"]
            elif args.load_mode == 'dictionary':
                model.load_state_dict(checkpoint["generator"])
            else:
                raise ValueError(f'Loading mode {args.load_mode} is not recognized')
        elif args.reload_model:
            model = import_model(model, args.reload_model, mode=args.load_mode)
    # Greedy mode
    else:
        if args.reload_model:
            pre_model = import_model(None, args.reload_model, mode=args.load_mode)
        container.add_module(pre_model, resizing_factor=args.resizing_factor)

        container.add_module(model, resizing_factor=1.0) # last model is the one specified in args
        model = container
    
    if args.reload_checkpoint:
        if args.load_mode == 'serialized':
            discriminator = checkpoint["discriminator"]
        elif args.load_mode == 'dictionary':
            discriminator.load_state_dict(checkpoint["discriminator"])
        else:
            raise ValueError(f'Loading mode {args.load_mode} is not recognized.')
    elif args.reload_discr:
        discriminator = import_model(discriminator, args.reload_discr, mode=args.load_mode)
    # --- Reload operation ---

    # <<< put models to device <<<
    model.to(args.device)
    discriminator.to(args.device)
    # --- put models to device ---
    

    # <<< define optimizer <<<

    if not args.greedy_mode:
        trainable_parameters = model.parameters()
    else:
        trainable_parameters = model.give_trainable_params( modules=-1 )

    if args.optimizer == 'Adam':
        optimizer_G = torch.optim.AdamW(
            trainable_parameters,
            lr              = args.lr,
            weight_decay    = args.weightd,
            betas           = (args.beta1, args.beta2)
            )
        optimizer_D = torch.optim.AdamW(
            discriminator.parameters(),
            lr              = args.lr,
            weight_decay    = args.weightd,
            betas           = (args.beta1, args.beta2)
            )
        
    elif args.optimizer == 'RMSprop':
        optimizer_G = torch.optim.RMSprop(
            trainable_parameters,
            lr              = args.lr,
            weight_decay    = args.weightd,
            alpha           = args.beta2
            )
        optimizer_D = torch.optim.RMSprop(
            discriminator.parameters(),
            lr              = args.lr,
            weight_decay    = args.weightd,
            alpha           = args.beta2
            )
    else:
        raise NotImplementedError( f'Optimizer "{args.optimizer}" is not implemented.' )

    if args.reload_checkpoint:
        if args.load_mode == 'serialized':
            optimizer_G = checkpoint["optimizer_G"]
            optimizer_D = checkpoint["optimizer_D"]
        elif args.load_mode == 'dictionary':
            optimizer_G.load_state_dict(checkpoint["optimizer_G"])
            optimizer_D.load_state_dict(checkpoint["optimizer_D"])
        else:
            raise ValueError(f'Loading mode {args.load_mode} is not recognized.')

    
    # --- define optimizer ---

    # save inputs
    save_args(args)
    
    # define loss function
    if args.GAN_method == 'WGAN':
        loss_fn = lambda x,y: torch.mean(x)
    elif args.GAN_method == 'SGAN':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.GAN_method == 'LSGAN':
        loss_fn = nn.MSELoss()
    else:
        raise NotImplementedError( f'GAN method "{args.GAN_method}" is not implemented.' )
    
    models = (model, discriminator)
    optimizers = optimizer_G, optimizer_D
    
    os.mkdir(f'{args.paths["master"]}/relaunch')
    os.system(f'cp *.py {args.paths["master"]}/relaunch/')
    os.system(f'cp -r src {args.paths["master"]}/relaunch/src')
 
    train(models, loss_fn, optimizers, (train_loader, valid_loader), args)


# --- main function ---


# <<< main calling <<<
if __name__ == '__main__':    
    main()
            
# --- main calling ---
