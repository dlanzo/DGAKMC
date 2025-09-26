# <<< import external stuff <<<
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import torch.nn.functional as functorch
from torchvision import utils, datasets, transforms
from torch.nn.utils.parametrizations import spectral_norm
from torch.fft import fft2, ifft2, rfft2, irfft2

from collections.abc import Iterable

import PIL
from PIL import Image

import numpy as np
# --- import external stuff ---

# <<< import numba <<<
try:
    from numba import njit, prange
except ImportError:
    def njit(fun): # <- alternative definition of njit
        return fun
    def prange(x): # <- alternative definition of prange
        return range(x)
# --- import numba ---


class ResBlock(nn.Module):
    '''
    Simple definition of residual block (fully connected); acts as the superclass for other residual blocks
    '''
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.input_dim      = input_dim
        self.output_dim     = output_dim
        
        self.net        = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.Tanh(),
            nn.Linear(self.output_dim, self.output_dim)
            )
        self.shortcut   = nn.Linear(self.input_dim, self.output_dim)
        
    def forward(self, x):
        return self.net(x)+self.shortcut(x)


class ResBlockConvSpectralNorm(ResBlock):
    '''
    Convolutional residual block with spectral normalization (to be used with WGAN/for regularization)
    '''
    
    def __init__(self, input_dim, output_dim):
    
        super().__init__(input_dim, output_dim)
        
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d( # spectral norm enforces Lipschitz constraint to the discriminator
                in_channels     = self.input_dim,
                out_channels    = self.output_dim,
                kernel_size     = 3,
                stride          = 1,
                padding         = 1,
                padding_mode    = 'circular'
                ), n_power_iterations=3),
            nn.Tanh(),
            spectral_norm(nn.Conv2d(
                in_channels     = self.output_dim,
                out_channels    = self.output_dim,
                kernel_size     = 3,
                stride          = 1,
                padding         = 1,
                padding_mode    = 'circular'
                ), n_power_iterations=3)
            )
            
        self.shortcut = spectral_norm(nn.Conv2d(
            in_channels     = self.input_dim,
            out_channels    = self.output_dim,
            kernel_size     = 3,
            stride          = 1,
            padding         = 1,
            padding_mode    = 'circular'
            ), n_power_iterations=3)
    

class ResBlockConv(ResBlock):
    '''
    Simple implementation of residual block
    '''
    
    def __init__(self,
        input_dim   : int,
        output_dim  : int,
        kernel_size : int
        ) -> None:
    
        super(ResBlockConv, self).__init__(input_dim, output_dim)

        self.kernel_size = kernel_size

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels     = self.input_dim,
                out_channels    = self.output_dim,
                kernel_size     = self.kernel_size,
                stride          = 1,
                padding         = self.kernel_size//2,
                padding_mode    = 'circular'
                ),
            nn.Tanh(),
            nn.Conv2d(
                in_channels     = self.output_dim,
                out_channels    = self.output_dim,
                kernel_size     = self.kernel_size,
                stride          = 1,
                padding         = self.kernel_size//2,
                padding_mode    = 'circular'
                )
            )
        
        if self.input_dim != self.output_dim:
            self.shortcut = nn.Conv2d(
                in_channels     = self.input_dim,
                out_channels    = self.output_dim,
                kernel_size     = 1,
                stride          = 1,
                padding         = 0,
                padding_mode    = 'circular'
                )
        else:
            self.shortcut = nn.Identity()


    def symmetrize(self):
        # This applies symmetrization to convolutional kernels
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                print('Symmetry parametrization...', end='')
                parametrize.register_parametrization( layer, "weight", SquareSymmetric() )
                print('ok!')

        if isinstance(self.shortcut, nn.Conv2d):
            print('Symmetry parametrization...', end='')
            parametrize.register_parametrization( self.shortcut, "weight", SquareSymmetric() )
            print('ok!')
        

class SquareSymmetric(nn.Module):
    # This implement square symmetry (i.e. radial convolutional kernels)
    def forward( self, X ):

        kernel = X

        for kk in [1,2,3]:
            kernel = kernel + torch.rot90(X, k=kk, dims=(-1,-2))

        kernel = kernel + torch.flip(kernel, dims=(-1,))
        kernel = kernel + torch.flip(kernel, dims=(-2,))

        kernel = kernel + torch.transpose(kernel, -1, -2)
        kernel = kernel + torch.transpose( torch.flip(kernel, dims=(-1,)) , -1, -2)

        return kernel/15



class KMCGeneratorZeroOrder(nn.Module):
    '''
    This is the implementation of a simple ResNet for performing the propagation in time of the KMC evolution; conservation of mass by construction is enforced by removing mean (no flux prediction)
    '''
    def __init__(
        self,
        noise_dim       : int,
        kernel_size     : int,
        res_depth       : int,
        hidden_dim      : int,
        conservative    : bool  = True,
        input_dim       : int   =1
        ) -> None:
        '''
        Generator method
        '''
        super().__init__()
        
        self.noise_dim      = noise_dim
        self.kernel_size    = kernel_size
        self.res_depth      = res_depth
        self.hidden_dim     = hidden_dim

        self.conservative   = conservative

        self.input_dim      = input_dim # can be != 1 for greedy algorithm
        
        net_list = []
        net_list.append( ResBlockConv(self.input_dim+self.noise_dim, self.hidden_dim, self.kernel_size) )
        for _ in range(self.res_depth):
            net_list.append( ResBlockConv(self.hidden_dim, self.hidden_dim, self.kernel_size) )
            # this is removed now -> #net_list.append( nn.Tanh() )
        net_list.append( ResBlockConv(self.hidden_dim, 1, self.kernel_size) )
        
        self.net = nn.Sequential(*net_list)
        
        del net_list


    def symmetrize(self):
        for layer in self.net: # apply symmetrization to all sublayers
            layer.symmetrize()


    def forward(self, x):
        noise_shape = list(x.shape)
        noise_shape[1] = self.noise_dim
        noise = torch.randn( noise_shape, device=x.device )
        
        x_cat = torch.cat( (x, noise), dim=1 )
        x_dot = self.net(x_cat)
        if self.conservative: # remove mean value from x_dot
            x_dot = x_dot - torch.mean( x_dot, dim=(-1,-2), keepdim=True )
        
        return x + x_dot



class KMCGeneratorFlux(nn.Module):
    '''
    This is a new tentative implementation of a simple ResNet for performing the propagation in time of the KMC state; conservation of mass by construction is enforced by predicting the next state as div(J), J bein a zero-mean flux
    '''
    def __init__(
        self,
        noise_dim   : int,
        res_depth   : int,
        hidden_dim  : int,
        input_dim   : int   = 1,
        ) -> None:
        # This is the constructor...

        super().__init__()

        self.noise_dim  = noise_dim
        self.res_depth  = res_depth
        self.hidden_dim = hidden_dim

        self.input_dim  = input_dim

        net_list = []
        net_list.append( ResBlockConv( self.input_dim+self.noise_dim, self.hidden_dim ) ) # first layer converts to input+latent channels to hidden channels
        for _ in range(self.res_depth):
            net_list.append( ResBlockConv(self.hidden_dim, self.hidden_dim) )
        net_list.append( ResBlockConv(self.hidden_dim, 2 ) ) # we get to the J prediction

        self.net = nn.Sequential( *net_list ) # this is the subnet predicting the flux

        del net_list # clear memory: we no longer need this

        self.divergence_filters = None


    def build_divergence_layers(
        self,
        x           : torch.Tensor
        ) -> None:

        device  = x.device # get device

        grad1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='circular')
        grad2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='circular')
        
        gradx_matrix = np.array([[0,0,0],[1,0,-1],[0,0,0]])
        grady_matrix = np.array([[0,1,0],[0,0,0],[0,-1,0]])
        
        grad1.weight = nn.Parameter(torch.from_numpy(gradx_matrix).float().unsqueeze(0).unsqueeze(0))
        grad2.weight = nn.Parameter(torch.from_numpy(grady_matrix).float().unsqueeze(0).unsqueeze(0))
        
        grad1.requires_grad = False
        grad2.requires_grad = False
        
        grad1.to(x.device)
        grad2.to(x.device)
        
        self.divergence_filters = [grad1, grad2]


    def divergence(
        self,
        x       : torch.Tensor
        ) -> torch.Tensor:
        # calculate the divergence
        
        if self.divergence_filters is None:
            self.build_divergence_layers(x)

        Jx, Jy = torch.split( x, 1, dim=1 )

        # out-project the mean component
        Jx = Jx - torch.mean( Jx, dim=(-1,-2), keepdim=True )
        Jy = Jy - torch.mean( Jy, dim=(-1,-2), keepdim=True )

        gradx = self.divergence_filters[0](Jx)
        grady = self.divergence_filters[1](Jy)


        return gradx+grady



    def forward(
        self,
        x       : torch.Tensor
        ) -> torch.Tensor:
        # this is the time propagation
        
        noise_shape     = list(x.shape)
        noise_shape[1]  = self.noise_dim
        noise           = torch.randn( noise_shape, device=x.device )
    
        x_cat           = torch.cat( (x, noise), dim=1 )
        fluxes          = self.net(x_cat)

        div             = self.divergence(fluxes)


        return x + div






class KMCGeneratorZeroOrder_lastTanh(nn.Module):
    '''
    This is the implementation of a simple ResNet for performing the propagation in time of the KMC evolution; last layer is activated by Tanh (maybe this helps with extreme variations?)
    '''
    def __init__(self, noise_dim, res_depth, hidden_dim):
        '''
        Generator method
        '''
        super().__init__()
        
        self.noise_dim  = noise_dim
        self.res_depth  = res_depth
        self.hidden_dim = hidden_dim
        
        net_list = []
        net_list.append( ResBlockConv(1+self.noise_dim, self.hidden_dim) )
        for _ in range(self.res_depth):
            net_list.append( ResBlockConv(self.hidden_dim, self.hidden_dim) )
        net_list.append( ResBlockConv(self.hidden_dim, 1) )
        net_list.append( nn.Tanh() )
        
        self.net = nn.Sequential(*net_list)
        
        del net_list
        

    def forward(self, x):
        noise_shape = list(x.shape)
        noise_shape[1] = self.noise_dim
        noise = torch.randn( noise_shape, device=x.device )
        
        x_cat = torch.cat( (x, noise), dim=1 )
        x_dot = self.net(x_cat)
        x_dot = x_dot - torch.mean( x_dot, dim=(-1,-2), keepdim=True )
        
        return x + x_dot
    
    
    
class KMCGeneratorZeroOrder_splitted(nn.Module):
    '''
    This is the implementation of a simple ResNet for performing the propagation in time of the KMC evolution; the prediction is splitted between a variation and a mask term
    '''
    def __init__(self, noise_dim, res_depth, hidden_dim):
        '''
        Generator method
        '''
        super().__init__()
        
        self.noise_dim  = noise_dim
        self.res_depth  = res_depth
        self.hidden_dim = hidden_dim
        
        net_list = []
        net_list.append( ResBlockConv(1+self.noise_dim, self.hidden_dim) )
        for _ in range(self.res_depth):
            net_list.append( ResBlockConv(self.hidden_dim, self.hidden_dim) )
        net_list.append( ResBlockConv(self.hidden_dim, 2) )
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        self.net = nn.Sequential(*net_list)
        
        del net_list
        

    def forward(self, x):
        noise_shape = list(x.shape)
        noise_shape[1] = self.noise_dim
        noise = torch.randn( noise_shape, device=x.device )
        
        x_cat = torch.cat( (x, noise), dim=1 )
        x_dot = self.net(x_cat)
        
        variation, mask = torch.split(x_dot, 1, dim=1)
        variation = self.sigmoid(variation)
        mask  = self.tanh(mask)
        x_dot = variation*mask
        
        x_dot = x_dot - torch.mean( x_dot, dim=(-1,-2), keepdim=True )
        
        return x + x_dot



class KMCGeneratorContainer(nn.Module):
    '''
    This is a class to contain multiple models. This allows for an easier composition of NN models and/or greedy training.
    '''
    def __init__(self):

        super().__init__()

        self.module_list = nn.ModuleList([])
        self.resizing_factors = []

    def add_module(self, other, resizing_factor=1):

        if isinstance(other, KMCGeneratorContainer): # unpacking
            for module, resizing_factor in zip(other.module_list, other.resizing_factors):
                self.module_list.append( module )
                self.resizing_factors.append( resizing_factor )
        elif not issubclass(type(other), nn.Module): # typechecking
            raise TypeError(f"Module of class {type(other)} cannot be added to container")
        else:
            self.module_list.append(other)
            self.resizing_factors.append(resizing_factor)

    def give_trainable_params(self, modules='all'):
        
        if isinstance(modules, str):
            if modules.lower() == 'all': return self.parameters()
            else: raise ValueError(f"String keyword {modules} invalid for fetching training parameters.")
        elif isinstance(modules, int):
            return self.module_list[modules].parameters()
        elif isinstance(modules, Iterable):
            out_params = []
            for idx in modules:
                out_params += list(self.module_list[modules].parameters())
            return params
        else:
            raise TypeError(f"Invalid modules argument ({type(modules)} was given).")

    def forward(self, x):

        out = x

        for idx, module in enumerate(self.module_list):

            if self.resizing_factors[idx] != 1:
                input_x = functorch.upsample(out, scale_factor=resizing_factors[idx]**(-1))
            else:
                input_x = out

            pred = module(input_x)

            if self.resizing_factors[idx] != 1:
                pred = functorch.upsample(pred, scale_factor=resizing_factors[idx])

            out = pred

        return out




    
class KMCDiscriminator(nn.Module):
    '''
    This is the implementation of a simple image classifier for the KMC evolutions
    '''
    def __init__(self, initial_res, hidden_dim, kernel_size):
        
        super(KMCDiscriminator, self).__init__()
        
        self.initial_res    = initial_res
        self.hidden_dim     = hidden_dim
        self.kernel_size    = kernel_size

        self.down_steps     = int( np.log2(self.initial_res) )-1
        
        net_list = []
        net_list.append( ResBlockConv(2, self.hidden_dim, self.kernel_size) )
        net_list.append( nn.MaxPool2d(2) )
        for _ in range(self.down_steps):
            net_list.append( ResBlockConv(self.hidden_dim, self.hidden_dim, self.kernel_size) )
            net_list.append( nn.MaxPool2d(2) )
        net_list.append( ResBlockConv(self.hidden_dim, 1, 3) )
        #net_list.append( nn.Sigmoid() )
        
        self.net = nn.Sequential(*net_list)
        
        del net_list

        
    def forward(self, x):
        return self.net(x)
    
    
class KMCDiscriminatorSpectralNorm(KMCDiscriminator):
    '''
    This is the implementation of a simple image classifier for the KMC evolutions
    '''
    def __init__(self, initial_res, hidden_dim):
        
        super(KMCDiscriminator, self).__init__()
        
        self.initial_res    = initial_res
        self.hidden_dim     = hidden_dim
        self.down_steps     = int( np.log2(self.initial_res) )-1
        
        net_list = []
        net_list.append( ResBlockConvSpectralNorm(2, self.hidden_dim) )
        net_list.append( nn.MaxPool2d(2) )
        for _ in range(self.down_steps):
            net_list.append( ResBlockConvSpectralNorm(self.hidden_dim, self.hidden_dim) )
            net_list.append( nn.MaxPool2d(2) )
        net_list.append( ResBlockConvSpectralNorm(self.hidden_dim, 1) )
        #net_list.append( nn.Sigmoid() )
        
        self.net = nn.Sequential(*net_list)
        
        del net_list
    
