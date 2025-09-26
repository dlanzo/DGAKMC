# this is a simple module containing the definition of some convolution operations

import torch
import torch.nn as nn

boxconv = nn.Conv2d(1,1,kernel_size=3,padding=1,padding_mode='circular',bias=False)
boxconv.weight.data = 1/9*torch.ones(1,1,3,3)
boxconv.weight.requires_grad = False
    
gaussconv_3 = nn.Conv2d(1,1,kernel_size=3, padding=1,padding_mode='circular', bias=False)
gaussconv_3.weight.data[0,0,...] = 1/16 * \
    torch.tensor([[1.,2.,1.],
                  [2.,4.,2.],
                  [1.,2.,1.]])
gaussconv_3.requires_grad = False

gaussconv_5 = nn.Conv2d(1,1,kernel_size=5, padding=2,padding_mode='circular', bias=False)
gaussconv_5.weight.data[0,0,...] = 1/256 * \
    torch.tensor([
        [1., 4., 6., 4., 1.],
        [4.,16.,24.,16., 4.],
        [6.,24.,36.,24., 6.],
        [4.,16.,24.,16., 4.],
        [1., 4., 6., 4., 1.]
        ])
gaussconv_5.requires_grad = False

clean_conv = nn.Conv2d(1,1,kernel_size=3, padding=1,padding_mode='circular', bias=False)
clean_conv.weight.data[0,0,...] =  \
    torch.tensor([[0.,0.,0.],
                  [0.,1.,0.],
                  [0.,0.,0.]])
clean_conv.requires_grad = False

# <<< finite difference operators <<<
dx = nn.Conv2d(1,1,kernel_size=3,padding=1,padding_mode='circular',bias=False)
dx.weight.data[0,0,...] = \
    torch.tensor([[0.,0.,0.],
                  [-1.,0.,1.],
                  [0.,0.,0.]])
dx.requires_grad = False


dy = nn.Conv2d(1,1,kernel_size=3,padding=1,padding_mode='circular',bias=False)
dy.weight.data[0,0,...] = \
    torch.tensor([[0.,1.,0.],
                  [0.,0.,0.],
                  [0.,-1.,0.]])
dy.requires_grad = False
# === finite difference operators ===


def grad_squared(phi):
    with torch.no_grad():
        return dx(phi)**2 + dy(phi)**2
