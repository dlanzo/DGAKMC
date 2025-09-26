# <<< importing stuff <<<
import os
from argparse import ArgumentParser
from warnings import warn
# --- importing stuff ---

class GeneralParser():
    '''
    This is a class implementing a general parser, containing args required for all tasks (e.g. reshape dimension, padding mode, NN topology)
    '''
    
    def __init__(self):
        
        self.parser = ArgumentParser()
       
        self.parser.add_argument(
            '--device',
            type        = str,
            default     = 'cuda',
            help        = 'Selects the device to be used ("cpu" or "cuda"). In the case of multiple GPUs, a specific "cuda:n" can be selected.'
            )
        
        self.parser.add_argument(
            '--size',
            type        = int,
            default     = 64,
            help        = 'Dimension at which images are to be rescaled (measured in pixels); currently working only for square images.'
            )
        
        self.parser.add_argument(
            '--cmap',
            type        = str,
            default     = 'gray',
            help        = 'Colormap to be used in plotting and gifs.'
            )
        
        self.parser.add_argument(
            '--hidden',
            type        = int,
            default     = 2,
            help        = 'Number of hidden layers.'
            )
        
        self.parser.add_argument(
            '--channels',
            type        = int,
            default     = 35,
            help        = 'Number of channels for each inner hidden layer.'
            )

        self.parser.add_argument(
            '--kernel_size',
            type        = int,
            default     = 3,
            help        = 'Size of convolutional kenrels'
            )

        self.parser.add_argument(
            '--seed',
            type        = int,
            default     = 0,
            help        = 'Seed for RNGs.'
            )
        
        self.parser.add_argument(
            '--nographics',
            action      = 'store_true',
            help        = 'Disable graphical output (e.g. training loss plots, predictions pngs etc.). If selected, it will also disable gifs (diregarding --nogifs use).'
            )
        
        self.parser.add_argument(
            '--nogifs',
            action      = 'store_true',
            help        = 'Diable gifs output. If --nographics has not been used, png outputs will still be present'
            )
        
        self.parser.add_argument(
            '--nocrop',
            action      = 'store_true',
            help        = 'Disable image cropping'
            )
        
        self.parser.add_argument(
            '--croplims',
            type        = float,
            nargs       = 2,
            default     = (0.25, 0.75),
            help        = 'Cropping bounds (as a fraction of total image; lower/upper)'
            )
        
        self.parser.add_argument(
            '--debug',
            action      = 'store_true',
            help        = 'Launch program in debug mode. This will lead to partial evaluation of training/validation and test sets.'
            )
        
        self.parser.add_argument(
            '--id',
            type        = str,
            default     = '',
            help        = 'ID of the training procedure (will be used for model, training outputs, testing, inference...).'
            )
        
        self.parser.add_argument(
            '--nproc',
            type        = int,
            default     = 1,
            help        = 'Number of procs to be used in parallelized operations (e.g. num_workers in dataloaders or graphical output processes).'
            )

        self.parser.add_argument(
            '--load_mode',
            type        = str,
            default     = 'dictionary',
            help        = 'Loading mode (dictionary or serialized).'
            )

        self.parser.add_argument(
            '--save_mode',
            type        = str,
            default     = 'dictionary',
            help        = 'Saving mode (dictionary or serialized)'
            )

        self.parser.add_argument(
            '--greedy_mode',
            action      = 'store_true',
            help        = 'Enable greedy mode training.'
            )

        self.parser.add_argument(
            '--resizing_factor',
            type        = float,
            default     = 1.0,
            help        = 'Resizing factor for greedy training mode.'
            )

        self.parser.add_argument(
            '--nonconservative',
            action      = 'store_true',
            help        = 'Toggles non-conservative dynamics'
            )


    def parse_args(self):
        
        args = self.parser.parse_args()
        
        # add "derived" arguments
        if args.nographics:
            args.graphics   = False
            args.gifs       = False
        elif args.nogifs:
            args.graphics   = True
            args.gifs       = False
        else:
            args.graphics   = True
            args.gifs       = True
        
        if args.nocrop:
            args.crop = False
        else:
            args.crop = True

        if args.nonconservative:
            args.conservative = False
        else:
            args.conservative = True

        if args.greedy_mode:
            args.save_mode = 'serialized'
            args.load_mode = 'serialized'

        return args


class TrainingParser( GeneralParser ):
    '''
    This class implements a parser specific for training tasks. Therefore, it will contain number of epochs, lr, ...
    '''
    def __init__(self):
        
        super().__init__()
        
        self.parser.add_argument(
            '--epochs',
            type        = int,
            default     = 1_000,
            help        = 'Number of training epochs.'
            )
        
        self.parser.add_argument(
            '--lr',
            type        = float,
            default     = 5e-4,
            help        = 'Learning rate to be used.'
            )

        self.parser.add_argument(
            '--weightd',
            type        = float,
            default     = 0.0,
            help        = 'Optimizer weight decay.'
            )
        
        self.parser.add_argument(
            '--batch',
            type        = int,
            default     = 1,
            help        = 'Training batch dimension.'
            )
        
        self.parser.add_argument(
            '--rotation',
            action      = 'store_true',
            help        = 'Rotate images in training procedure (continuous angle).'
            )
        
        self.parser.add_argument(
            '--rotation90',
            action      = 'store_true',
            help        = 'Rotate images in training procedure (90 degrees and multiples).'
            )

        self.parser.add_argument(
            '--train_set',
            type        = str,
            default     = 'data/train.txt',
            help        = 'File containing paths of images to be used in training.'
            )
        
        self.parser.add_argument(
            '--valid_set',
            type        = str,
            default     = 'data/valid.txt',
            help        = 'File containing paths of images to be used in validation.'
            )
       
        self.parser.add_argument(
            '--logfreq',
            type        = int,
            default     = 1,
            help        = 'Logging frequency on terminal.'
            )

        self.parser.add_argument(
            '--savefreq',
            type        = int,
            default     = 100,
            help        = 'Save frequency for gif seqences and modelsfor gif seqences and models (interations INSIDE the epoch - default is 100; impose > #batch to save only models at the end of epoch)'
            )
        
        self.parser.add_argument(
            '--superbatch',
            type        = int,
            default     = 1,
            help        = 'Model parameters update frequency (allows for simulating the effects on training of bigger batches).'
            )
        
        self.parser.add_argument(
            '--bootstrap',
            action      = 'store_true',
            help        = 'Use a bootstrap procedure to resample the training/validation sets.'
            )
        
        self.parser.add_argument(
            '--reload_model',
            type        = str,
            default     = '',
            help        = 'Specify path of a .pt model (generator) to reload to continue training'
            )

        self.parser.add_argument(
            '--reload_discr',
            type        = str,
            default     = '',
            help        = 'Specify path of a .pt model (discriminator) to reload and continue training'
            )

        self.parser.add_argument(
            '--reload_optimizer_G',
            type        = str,
            default     = '',
            help        = 'Specify path of a .pt optimizer object (generator) to be reloaded'
            )

        self.parser.add_argument(
            '--reload_optimizer_D',
            type        = str,
            default     = '',
            help        = 'Specify path of a .pt optimizer object (discriminator) to be reloaded'
            )

        self.parser.add_argument(
            '--reload_checkpoint',
            type        = str,
            default     = '',
            help        = 'Specify the path of a .pt checkpoint object used to restart training (will override other reload options such as --reload_optimizer and --reload_model).'
            )

        self.parser.add_argument(
            '--save_checkpoints',
            action      = 'store_true',
            help        = 'Enables saving mode thorugh packed checkpoints. This also saves the optimizers states, so that re-starting a training should not require a lr warmup and should lead to fully reproducible results. This is also more compact from a file point of view, but requires unpacking a dictionary for evaluation (even if serialized saving is enabled).'
            )

        self.parser.add_argument(
            '--noise_shape',
            type        = int,
            default     = 75,
            help        = 'The number of random numbers extracted FOR EVERY PIXEL in z. This means that z.shape = [batch, noise_shape, Lx, Ly]'
            )
        
        self.parser.add_argument(
            '--k_gen_training',
            type        = int,
            default     = 5,
            help        = 'The number of update iterations for the discriminator (5 is the original value in WGAN)'
            )
        
        self.parser.add_argument(
            '--mani_sigma',
            type        = float,
            default     = 0.0,
            help        = 'Additive gaussian noise strength. This smooths out the data distribution and should make convergence easier'
            )
        
        self.parser.add_argument(
            '--optimizer',
            type        = str,
            default     = 'RMSprop',
            help        = 'Optimizer to be used in training'
            )
        
        self.parser.add_argument(
            '--beta1',
            type        = float,
            default     = 0.5,
            help        = 'beta1 parameter in Adam optimizer (momentum term)'
            )
        
        self.parser.add_argument(
            '--beta2',
            type        = float,
            default     = 0.9,
            help        = 'beta2 parameter in Adam optimizer or alpha parameter in RMSprop (RMS term)'
            )
        
        self.parser.add_argument(
            '--show_grads',
            action      = 'store_true',
            help        = 'Show gradient values during training (slows walltime a bit because of extra gradient norm calculations)'
            )
        
        self.parser.add_argument(
            '--Lip_constraint',
            type        = str,
            default     = 'GP',
            help        = 'Lipschitz constraint regularization implementation (Gradient Penalty -"GP"- Spectral Normalization -"SN"- or "none")'
            )
        
        self.parser.add_argument(
            '--GAN_method',
            type        = str,
            default     = 'WGAN',
            help        = 'GAN method to be used. Implemented options: Standard GAN (SGAN), Lest Squares GAN (LSGAN), Relativistic average GAN (RaGAN), Wasserstein GAN (WGAN).'
            )

        self.parser.add_argument(
            '--full_loading',
            action      = 'store_true',
            help        = 'Enable full dataset loading in RAM (check that the machine running this has enough memory'
            )


        
    def parse_args(self):
        
        # rescale learning rate
        args = super().parse_args()
        args.lr /= args.superbatch
        
        # create helper arg on reloading
        if args.reload_model != '' or args.reload_discr != '':
            args.reload = True
        else:
            args.reload = False
        
        return args
