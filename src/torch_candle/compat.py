import sys
import torch_candle

def enable_torch_compat():
    """
    Enables drop-in PyTorch compatibility by registering torch_candle inside 
    Python's active system module registry. Any subsequent imports of `torch`
    will automatically resolve to `torch_candle`!
    """
    # Save real PyTorch in registry before overriding
    if 'real_torch' not in sys.modules:
        if 'torch' in sys.modules and sys.modules['torch'] is not torch_candle:
            sys.modules['real_torch'] = sys.modules['torch']
        else:
            try:
                import importlib
                sys.modules['real_torch'] = importlib.import_module('torch')
            except ImportError:
                pass

    # Expose core modules
    sys.modules['torch'] = torch_candle
    
    # Expose sub-packages
    if hasattr(torch_candle, 'nn'):
        sys.modules['torch.nn'] = torch_candle.nn
        if hasattr(torch_candle.nn, 'functional'):
            sys.modules['torch.nn.functional'] = torch_candle.nn.functional
            
    if hasattr(torch_candle, 'optim'):
        sys.modules['torch.optim'] = torch_candle.optim
        if hasattr(torch_candle.optim, 'lr_scheduler'):
            sys.modules['torch.optim.lr_scheduler'] = torch_candle.optim.lr_scheduler
            
    if hasattr(torch_candle, 'cuda'):
        sys.modules['torch.cuda'] = torch_candle.cuda
        if hasattr(torch_candle.cuda, 'amp'):
            sys.modules['torch.cuda.amp'] = torch_candle.cuda.amp
            
    if hasattr(torch_candle, 'amp'):
        sys.modules['torch.amp'] = torch_candle.amp
        
    if hasattr(torch_candle, 'linalg'):
        sys.modules['torch.linalg'] = torch_candle.linalg
        
    if hasattr(torch_candle, 'fft'):
        sys.modules['torch.fft'] = torch_candle.fft
        
    if hasattr(torch_candle, 'autograd'):
        sys.modules['torch.autograd'] = torch_candle.autograd
        
    if hasattr(torch_candle, 'distributions'):
        sys.modules['torch.distributions'] = torch_candle.distributions
        
    if hasattr(torch_candle, 'utils'):
        sys.modules['torch.utils'] = torch_candle.utils
        if hasattr(torch_candle.utils, 'data'):
            sys.modules['torch.utils.data'] = torch_candle.utils.data
            
    if hasattr(torch_candle, 'multiprocessing'):
        sys.modules['torch.multiprocessing'] = torch_candle.multiprocessing
        
    if hasattr(torch_candle, 'func'):
        sys.modules['torch.func'] = torch_candle.func
        
    if hasattr(torch_candle, 'jit'):
        sys.modules['torch.jit'] = torch_candle.jit
        
    if hasattr(torch_candle, 'c10'):
        sys.modules['torch.c10'] = torch_candle.c10
        
    if hasattr(torch_candle, 'aten'):
        sys.modules['torch.aten'] = torch_candle.aten
        
    if hasattr(torch_candle, 'caffe2'):
        sys.modules['torch.caffe2'] = torch_candle.caffe2
        
    if hasattr(torch_candle, 'torchgen'):
        sys.modules['torch.torchgen'] = torch_candle.torchgen
        
    if hasattr(torch_candle, 'distributed'):
        sys.modules['torch.distributed'] = torch_candle.distributed
        
    if hasattr(torch_candle, 'backends'):
        sys.modules['torch.backends'] = torch_candle.backends
