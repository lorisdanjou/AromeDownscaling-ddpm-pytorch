import numpy as np
from skimage.metrics import structural_similarity

def tensor2image(tensor):
    """
    Takes a torch tensor as input (2, 3 or 4D) and returns a numpy array.
    """
    n_dim = tensor.dim() 
    if n_dim == 4:
        if tensor.shape[0] == 1:
            return tensor.cpu().numpy()[0, :, :, :].transpose((1,2,0))
        else:
            raise ValueError("batch_size > 1")
    elif n_dim == 3:
        return tensor.cpu().numpy().transpose((1,2,0))
    elif n_dim == 2:
        return tensor.cpu().numpy()
    else:
        raise ValueError("n_dim nor in [2, 3, 4]")

def mse_map(a, b):
    """
    Computes the squared error map between two numpy arrays.
    """
    return (a - b)**2

def mae_map(a, b):
    """
    Computes the absolute error map between two numpy arrays.
    """
    return np.abs(a - b)

def bias_map(a, b):
    """
    Computes the bias map between two numpy arrays.
    """
    return a - b

def ssim_map(a, b):
    """
    Computes the SSIM map (using scikit-learn) between two numpy arrays.
    """
    ssim_m, ssim_map = structural_similarity(
        a,
        b, 
        data_range=b.max() - b.min(),
        win_size=None,
        full=True
    )
    return ssim_map

def score_value(a, b, score):
    """
    Computes a given score map between two numpy arrays and returns the mean value.
    """
    if score == "mse":
        return mse_map(a, b).mean()
    elif score == "mae":
        return mae_map(a, b).mean()
    elif score == "bias":
        return bias_map(a, b).mean()
    elif score == "ssim":
        return ssim_map(a, b).mean()
    else:
        raise NotImplementedError
    
