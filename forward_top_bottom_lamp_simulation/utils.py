import torch

def relative_lp_error(pred, y, p=2):
    """
    Calculate relative L2 error norm
    Parameters:
    -----------
    pred: torch.Tensor
        Prediction
    y: torch.Tensor
        Ground truth
    Returns:
    --------
    error: float
        Calculated relative L2 error norm (percentage) on cpu
    """
    
    pred = torch.round(pred) # this made for power, because we have a floating point with 4 number after the commaand the power is a integer number expressed in floating point

    error = (
        torch.mean(torch.linalg.norm(pred - y, ord=p) / torch.linalg.norm(y, ord=p))
        .cpu()
        .numpy()
    )
    return error * 100

def squared_error(pred, y):
    return (pred - y) ** 2