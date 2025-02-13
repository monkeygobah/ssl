import numpy as np

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch, style = 'linear', consistency = 10, consistency_rampup = 5, thresh = 5):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if epoch + 1 < thresh:
        return 0
    # if style=='linear' :
    #     return consistency * linear_rampup(epoch, consistency_rampup)
    else:
        epoch_to_use = max(0, epoch + 1 - thresh)
        return consistency * sigmoid_rampup(epoch_to_use, consistency_rampup)



def linear_rampup(current, rampup_length):
    """
    Linear rampup from 0 to 1 over rampup_length epochs.
    If current >= rampup_length, we return 1.0.
    """
    if rampup_length == 0:
        return 1.0
    else:
        current = min(current, rampup_length)
        return current / rampup_length

