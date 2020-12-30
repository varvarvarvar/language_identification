import random
import numpy as np
import torch

def freeze_seed(seed=1234):
    random.seed(seed)
    np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
