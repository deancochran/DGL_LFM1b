import torch as th
from sentence_transformers import SentenceTransformer

class SequenceEncoder(object):
    '''Converts a list of unique string values into a PyTorch tensor`'''
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
    @th.no_grad()
    def __call__(self, list):
        return self.model.encode(list, show_progress_bar=True,convert_to_tensor=True, device=self.device)

class CategoricalEncoder(object):
    '''Converts a list of string categorical values into a PyTorch tensor`'''
    def __init__(self, device=None):
        self.device = device
    def __call__(self, list):
        categories = set(category for category in list)
        mapping = {category: i for i, category in enumerate(categories)}
        x = th.zeros(len(list), len(mapping), device=self.device)
        for i, category in enumerate(list):
            x[i, mapping[category]] = 1
        return x.to(device=self.device)

class IdentityEncoder(object):
    '''Converts a list of floating-point values into a PyTorch tensor`'''
    def __init__(self, dtype=None, device=None):
        self.dtype = dtype
        self.device = device
    def __call__(self, list):
        return th.Tensor(list).view(-1, 1).to(self.dtype).to(self.device)