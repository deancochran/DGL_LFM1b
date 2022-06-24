from numpy import dtype
import torch as th
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class SequenceEncoder(object):
    '''Converts a list of unique string values into a PyTorch tensor`'''
    def __init__(self, model_name='all-MiniLM-L6-v2', device='cpu'):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
    @th.no_grad()
    def __call__(self, list):
        return self.model.encode(list, show_progress_bar=True,convert_to_tensor=True, device=self.device)

class CategoricalEncoder(object):
    '''Converts a list of string categorical values into a PyTorch tensor`'''
    def __init__(self, device='cpu'):
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
    def __init__(self, dtype=th.float, device='cpu'):
        self.dtype = dtype
        self.device = device
    def __call__(self, list):
        return th.Tensor(list).view(-1, 1).to(self.dtype).to(self.device)

# class BinaryEncoder(object):
#     '''Converts a list of categorical numbers into a pytorch tensor'''
#     def __init__(self, device=None, dtype=th.float):
#         self.dtype=dtype
#         self.device=device
#     def __call__(self, arr):
#         result = []
#         for i, val in enumerate(tqdm(reversed(arr), total=len(arr))):
#             encoding=[float(i) for i in bin(val)[2:]]
#             if (i==0):
#                 max_size=len(encoding)
            
#             if max_size > len(encoding):
#                 diff=max_size - len(encoding)
#                 encoding=[0 for _ in range(diff)] + encoding
#             result.append(encoding)

#         return th.tensor(list(reversed(result))).to(self.dtype).to(self.device)