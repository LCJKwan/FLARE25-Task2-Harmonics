from monai.transforms import Transform
from monai.data import MetaTensor
from skimage.morphology import remove_small_objects
from torch import Tensor, tensor

class RemoveSmallObjectsPerClassd(Transform):
    def __init__(self, keys, 
            labels=list(range(1, 14)),
            # min_sizes=[1e5, 1e4, 1e4, 2e3, 2e3, 2e3, 200, 500, 1e3, 500, 1e4, 2e3, 1e4], Original too much?
            min_sizes=[1e4, 1e3, 1e3, 1e3, 1e3, 1e3, 100, 100, 500, 100, 1e3, 1e3, 1e3],
            # min_sizes=[1e5, 1e4, 1e3, 1e3, 1e3, 1e3, 100, 100, 300, 200, 1e3, 1e3, 1e4], For small?
            connectivity=1):
        self.keys = keys
        self.labels = labels
        self.min_sizes = min_sizes
        self.conn = connectivity

    def __call__(self, data):
        for key in self.keys:
            img = data[key].cpu().numpy()
            for lbl, ms in zip(self.labels, self.min_sizes):
                mask = (img == lbl)
                if mask.any():
                    cleaned_mask = remove_small_objects(mask, min_size=ms, connectivity=self.conn)
                    img[mask & (~cleaned_mask)] = 0
        
            if isinstance(data[key], MetaTensor):
                data[key] = MetaTensor(img, meta=data[key].meta)
            elif isinstance(data[key], Tensor):
                data[key] = tensor(img, dtype=data[key].dtype, device=data[key].device)
            else:
                data[key] = img
        return data