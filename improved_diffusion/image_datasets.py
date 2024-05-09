from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import copy
import random
import os 

def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    
    classes = None
    initial_cond_grouping = os.environ.get("INITIAL_COND_GROUPING")
    
    # 3/28
    #separate .jpg or .png from filename

    #class labeles are seperated by _'s 
    class_names_i = []
    class_names_j = []
    class_names_k = []
    

    for path in all_files:
        basename_no_ext = bf.basename(path).split(".")[0]
        
        IK, J = basename_no_ext.split('_')
        _, I, K = IK.split('-')
        I, K = int(I[1:]), int(K[1:])
        J = int(J[1:])
        class_names_i.append(I - 1)
        class_names_j.append(J)
        class_names_k.append(K - 1)
            
    print('Is', class_names_i[:100])
    print('Is', class_names_j[:100])
    print('Is', class_names_k[:100])
    print('classcond', class_cond)
    
    sorted_classes_j = {x: i for i, x in enumerate(sorted(set(class_names_j)))}
    class_names_j = [sorted_classes_j[x] for x in class_names_j]

    #orted_classes_k = {x: i for i, x in enumerate(sorted(set(class_names_k)))}
    #class_names_k = [sorted_classes_k[x] for x in class_names_k]

    #Count number of seeds
    num_seeds = len(set(sorted_classes_j))

    #dict with class mapping for each type of class
    
    classes = {'i': class_names_i, 'k': class_names_k, 'j': class_names_j}
    
    dataset = ImageDataset(
        image_size,
        all_files,
        class_cond=class_cond,
        classes=classes,
        num_seeds=num_seeds,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    
    # 4/1 PLACEHOLDER CODE REPLACE WITH ARG
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    elif initial_cond_grouping:
        #Custom dataloader for within seed batch sampling 3/29
        loader = DataLoader(
            dataset, batch_size=batch_size, sampler=Sampler(dataset.classes(), batch_size), num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, class_cond, num_seeds, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes_i = None if classes is None else classes['i'][shard:][::num_shards]
        self.local_classes_j = None if classes is None else classes['j'][shard:][::num_shards]
        self.local_classes_k = None if classes is None else classes['k'][shard:][::num_shards]
        self.class_cond = class_cond
        self.num_seeds = num_seeds
        #create empty list to place indices for each class in num seeds 3/29
        self.indices = [[] for _ in range(num_seeds)] 

        for i, x in enumerate(self.local_classes_j):
            self.indices[x].append(i)
            
    def classes(self):
        return self.indices
        
    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        # I1 = 0 ... I11 = 10, K1 = 11, ... K11 = 21
        
        if self.class_cond:
            out_dict["y"] = np.array((self.local_classes_i[idx], 11+ self.local_classes_k[idx]), dtype=np.int64)
        print(out_dict)
        '''
        if self.local_classes_i is not None:
            out_dict["i"] = np.array(self.local_classes_i[idx], dtype=np.int64)

        if self.local_classes_k is not None:
            out_dict["k"] = np.array(self.local_classes_k[idx], dtype=np.int64)

        if self.local_classes_j is not None:
            out_dict["j"] = np.array(self.local_classes_j[idx], dtype=np.int64)
        '''
        
        return np.transpose(arr, [2, 0, 1]), out_dict
        
#Custom sampler for within-class batch sampling 3/29
class Sampler():
    def __init__(self, classes, batch_size):
        #only need class labeles for random seed 3/29
        self.classes = classes
        self.batch_size = batch_size

    def __iter__(self):
        classes = copy.deepcopy(self.classes)
        
        #classes = list of lists. each sublist contains indexes of datapoints from identical seeds
        
        
        
        # each class gets nInSeed // nbatchs instances with some remainer
        batches = []
        
        #iterate over classes to create batch indices
        #also to shuffle the copy of class
        
        for i in range(len(classes)):
            #shuffle classes[i] (not related to following code)
            random.shuffle(classes[i])
            
            nbatches = len(classes[i]) // self.batch_size
            
            for _ in range(nbatches):
                batches.append(i)

        #end: iterable, groups of 3 from same seed
        random.shuffle(batches)

        res = []
        for a in batches:
            for _ in range(self.batch_size):
                res.append(classes[a].pop())

        return iter(res)
