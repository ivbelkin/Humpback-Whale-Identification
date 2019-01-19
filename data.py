import numpy as np
import pandas as pd
import collections
import cv2

from torch.utils.data.sampler import Sampler
from torchvision import transforms

from catalyst.dl.datasource import AbstractDataSource
from catalyst.dl.utils import UtilsFactory
from catalyst.data.augmentor import Augmentor
from catalyst.data.reader import (
    ImageReader, TextReader, ReaderCompose)
from catalyst.legacy.utils.parse import parse_csv2list

from albumentations import (
    Resize, Compose, Normalize)

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# ---- Augmentations ----

IMG_SIZE = 128


AUG_INFER = Compose([
    Resize(IMG_SIZE, IMG_SIZE),
    Normalize(),
])


INFER_TRANSFORM_FN = [
    Augmentor(
        dict_key="Image0",
        augment_fn=lambda x: AUG_INFER(image=x)["image"]),
    Augmentor(
        dict_key="Image1",
        augment_fn=lambda x: AUG_INFER(image=x)["image"])
]

# ---- Data ----

def parse_train_csv(train_csv, *, train_folder, train_ext, folds_seed, n_folds, train_folds, valid_folds):
    print("WARNING: dummy 'parse_train_csv'")
    df = pd.read_csv(train_csv)
    df = df.sample(frac=1, random_state=folds_seed).reset_index(drop=True)
    list_ = parse_csv2list(df)
    return list_[:-1000], list_[-1000:]
    
    
def parse_infer_folder(infer_folder):
    raise NotImplementedError()
    
    
class RowsReader(object):
    
    def __init__(self, reader: callable, format_: str = "{key}{i}"):
        self.reader = reader
        self.format_ = format_
        
    def __call__(self, rows):
        result = {}
        for i, row in enumerate(rows):
            dict_ = self.reader(row)
            for key, value in dict_.items():
                result[self.format_.format(key=key, i=i)] = value
        return result
    
    
class SiameseLabelMixin(object):
    """
    Adds target 1 for samples with same id_key, 0 otherwise
    """
    
    def __init__(self, dict_first_id_key, dict_second_id_key):
        self.dict_first_id_key = dict_first_id_key
        self.dict_second_id_key = dict_second_id_key
        
    def __call__(self, dict_):
        result = dict(
            target=int(dict_[self.dict_first_id_key] == dict_[self.dict_second_id_key])
        )
        return result
    

class SiameseSampler(Sampler):
    """
    Sample pairs with the same label and different equiprobably
    """
    
    def __init__(self, first_labels, second_labels, mode="train"):
        print("WARNING: dummy 'SiameseSampler'")
        self.first_labels = first_labels
        self.second_labels = second_labels
        
    def __iter__(self):
        first_idx = np.arange(len(self.first_labels))
        second_idx = np.arange(len(self.second_labels))
        np.random.shuffle(second_idx)
        return iter(map(list, zip(first_idx, second_idx)))
    
    def __len__(self):
        return len(self.first_labels)


class SiameseDataSource(AbstractDataSource):
    
    @staticmethod
    def prepare_transforms(*, mode, stage, **kwargs):
        print("WARNING: dummy 'SiameseDataSource.prepare_transforms'")
        return transforms.Compose(INFER_TRANSFORM_FN)
    
    @staticmethod
    def prepare_loaders(
        *, 
        mode, 
        stage=None, 
        n_workers=None, 
        batch_size=None,
        train_folder=None,  # all train data, folder with files like 00fj49fd.jpg [.pth]
        train_csv=None,  # csv with whale ids
        train_ext=".jpg",  # replace extension of train files with train_ext if needed
        infer_folder=None,  # all test images, if None - dont create infer loader
        folds_seed=42, n_folds=5,
        train_folds=None, valid_folds=None,
    ):
        loaders = collections.OrderedDict()
        
        train_list, valid_list = parse_train_csv(
            train_csv, 
            train_folder=train_folder,
            train_ext=train_ext,
            folds_seed=folds_seed, 
            n_folds=n_folds, 
            train_folds=train_folds, 
            valid_folds=valid_folds)
        
        open_fn = ReaderCompose(
            readers=[RowsReader(reader=ReaderCompose(
                readers=[
                    ImageReader(row_key="Image", dict_key="Image", datapath=train_folder),
                    TextReader(row_key="Id", dict_key="Id")]),
                format_="{key}{i}")],
            mixins=[SiameseLabelMixin(
                dict_first_id_key="Id0", dict_second_id_key="Id1")]
        )
        
        if len(train_list) > 0:
            train_labels = [x["Id"] for x in train_list]
            sampler = SiameseSampler(train_labels, train_labels)
            train_loader = UtilsFactory.create_loader(
                data_source=np.array(train_list),  # wrap in ndarray to enable indexing with list
                open_fn=open_fn,
                dict_transform=SiameseDataSource.prepare_transforms(
                    mode="train", stage=stage),
                dataset_cache_prob=-1,
                batch_size=batch_size,
                workers=n_workers,
                shuffle=False,
                sampler=sampler,
            )
            print("Train samples:", len(train_loader) * batch_size)
            print("Train batches:", len(train_loader))
            loaders["train"] = train_loader
        
        if len(valid_list) > 0:
            valid_labels = [x["Id"] for x in valid_list]
            sampler = SiameseSampler(valid_labels, train_labels)
            valid_loader = UtilsFactory.create_loader(
                data_source=np.array(valid_list),  # wrap in ndarray to enable indexing with list
                open_fn=open_fn,
                dict_transform=SiameseDataSource.prepare_transforms(
                    mode="train", stage=stage),
                dataset_cache_prob=-1,
                batch_size=batch_size,
                workers=n_workers,
                shuffle=False,
                sampler=sampler,
            )
            print("Valid samples:", len(valid_loader) * batch_size)
            print("Valid batches:", len(valid_loader))
            loaders["valid"] = valid_loader
        
        if infer_folder is not None:
            raise NotImplementedError()
            
        return loaders
    