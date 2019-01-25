import numpy as np
import pandas as pd
import collections
import cv2
import random
import torch
import os

from typing import Dict, Callable

from pathlib import Path

from torch.utils.data.sampler import Sampler
from torchvision import transforms

from catalyst.dl.datasource import AbstractDataSource
from catalyst.dl.utils import UtilsFactory
from catalyst.data.augmentor import Augmentor
from catalyst.data.reader import (
    ImageReader, TextReader, ReaderCompose)
from catalyst.legacy.utils.parse import parse_csv2list

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Resize, Normalize
)

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# ---- Augmentations ----

IMG_SIZE = 128


class AugmentorMany:
    """
    Augmentation abstraction to use with data dictionaries.
    """

    def __init__(
        self, dict_keys: [str], augment_fn: Callable, default_kwargs: Dict = None
    ):
        """
        :param dict_keys: keys to transform
        :param augment_fn: augmentation function to use
        :param default_kwargs: default kwargs for augmentations function
        """
        self.dict_keys = dict_keys
        self.augment_fn = augment_fn
        self.default_kwargs = default_kwargs or {}

    def __call__(self, dict_):
        for key in self.dict_keys:
            dict_[key] = self.augment_fn(dict_[key], **self.default_kwargs)
        return dict_


def strong_aug(p=0.5):
    return Compose([
        Resize(IMG_SIZE, IMG_SIZE),
        HorizontalFlip(),
        OneOf([
            IAAPerspective(),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=0, p=0.2),
        ]),
        #RandomBrightnessContrast(),
        Normalize(),
    ])


AUG_TRAIN = strong_aug(p=0.5)
AUG_INFER = Compose([
    Resize(IMG_SIZE, IMG_SIZE),
    Normalize(),
])

TRAIN_TRANSFORM_FN = [
    AugmentorMany(
        dict_keys=["Image0", "Image1"],
        augment_fn=lambda x: AUG_TRAIN(image=x)["image"]),
    AugmentorMany(
        dict_keys=["Image0", "Image1"],
        augment_fn=lambda x: torch.tensor(x).permute(2, 0, 1)),
]

INFER_TRANSFORM_FN = [
    AugmentorMany(
        dict_keys=["Image0", "Image1"],
        augment_fn=lambda x: AUG_INFER(image=x)["image"]),
    AugmentorMany(
        dict_keys=["Image0", "Image1"],
        augment_fn=lambda x: torch.tensor(x).permute(2, 0, 1)),
]

# ---- Data ----

def parse_train_csv(train_csv, *, train_ext, folds_seed, n_folds, train_folds, valid_folds):
    def change_ext(path: str, ext: str):
        return str(Path(path).with_suffix(ext))
    
    def add_size_column(df):
        size = df.groupby("Id")\
                 .apply(lambda x: pd.Series({"size": len(x)}))\
                 .reset_index()
        return df.merge(size, how="left", on="Id")
    
    def filter_df(df):
        return df.query("Id != 'new_whale' and size > 1")
    
    def add_fold_column(df, folds_seed, n_folds):
        np.random.seed(folds_seed)
        def process_same_size(df):
            def process_large(df):
                df = df.sample(frac=1)\
                       .reset_index(drop=True)
                return pd.DataFrame(dict(
                    Image=df["Image"],
                    fold=1 + np.arange(len(df)) % n_folds))
            
            def process_small(df):
                df = df.sample(frac=1)\
                       .reset_index(drop=True)
                fold = 1 + np.random.randint(n_folds)
                return pd.DataFrame(dict(
                    Image=df["Image"],
                    fold=[fold] * len(df)))
            
            size = df["size"].iloc[0]
            n = len(df)
            if size / n_folds >= 2:
                fold = df.groupby("Id")\
                         .apply(process_large)\
                         .reset_index(drop=True)
            elif size >= 2 and n >= n_folds:
                fold = df.groupby("Id")\
                         .apply(process_small)\
                         .reset_index(drop=True)
            else:
                print("Skip train ids:", ", ".format(set(df["Id"])))
            return fold
            
        fold = df.groupby("size")\
                 .apply(process_same_size)\
                 .reset_index(drop=True)
        return df.merge(fold, how="left", on="Image")
    
    def select_folds(df, folds):
        return df.query("fold in {}".format(tuple(folds)))
    
    def split_by_fold(df, train_folds, valid_folds):
        train_df = select_folds(df, train_folds)
        valid_df = select_folds(df, valid_folds)
        return train_df, valid_df
    
    df = pd.read_csv(train_csv)
    df["Image"] = df["Image"].map(lambda x: change_ext(x, train_ext))
    df = add_size_column(df)
    df_filtered = filter_df(df)
    df_filtered = add_fold_column(df_filtered, folds_seed, n_folds)
    train_df, valid_df = split_by_fold(df_filtered, train_folds, valid_folds)
    
    df = df.drop(["size"], axis=1)
    train_df = train_df.drop(["size", "fold"], axis=1)
    valid_df = valid_df.drop(["size", "fold"], axis=1)

    df_list = parse_csv2list(df)
    train_list = parse_csv2list(train_df)
    valid_list = parse_csv2list(valid_df)
    
    return df_list, train_list, valid_list
    
    
def parse_infer_folder(infer_folder):
    return [{"Image": path} for path in os.listdir(infer_folder)]

    
class RowsReader(object):
    
    def __init__(
        self, 
        reader: callable = None, 
        readers: [callable] = None, 
        suffixes: [str] = None,
    ):
        assert (reader is not None) or (readers is not None)
        
        self.reader = reader
        self.readers = readers
        self.suffixes = suffixes
        
    def __call__(self, rows):
        self._check_rows(rows)
        result = {}
        suffixes = self.suffixes or list(range(len(rows)))
        if self.readers is not None:
            for row, reader, suffix in zip(rows, self.readers, suffixes):
                self._read(row, reader, suffix, result)
        if self.reader is not None:
            for row, suffix in zip(rows, suffixes):
                self._read(row, self.reader, suffix, result)
        return result
    
    def _check_rows(self, rows):
        assert (self.readers is None) or (len(self.readers) == len(rows))
        assert (self.suffixes is None) or (len(self.suffixes == len(rows)))
        
    def _read(self, row, reader, suffix, result):
        dict_ = reader(row)
        for key, value in dict_.items():
            result[key + str(suffix)] = value
    
    
class SiameseLabelMixin(object):
    """
    Adds target 1 for samples with same id_key, 0 otherwise
    """
    
    def __init__(self, dict_first_id_key, dict_second_id_key, output_key="targets"):
        self.dict_first_id_key = dict_first_id_key
        self.dict_second_id_key = dict_second_id_key
        self.output_key = output_key
        
    def __call__(self, dict_):
        result = {
            self.output_key:
                int(dict_[self.dict_first_id_key] == dict_[self.dict_second_id_key])
        }
        return result
    

class SiameseSampler(Sampler):
    """
    Sample pairs of indices
    """
    
    def __init__(
        self, 
        *,
        mode: str = "train",
        train_idxs: [] = [],
        train_labels: [] = [],
        valid_idxs: [] = None,
        valid_labels: [] = None,
        infer_idxs: [] = None,
        size: int = None,
    ):
        """
        :param: mode, one of "train", "valid", "infer"
            "train":
                both *_labels must be specified
                return indices of samples with the same label with probability 0.5
                number of samples in epoch is equal to len(first_idxs)
            "valid":
                if both *_labels are None, sample random pairs of indices
                otherwise behave as in "train" mode
                number of samples in epoch is equal to len(first_idxs)
            "infer":
                both *_labels are ignored
                sample all possible pairs
                number of samples in epoch is equal to len(first_idxs) * len(second_idxs)
        
        Note: never return pairs like (i, i) !
        """        
        self.mode = mode
        self.train_idxs = train_idxs
        self.train_labels = train_labels
        self.valid_idxs = valid_idxs
        self.valid_labels = valid_labels
        self.infer_idxs = infer_idxs
        self.size = size
        
    def __iter__(self):
        if self.mode == "train":
            return self._get_train_iter()
        elif self.mode == "valid":
            return self._get_valid_iter()
        elif self.mode == "infer":
            return self._get_infer_iter()
        else:
            raise Exception("Not supported")
    
    def __len__(self):
        if self.mode == "infer":
            return len(self.train_idxs) * len(self.infer_idxs)
        return self.size
            
    def _get_train_iter(self):
        label2idxs = self._label_to_idxs(self.train_idxs, self.train_labels)
        def train_iter():
            for _ in range(self.size):
                if random.randint(0, 1):
                    # both with the same label
                    label = random.sample(label2idxs.keys(), 1)[0]
                    i, j = random.sample(label2idxs[label], 2)
                else:
                    # with different labels
                    labels = random.sample(label2idxs.keys(), 2)
                    i = random.sample(label2idxs[labels[0]], 1)[0]
                    j = random.sample(label2idxs[labels[1]], 1)[0]
                yield [i, j]
        return train_iter()
            
    def _get_valid_iter(self):
        train_label2idxs = self._label_to_idxs(self.train_idxs, self.train_labels)
        valid_label2idxs = self._label_to_idxs(self.valid_idxs, self.valid_labels)
        common_labels = set(train_label2idxs).intersection(valid_label2idxs)
        def valid_iter():
            for _ in range(self.size):
                if random.randint(0, 1) and len(self.train_idxs) > 0:
                    # one from train, another from valid
                    if random.randint(0, 1):
                        # both with the same label:
                        label = random.sample(common_labels, 1)[0]
                        i = random.sample(train_label2idxs[label], 1)[0]
                        j = random.sample(valid_label2idxs[label], 1)[0]
                    else:
                        # with different labels:
                        train_label = random.sample(train_label2idxs.keys(), 1)[0]
                        valid_label = random.sample(valid_label2idxs.keys(), 1)[0]
                        i = random.sample(train_label2idxs[train_label], 1)[0]
                        j = random.sample(valid_label2idxs[valid_label], 1)[0]
                else:
                    # both from valid
                    if random.randint(0, 1):
                        # both with the same label
                        label = random.sample(valid_label2idxs.keys(), 1)[0]
                        while len(valid_label2idxs[label]) < 2:
                            label = random.sample(valid_label2idxs.keys(), 1)[0]
                        i, j = random.sample(valid_label2idxs[label], 2)
                    else:
                        # with different labels
                        labels = random.sample(valid_label2idxs.keys(), 2)
                        i = random.sample(valid_label2idxs[labels[0]], 1)[0]
                        j = random.sample(valid_label2idxs[labels[1]], 1)[0]
                yield [i, j]
        return valid_iter()
    
    def _get_infer_iter(self):
        def infer_iter():
            for i in self.train_idxs:
                for j in self.infer_idxs:
                    yield [i, j]
        return infer_iter()
    
    def _label_to_idxs(self, idxs, labels):
        label2idxs = collections.defaultdict(lambda: set())
        for i, label in zip(idxs, labels):
            label2idxs[label].add(i)
        return label2idxs


class SiameseDataSource(AbstractDataSource):
    
    @staticmethod
    def prepare_transforms(*, mode, stage, **kwargs):
        if mode == "train":
            return transforms.Compose(TRAIN_TRANSFORM_FN)
        elif mode == "valid":
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
        
        all_list, train_list, valid_list = parse_train_csv(
            train_csv,
            train_ext=train_ext,
            folds_seed=folds_seed, 
            n_folds=n_folds, 
            train_folds=train_folds, 
            valid_folds=valid_folds)
        
        train_len = len(train_list)
        train_labels = [x["Id"] for x in train_list]
        train_idxs = list(range(train_len))
        
        valid_len = len(valid_list)
        valid_labels = [x["Id"] for x in valid_list]
        valid_idxs = list(range(train_len, train_len + valid_len))
                
        # train on train-train samples
        if train_len > 0:
            sampler = SiameseSampler(
                mode="train", 
                train_idxs=train_idxs,
                train_labels=train_labels,
                size=train_len,
            )
            loader = UtilsFactory.create_loader(
                data_source=np.array(train_list),  # wrap in ndarray to enable indexing with list
                open_fn=SiameseDataSource._get_train_open_fn(train_folder),
                dict_transform=SiameseDataSource.prepare_transforms(
                    mode="train", stage=stage),
                dataset_cache_prob=-1,
                batch_size=batch_size,
                workers=n_workers,
                shuffle=False,
                sampler=sampler,
            )
            print("train samples:", len(loader) * batch_size)
            print("train batches:", len(loader))
            loaders["train"] = loader
        
        if len(valid_list) > 0:
            sampler = SiameseSampler(
                mode="valid",
                train_idxs=train_idxs,
                train_labels=train_labels,
                valid_idxs=valid_idxs,
                valid_labels=valid_labels,
                size=valid_len,
            )
            loader = UtilsFactory.create_loader(
                data_source=np.array(train_list + valid_list),  # wrap in ndarray to enable indexing with list
                open_fn=SiameseDataSource._get_train_open_fn(train_folder),
                dict_transform=SiameseDataSource.prepare_transforms(
                    mode="valid", stage=stage),
                dataset_cache_prob=-1,
                batch_size=batch_size,
                workers=n_workers,
                shuffle=False,
                sampler=sampler,
            )
            print("valid samples:", len(loader) * batch_size)
            print("valid batches:", len(loader))
            loaders["valid"] = loader
        
        if infer_folder is not None:
            infer_list = parse_infer_folder(infer_folder)
            all_labels = [x["Id"] for x in all_list]
            all_len = len(all_list)
            infer_len = len(infer_list)
            sampler = SiameseSampler(
                mode="infer", 
                train_idxs=list(range(all_len)),
                infer_idxs=list(range(all_len, all_len + infer_len))
            )
            loader = UtilsFactory.create_loader(
                data_source=np.array(all_list + infer_list),
                open_fn=SiameseDataSource._get_infer_open_fn(train_folder, infer_folder),
                dict_transform=SiameseDataSource.prepare_transforms(
                    mode="infer", stage=stage),
                dataset_cache_prob=-1,
                batch_size=batch_size,
                workers=n_workers,
                shuffle=False,
                sampler=sampler,
            )
            print("infer samples:", len(loader) * batch_size)
            print("infer batches:", len(loader))
            loaders["infer"] = loader
            
        return loaders
    
    @staticmethod
    def _get_train_open_fn(train_folder):
        return ReaderCompose(
            readers=[RowsReader(reader=ReaderCompose(
                readers=[
                    ImageReader(row_key="Image", dict_key="Image", datapath=train_folder),
                    TextReader(row_key="Id", dict_key="Id"),
                    TextReader(row_key="Image", dict_key="ImageFile")]))],
            mixins=[SiameseLabelMixin(
                dict_first_id_key="Id0", dict_second_id_key="Id1")]
        )
    
    @staticmethod
    def _get_infer_open_fn(train_folder, infer_folder):
        return RowsReader(
            reader=TextReader(row_key="Image", dict_key="ImageFile"),
            readers=[
                ImageReader(row_key="Image", dict_key="Image", datapath=train_folder),
                ImageReader(row_key="Image", dict_key="Image", datapath=infer_folder)])
