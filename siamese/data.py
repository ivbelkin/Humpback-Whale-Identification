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
    Resize, Compose, Normalize)

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


AUG_INFER = Compose([
    Resize(IMG_SIZE, IMG_SIZE),
    Normalize(),
])


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
        mode: str = None,
        first_idxs: [] = [],
        first_labels: [] = None,
        second_idxs: [] = [],
        second_labels: [] = None,
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
        self._check_args(mode, first_idxs, first_labels, second_idxs, second_labels)
        
        if first_labels is not None and second_labels is not None:
            first_idxs, first_labels, second_idxs, second_labels = \
                self._prepare_data(first_idxs, first_labels, second_idxs, second_labels)
        
        self.mode = mode
        self.first_idxs = first_idxs
        self.first_labels = first_labels
        self.second_idxs = second_idxs
        self.second_labels = second_labels
        
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
        if self.mode == "train":
            return len(self.first_idxs)
        elif self.mode == "valid":
            return len(self.first_idxs)
        elif self.mode == "infer":
            return len(self.first_idxs) * len(self.second_idxs)
        else:
            raise Exception("Not supported")
            
    def _check_args(self, mode, first_idxs, first_labels, second_idxs, second_labels):
        assert mode in ["train", "valid", "infer"]
        assert len(first_idxs) > 0
        assert len(second_idxs) > 0
        assert first_labels is None or (len(first_idxs) == len(first_labels))
        assert second_labels is None or (len(second_idxs) == len(second_labels))
            
    def _get_train_iter(self):
        idxs = []
        second_label2idxs = self._label_to_idxs(
            self.second_idxs, self.second_labels)
        second_all_idxs = set(self.second_idxs)
        for i, label in zip(self.first_idxs, self.first_labels):
            same = second_label2idxs[label]
            if np.random.randint(2):
                j = random.sample(same.difference({i}), 1)[0]
            else:
                j = random.sample(
                    second_all_idxs.difference(same).difference({i}), 1)[0]
            idxs.append([i, j])
        random.shuffle(idxs)
        return iter(idxs)
            
    def _get_valid_iter(self):
        def get_j(i):
            j = random.choice(self.second_idxs)
            while j == i:
                j = random.choice(self.second_idxs)
            return j
        if self.first_labels is not None and self.second_labels is not None:
            return self._get_train_iter()
        else:
            idxs = [[i, get_j(i)] for i in self.first_idxs]
            return iter(idxs)
    
    def _get_infer_iter(self):
        def infer_iter():
            for i in self.first_labels:
                for j in self.second_labels:
                    yield [i, j]
        return infer_iter()
    
    def _label_to_idxs(self, idxs, labels):
        label2idxs = collections.defaultdict(lambda: set())
        for i, label in zip(idxs, labels):
            label2idxs[label].add(i)
        return label2idxs
    
    def _prepare_data(self, first_idxs, first_labels, second_idxs, second_labels):
        common = set(first_labels).intersection(second_labels)
        first_idxs_, first_labels_ = self._filter(first_idxs, first_labels, common)
        second_idxs_, second_labels_ = self._filter(second_idxs, second_labels, common)
        return first_idxs_, first_labels_, second_idxs_, second_labels_
        
    def _filter(self, idxs, labels, common):
        idxs_, labels_ = [], []
        for idx, label in zip(idxs, labels):
            if label in common:
                idxs_.append(idx)
                labels_.append(label)
        return idxs_, labels_


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
        
        all_list, train_list, valid_list = parse_train_csv(
            train_csv,
            train_ext=train_ext,
            folds_seed=folds_seed, 
            n_folds=n_folds, 
            train_folds=train_folds, 
            valid_folds=valid_folds)
                
        # train on train-train samples
        if len(train_list) > 0:
            train_len = len(train_list)
            train_labels = [x["Id"] for x in train_list]
            train_idxs = list(range(train_len))
            sampler = SiameseSampler(
                mode="train", 
                first_idxs=train_idxs,
                first_labels=train_labels,
                second_idxs=train_idxs,
                second_labels=train_labels
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
            print("train-train samples:", len(loader) * batch_size)
            print("train-train batches:", len(loader))
            loaders["train-train"] = loader
        
        if len(valid_list) > 0:
            valid_labels = [x["Id"] for x in valid_list]
            sampler = SiameseSampler(
                mode="valid",
                first_idxs=list(range(len(train_list), len(train_list) + len(valid_list))),
                first_labels=valid_labels, 
                second_idxs=list(range(len(train_list) + len(valid_list))),
                second_labels=train_labels + valid_labels
            )
            valid_loader = UtilsFactory.create_loader(
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
            print("Valid samples:", len(valid_loader) * batch_size)
            print("Valid batches:", len(valid_loader))
            loaders["valid"] = valid_loader
        
        if infer_folder is not None:
            infer_list = parse_infer_folder(infer_folder)
            all_labels = [x["Id"] for x in all_list]
            sampler = SiameseSampler(
                mode="infer", 
                first_idxs=list(range(len(all_list))),
                second_idxs=list(range(len(all_list), len(all_list) + len(infer_list))),
            )
            infer_loader = UtilsFactory.create_loader(
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
            print("Infer samples:", len(infer_loader) * batch_size)
            print("Infer batches:", len(infer_loader))
            loaders["infer"] = infer_loader
            
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
