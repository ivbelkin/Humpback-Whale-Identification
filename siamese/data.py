import numpy as np
import pandas as pd
import collections
import cv2
import random
import torch
import os

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


AUG_INFER = Compose([
    Resize(IMG_SIZE, IMG_SIZE),
    #Normalize(),
])


INFER_TRANSFORM_FN = [
    Augmentor(
        dict_key="Image0",
        augment_fn=lambda x: AUG_INFER(image=x)["image"]),
    Augmentor(
        dict_key="Image0",
        augment_fn=lambda x: torch.tensor(x).permute(2, 0, 1)),
    
    Augmentor(
        dict_key="Image1",
        augment_fn=lambda x: AUG_INFER(image=x)["image"]),
    Augmentor(
        dict_key="Image1",
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
    
    def __init__(self, readers: [callable], format_: str = "{key}{i}"):
        self.readers = readers
        self.format_ = format_
        
    def __call__(self, rows):
        self._check_rows(rows)
        readers = self.readers if len(self.readers) > 1 else self.readers * len(rows)
        result = {}
        for i, (row, reader) in enumerate(zip(rows, readers)):
            dict_ = reader(row)
            for key, value in dict_.items():
                result[self.format_.format(key=key, i=i)] = value
        return result
    
    def _check_rows(self, rows):
        assert (len(self.readers) == 1) or (len(self.readers) == len(rows))
    
    
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
    
    def __init__(
        self, 
        *,
        mode: str = "train",
        train_labels: [] = None, 
        valid_labels: [] = None, 
        infer_size: int = None, 
    ):
        """
        :param: mode = "train", "valid", "infer"
            "train": 
                data = train_labels
                train_labels x train_labels
            "valid": 
                data = train_labels + valid_labels
                valid_labels x (train_labels + valid_labels)
            "infer":
                data = train_labels + infer_labels
                infer_labels x (train_labels + infer_labels)
        """

        self.mode = mode
        self.train_labels = train_labels
        self.valid_labels = valid_labels
        self.infer_size = infer_size
        
    def __iter__(self):
        if self.mode in ["train"]:
            return self._get_train_iter()
        elif self.mode in ["valid"]:
            return self._get_valid_iter()
        elif self.mode in ["infer"]:
            return self._get_infer_iter()
        else:
            raise Exception("Not supported")
    
    def __len__(self):
        if self.mode in ["train"]:
            return len(self.train_labels)
        elif self.mode in ["valid"]:
            return len(self.valid_labels)
        elif self.mode in ["infer"]:
            return len(self.train_labels) * self.infer_size
        else:
            raise Exception("Not supported")
            
    def _get_train_iter(self):
        idxs = []
        label2idxs = self._label_to_idxs(self.train_labels)
        all_idxs = set(range(len(self.train_labels)))
        for i, label in enumerate(self.train_labels):
            same = label2idxs[label].copy()
            if np.random.randint(2):
                same.remove(i)
                j = random.choice(same)
            else:
                j = random.sample(all_idxs.difference(same), 1)[0]
            idxs.append([i, j])
        return iter(idxs)
            
    def _get_valid_iter(self):
        train_size = len(self.train_labels)
        valid_size = len(self.valid_labels)
        idxs = [[train_size + i, np.random.randint(train_size + valid_size)] 
                    for i in range(valid_size)]
        return iter(idxs)
    
    def _get_infer_iter(self):
        train_size = len(self.train_labels)
        def infer_iter():
            for i in range(train_size):
                for j in range(self.infer_size):
                    yield [i, train_size + j]
        return infer_iter()
    
    def _label_to_idxs(self, labels):
        label2idxs = collections.defaultdict(lambda: [])
        for idx, label in enumerate(labels):
            label2idxs[label].append(idx)
        for l, idxs in label2idxs.items():
            assert len(idxs) > 1, l
        return label2idxs


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
                
        if len(train_list) > 0:
            train_labels = [x["Id"] for x in train_list]
            sampler = SiameseSampler(mode="train", train_labels=train_labels)
            train_loader = UtilsFactory.create_loader(
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
            print("Train samples:", len(train_loader) * batch_size)
            print("Train batches:", len(train_loader))
            loaders["train"] = train_loader
        
        if len(valid_list) > 0:
            valid_labels = [x["Id"] for x in valid_list]
            sampler = SiameseSampler(mode="valid", 
                                     train_labels=train_labels, 
                                     valid_labels=valid_labels)
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
            sampler = SiameseSampler(mode="infer", 
                                     train_labels=all_labels,
                                     infer_size=len(infer_list))
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
            readers=[RowsReader(readers=[ReaderCompose(
                readers=[
                    ImageReader(row_key="Image", dict_key="Image", datapath=train_folder),
                    TextReader(row_key="Id", dict_key="Id")])],
                format_="{key}{i}")],
            mixins=[SiameseLabelMixin(
                dict_first_id_key="Id0", dict_second_id_key="Id1")]
        )
    
    @staticmethod
    def _get_infer_open_fn(train_folder, infer_folder):
        return RowsReader(
            readers=[
                ImageReader(row_key="Image", dict_key="Image", datapath=train_folder),
                ImageReader(row_key="Image", dict_key="Image", datapath=infer_folder)
            ],
            format_="{key}{i}")
