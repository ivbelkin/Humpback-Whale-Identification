import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import numpy as np

from typing import Dict, Callable

from collections import OrderedDict

from catalyst.contrib.registry import Registry
from catalyst.contrib.models import ResnetEncoder
from catalyst.dl.runner import AbstractModelRunner
from catalyst.dl.callbacks import Callback
from catalyst.dl.state import RunnerState

# ---- Model ----

class Baseline(nn.Module):
    
    def __init__(self, enc):
        super().__init__() 
        self.enc = enc
        self.head = nn.Linear(enc.out_features, 1)
        
    def forward(self, Image1, Image2):
        x1 = self.enc(Image1)
        x2 = self.enc(Image2)
        x = self.distance(x1, x2)
        logits = self.head(x)
        return logits.view(-1)
    
    def distance(self, x1, x2): 
        return (x1 - x2).abs_()
    
    
@Registry.model
def resnet_baseline(resnet):
    enc = ResnetEncoder(**resnet)
    model = Baseline(enc)
    return model


# ---- Logging ----

def prepare_logdir(config):
    model_params = config["model_params"]
    data_params = config["stages"]["data_params"]
    
    return f"-{data_params['valid_folds']}"\
           f"-{model_params['model']}"\
           f"-{model_params['resnet']['arch']}"
    
    
# ---- Callbacks ----

@Registry.callback
class LossCallback(Callback):

    def on_batch_end(self, state):
        logits = state.output["logits"]
        loss = state.criterion(logits, state.input["targets"].float())
        state.loss = loss
        
        
@Registry.callback
class PairScoresCallback(Callback):
    
    def __init__(self, *, loader_mode, **kwargs):
        super().__init__(**kwargs)
        self.loader_mode = loader_mode
    
    def on_batch_end(self, state):
        if state.loader_mode == self.loader_mode:
            file2idx = state.valid_file2idx
            file0 = state.input["ImageFile0"]  # from valid
            file1 = state.input["ImageFile1"]  # from train
            logits = state.output["logits"]
            for f0, f1, logit in zip(file0, file1, logits):
                i = file2idx[f0] - state.train_size
                j = file2idx[f1]
                state.pair_scores[i, j] = logit.item()
            
            
@Registry.callback
class FillingCallback(Callback):
    
    def __init__(self, *, loader_mode, **kwargs):
        super().__init__(**kwargs)
        self.loader_mode = loader_mode
    
    def on_batch_end(self, state):
        state.batch_metrics[self.loader_mode + "_filling"] = 0
        if state.loader_mode == self.loader_mode:
            value = (state.pair_scores != 0).mean()
            state.batch_metrics[self.loader_mode + "_filling"] = value

# ---- Runner ----

class ModelRunner(AbstractModelRunner):
    
    def _init_state(
        self, *, mode: str, stage: str = None, **kwargs
    ) -> RunnerState:
        """
        Inner method for children's classes for state specific initialization.
        :return: RunnerState with all necessary parameters.
        """
        assert "valid_size" in kwargs
        assert "train_size" in kwargs
        assert "valid_file2idx" in kwargs
        assert "_valid_loader_id" in kwargs
        
        # transfer previous counters from old state
        if self.state is not None:
            assert self.state._valid_loader_id == kwargs["_valid_loader_id"]
            additional_kwargs = {
                "step": self.state.step,
                "epoch": self.state.epoch + 1,
                "best_metrics": self.state.best_metrics,
                "pair_scores": self.state.pair_scores
            }
        else:
            additional_kwargs = {
                "pair_scores": np.zeros((kwargs["valid_size"], kwargs["train_size"]))
            }
            
        return RunnerState(
            device=self.device,
            model=self.model,
            stage=self.stage,
            criterion=self.criterion,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            **kwargs,
            **additional_kwargs
        )
        
    def run(
        self,
        *,
        loaders: Dict[str, data.DataLoader],
        callbacks: Dict[str, Callback],
        state_params: Dict = None,
        epochs: int = 1,
        start_epoch: int = 0,
        mode: str = "train",
        verbose: bool = False
    ):
        """
        Main method for running train/valid/infer/debug pipeline over model.
        :param loaders: OrderedDict of torch DataLoaders to run on
        :param callbacks: OrderedDict of callback to use
        :param state_params: params for state initialization
        :param epochs: number of epochs to run
        :param start_epoch:
        :param mode: mode - train/infer/debug
        :param verbose: boolean flag for tqdm progress bar
        """
        assert isinstance(loaders, OrderedDict)
        assert isinstance(callbacks, OrderedDict)
        
        state_params = state_params or {}
        if "valid" in loaders:
            loader = loaders["valid"]
            sampler = loader.sampler
            state_params["train_size"] = len(sampler.train_labels)
            state_params["valid_size"] = len(sampler.valid_labels)
            state_params["valid_file2idx"] = {x["Image"]:i for i, x in enumerate(loader.dataset.data)}
            state_params["_valid_loader_id"] = id(loader)
        
        super().run(
            loaders=loaders, 
            callbacks=callbacks, 
            state_params=state_params,
            epochs=epochs,
            start_epoch=start_epoch,
            mode=mode,
            verbose=verbose
        )
    
    @staticmethod
    def _batch_handler(*, dct: Dict, model: nn.Module) -> Dict:
        """
        Batch handler with model forward.
        :param dct: key-value storage with model inputs
        :param model: model to predict with
        :return: key-value storage with model predictions
        """
        Callback
        logits = model(dct["Image0"].float(), dct["Image1"].float())
        output = {"logits": logits}
        return output
