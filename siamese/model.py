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

from sklearn import metrics

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
            
    def on_loader_start(self, state):
        # just reserve matrix, if doesnt exist
        if state.loader_mode not in state.pair_scores:
            loader_params = state.loader_params[state.loader_mode]
            state.pair_scores[state.loader_mode] = np.zeros((
                len(loader_params["first_file2table"]),
                len(loader_params["second_file2table"])
            ))
            state.pair_labels[state.loader_mode] = -1 * np.ones((
                len(loader_params["first_file2table"]),
                len(loader_params["second_file2table"])
            ))
    
    def on_batch_end(self, state):
        loader_params = state.loader_params[state.loader_mode]
        file0 = state.input["ImageFile0"]
        file1 = state.input["ImageFile1"]
        logits = state.output["logits"]
        targets = state.input["targets"]
        for f0, f1, logit, target in zip(file0, file1, logits, targets):
            i = loader_params["first_file2table"][f0]
            j = loader_params["second_file2table"][f1]
            state.pair_scores[state.loader_mode][i, j] = logit.item()
            state.pair_labels[state.loader_mode][i, j] = target.item()
            
            
@Registry.callback
class FillingCallback(Callback):
        
    def on_batch_end(self, state):
        value = (state.pair_labels[state.loader_mode] > -0.5).mean()
        state.batch_metrics["filling"] = value
            
            
@Registry.callback
class F1Callback(Callback):
        
    def on_batch_end(self, state):
        lm = state.loader_mode
        idx = state.pair_labels[lm] > -0.5
        y_true = state.pair_labels[lm][idx].flatten().astype(int)
        y_pred = (state.pair_scores[lm][idx].flatten() > 0).astype(int)
        value = metrics.f1_score(y_true, y_pred)
        state.batch_metrics["f1_score"] = value


@Registry.callback
class PrecisionCallback(Callback):
    
    def on_batch_end(self, state):
        lm = state.loader_mode
        idx = state.pair_labels[lm] > -0.5
        y_true = state.pair_labels[lm][idx].flatten().astype(int)
        y_pred = (state.pair_scores[lm][idx].flatten() > 0).astype(int)
        value = metrics.precision_score(y_true, y_pred)
        state.batch_metrics["precision_score"] = value
        
        
@Registry.callback
class RecallCallback(Callback):
    
    def on_batch_end(self, state):
        lm = state.loader_mode
        idx = state.pair_labels[lm] > -0.5
        y_true = state.pair_labels[lm][idx].flatten().astype(int)
        y_pred = (state.pair_scores[lm][idx].flatten() > 0).astype(int)
        value = metrics.recall_score(y_true, y_pred)
        state.batch_metrics["recall_score"] = value
        
        
@Registry.callback
class AccuracyCallback(Callback):
    
    def on_batch_end(self, state):
        lm = state.loader_mode
        idx = state.pair_labels[lm] > -0.5
        y_true = state.pair_labels[lm][idx].flatten().astype(int)
        y_pred = (state.pair_scores[lm][idx].flatten() > 0).astype(int)
        value = metrics.accuracy_score(y_true, y_pred)
        state.batch_metrics["accuracy_score"] = value

# ---- Runner ----

class ModelRunner(AbstractModelRunner):
    
    def _init(self):
        """
        Inner method for children's classes for model specific initialization.
        As baseline, checks device support and puts model on it.
        :return:
        """
        super()._init()
        for p in self.model.enc.parameters():
            p.requires_grad = False
    
    def _init_state(
        self, *, mode: str, stage: str = None, **kwargs
    ) -> RunnerState:
        """
        Inner method for children's classes for state specific initialization.
        :return: RunnerState with all necessary parameters.
        """
        
        # transfer previous counters from old state
        if self.state is not None:
            # prevent running with another loader but with the same loader_mode
            for loader_mode, loader_params in self.state.loader_params.items():
                assert loader_params["id"] == kwargs["loader_params"][loader_mode]["id"]
            additional_kwargs = {
                "step": self.state.step,
                "epoch": self.state.epoch + 1,
                "best_metrics": self.state.best_metrics,
                "pair_scores": self.state.pair_scores,
                "pair_labels": self.state.pair_labels,
            }
        else:
            # just reserve state parameter for tables for different loaders
            additional_kwargs = {
                "pair_scores": {}, "pair_labels": {},
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
        
        state_params = state_params or {}
        # add mapping from image filename to index in pair_scores table
        state_params["loader_params"] = {}
        for loader_mode, loader in loaders.items():
            sampler = loader.sampler
            data = loader.dataset.data
            loader_params = {}
            loader_params["id"] = id(loader)  # to prevent run with another loader
            loader_params["first_file2table"] = {
                data[idx]["Image"]: i for i, idx in enumerate(sampler.first_idxs)}
            loader_params["second_file2table"] = {
                data[idx]["Image"]: i for i, idx in enumerate(sampler.second_idxs)}
            state_params["loader_params"][loader_mode] = loader_params
        
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
        logits = model(dct["Image0"].float(), dct["Image1"].float())
        output = {"logits": logits}
        return output
