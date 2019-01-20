import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict

from catalyst.contrib.registry import Registry
from catalyst.contrib.models import ResnetEncoder
from catalyst.dl.runner import AbstractModelRunner
from catalyst.dl.callbacks import Callback

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

# ---- Runner ----

class ModelRunner(AbstractModelRunner):
    
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
