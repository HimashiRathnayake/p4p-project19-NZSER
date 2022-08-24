import os
import torch
import torch.nn as nn
import numpy as np
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

# define constants and global variables
SAMPLING_RATE = 16000
IS_JL = True
device = 'cpu'

class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits

def train_model():
    pass

def load_model():
    # load model from repo
    model_name = os.path.dirname(os.path.realpath(__file__))
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    print(model_name)
    model = EmotionModel.from_pretrained(model_name)
    return processor, model

def process_func(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
    model: EmotionModel = None,
    processor: Wav2Vec2Processor = None,
) -> np.ndarray:
    r"""Predict emotions or extract embeddings from raw audio signal."""

    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = torch.from_numpy(y).to(device)

    # run through model
    with torch.no_grad():
        y = model(y)[0 if embeddings else 1]

    # convert to numpy
    y = y.detach().cpu().numpy()

    return y