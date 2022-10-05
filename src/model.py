import dataclasses
import datetime
import os
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset, Audio, Dataset
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor, EvalPrediction
import transformers
from transformers.trainer import Trainer, TrainingArguments
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2FeatureExtractor,
)
from transformers.integrations import TensorBoardCallback
import audeer, audmetric
import audiofile
import typing

metrics = {
    'PCC': audmetric.pearson_cc,
    'CCC': audmetric.concordance_cc,
    'MSE': audmetric.mean_squared_error,
    'MAE': audmetric.mean_absolute_error,
}

# define constants and global variables
SAMPLING_RATE = 16000
IS_JL = True
device = 'cpu'

log_root = audeer.mkdir('log')
model_root = audeer.mkdir('model')

class ConcordanceCorCoeff(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.mean = torch.mean

        self.var = torch.var

        self.sum = torch.sum

        self.sqrt = torch.sqrt

        self.std = torch.std

    def forward(self,prediction,ground_truth):

        mean_gt = self.mean(ground_truth,0)

        mean_pred = self.mean(prediction,0)

        var_gt = self.var(ground_truth,0)

        var_pred = self.var(prediction,0)

        v_pred = prediction - mean_pred

        v_gt = ground_truth - mean_gt

        cor = self.sum (v_pred * v_gt) /  (self.sqrt(self.sum(v_pred **  2)) *  self.sqrt(self.sum(v_gt **  2)))

        sd_gt = self.std(ground_truth)

        sd_pred = self.std(prediction)

        numerator=2*cor*sd_gt*sd_pred

        denominator=var_gt+var_pred+(mean_gt-mean_pred)**2

        ccc = numerator/denominator

        return 1-ccc



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

class Wav2Vec2ClassificationHead(torch.nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = torch.nn.Dropout(config.final_dropout)
        self.out_proj = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@dataclasses.dataclass
class SpeechClassifierOutput(transformers.file_utils.ModelOutput):
    loss: typing.Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    
class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):

    def __init__(self, config):

        super().__init__(config)

        self.num_labels = config.num_labels
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.head = Wav2Vec2ClassificationHead(config)
        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def pooling(
            self,
            hidden_states,
            attention_mask,
    ):
        if attention_mask is None:   # For evaluation with batch_size==1
            outputs = torch.mean(hidden_states, dim=1)
        else:
            attention_mask = self._get_feature_vector_attention_mask(
                hidden_states.shape[1],
                attention_mask,
            )
            hidden_states = hidden_states * torch.reshape(
                attention_mask,
                (-1, attention_mask.shape[-1], 1),
            )
            outputs = torch.sum(hidden_states, dim=1)
            attention_sum = torch.sum(attention_mask, dim=1)
            outputs = outputs / torch.reshape(attention_sum,(-1,1))
        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states_framewise = outputs.last_hidden_state
        hidden_states = self.pooling(
            hidden_states_framewise,
            attention_mask,
        )
        logits = self.head(hidden_states)
        loss = None

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
        )


class DataCollatorCTCWithPadding:
    def __init__(
        self,
        processor: Wav2Vec2Processor,
    ):
        r"""Data collator that will dynamically pad the inputs received."""
        self.processor = processor

    def __call__(
        self,
        data: typing.List[typing.Dict[str, typing.Union[typing.List[int], torch.Tensor]]],
    ) -> typing.Dict[str, torch.Tensor]:

        segments = [d['input_values'] for d in data]
        speech_list = []
        sampling_rate = self.processor.feature_extractor.sampling_rate
        for file, start, end in segments:
            offset = float(start)
            duration = float(end) - offset
            signal, sr = audiofile.read(file, offset=offset, duration=duration)
            speech_list.append(signal)
        input_features = self.processor(speech_list, sampling_rate=sampling_rate)
        label_features = [d['labels'] for d in data]
        d_type = torch.long if isinstance(label_features[0], int) else torch.float
        batch = self.processor.pad(
            input_features,
            padding=True,
            return_tensors='pt',
        )
        batch['labels'] = torch.tensor(label_features, dtype=d_type)

        return batch

class CTCTrainer(Trainer):

    def __init__(
        self,
        criterion: torch.nn.Module,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.criterion = criterion

    def compute_loss(self, model, inputs, return_outputs=False):
        r"""Compute loss."""

        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs[0]
        loss = self.criterion(logits, labels)

        return (loss, outputs) if return_outputs else loss

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: typing.Dict[str, typing.Union[torch.Tensor, typing.Any]],
    ) -> torch.Tensor:
        r"""Perform a training step on a batch of inputs."""

        model.train()
        inputs = self._prepare_inputs(inputs)

        loss = self.compute_loss(model, inputs)
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        loss = loss.sum()
        loss.backward()
        # self.scaler.scale(loss).backward()

        return loss.detach()

def compute_metrics(p: EvalPrediction):

    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)

    scores = {
        name: metric(p.label_ids, preds) for name, metric in metrics.items()
    }

    return scores

def train_model(trainDataset: Dataset, testDataset: Dataset):
    r"""Train model."""
    # load model from local repo
    model_path = os.path.dirname(os.path.realpath(__file__))
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    print(model_path)
    # model = EmotionModel.from_pretrained(model_name)
    # Generate labels array from 0 to 1 with 0.01 step
    labels = ['arousal', 'valence']
    model = Wav2Vec2ForSpeechClassification.from_pretrained(
        model_path,
        config=transformers.AutoConfig.from_pretrained(
                model_path,
                num_labels=len(labels),
                label2id={label: i for i, label in enumerate(labels)},
                id2label={i: label for i, label in enumerate(labels)},
                finetuning_task='emotion',
                )    
    )

    file_path = os.path.realpath(os.path.join(
    os.getcwd(), os.path.dirname(__file__)))
    root = (os.path.dirname(os.path.dirname(file_path)))

    training_args = TrainingArguments(
        output_dir=root + "/data/jl/training/",
        logging_dir=log_root,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='epoch',
        num_train_epochs=10,
        save_steps=1,
        eval_steps=1,
        logging_steps=1,
        # fp16=True,
        learning_rate=1e-4,
        metric_for_best_model='CCC',
        save_total_limit=3,
        greater_is_better=True,
        load_best_model_at_end=True,
        ignore_data_skip=True,
        )

    model_name = os.path.dirname(os.path.realpath(__file__))
    processor = Wav2Vec2Processor.from_pretrained(model_name)

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    trainer = CTCTrainer(

        ConcordanceCorCoeff(),

        model=model,

        data_collator=data_collator,

        args=training_args,

        compute_metrics=compute_metrics,

        train_dataset=trainDataset,

        eval_dataset=testDataset,

        tokenizer=processor.feature_extractor,

        callbacks=[TensorBoardCallback()],
    )

    trainer.train()
    trainer.save_model(f'modelSave-{datetime.now().strftime("%H-%M_%d-%m-%Y")}' )
    pass
    # x = model.train(True)

    # load data from data loader
    
    # Load in a custom dataset to the huggingface dataset format
    # https://huggingface.co/docs/datasets/loading_datasets.html#loading-a-local-dataset
    # https://huggingface.co/docs/datasets/loading_datasets.html#dataset-dict

# def load_model():
#     # load model from local repo
#     model_path = os.path.dirname(os.path.realpath(__file__))
#     processor = Wav2Vec2Processor.from_pretrained(model_path)
#     print(model_path)
#     # model = EmotionModel.from_pretrained(model_name)
#     model = Wav2Vec2ForSpeechClassification.from_pretrained(
#         model_path,
#         config=transformers.AutoConfig.from_pretrained(
#                 model_path,
#                 num_labels=len(labels),
#                 label2id={label: i for i, label in enumerate(labels)},
#                 id2label={i: label for i, label in enumerate(labels)},
#                 finetuning_task='emotion',
#                 )    
#     )
#     # train_model(model)
#     return processor, model

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