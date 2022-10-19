import numpy as np
from model import train_model
from utils.jl import display_jl_quadrant_chart, load_jl_datasets_disk, move_jl_wav_files, test_jl, load_jl_results, transform_jl_dataset

# Finetune on JL
train_dataset, test_dataset = load_jl_datasets_disk()
train_model(train_dataset=train_dataset,
            test_dataset=test_dataset, dataset_Name="jl")

""" Example usage of how to use audEERING fine-tuned model"""
# print(process_func(signal_dummy, sampling_rate))
#  Arousal    dominance valence
# [[0.5460759 0.6062269 0.4043165]]

# process_func(signals, SAMPLING_RATE, embeddings=True)
# Pooled hidden states of last transformer layer
# [[-0.00752167  0.0065819  -0.00746339 ...  0.00663631  0.00848747
#   0.00599209]]
