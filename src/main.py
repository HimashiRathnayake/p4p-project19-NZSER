import numpy as np
from model import load_model, train_model
from utils.jl import test_jl, load_jl_results
from utils.recola import recola_dataset, test_recola
from utils.semaine import test_semaine

# x = recola_dataset()
processor, model = load_model()
# train_model(model)
# test_msp(processor, model)
# test_recola(processor, model)
# test_semaine(processor, model)
test_jl(processor, model)
load_jl_results()

# Create lists for storing annotations
true_val = []
true_aro = []

pred_val = []
pred_aro = []

# print(process_func(signal_dummy, sampling_rate))
#  Arousal    dominance valence
# [[0.5460759 0.6062269 0.4043165]]

# process_func(signals, SAMPLING_RATE, embeddings=True)
# Pooled hidden states of last transformer layer
# [[-0.00752167  0.0065819  -0.00746339 ...  0.00663631  0.00848747
#   0.00599209]]
