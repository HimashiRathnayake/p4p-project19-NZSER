import dis
import numpy as np
from model import train_model
from utils.jl import display_jl_quadrant_chart, display_jl_quadrant_chart_sentence, load_jl_sentence, load_jl_wav_files, move_jl_wav_files, test_jl, load_jl_results, test_jl_sentence, transform_jl_dataset
from utils.recola import recola_dataset, test_recola
from utils.semaine import move_semaine_annotation_files, move_semaine_wav_files, test_semaine

# train_dataset, test_dataset = transform_jl_dataset()
# train_dataset, train_dataloader, test_dataset, test_dataloader = recola_dataset()
# processor, model = load_model()
# train_model(train_dataset, test_dataset)
# test_msp(processor, model)
# test_recola(processor, model)
# test_semaine(processor, model)
# test_jl(processor, model)
# test_jl_sentence(processor, model)
# load_jl_sentence()
# load_jl_results()
# display_jl_quadrant_chart(10,20)
# display_jl_quadrant_chart_sentence()
move_semaine_wav_files()
move_semaine_annotation_files()

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
