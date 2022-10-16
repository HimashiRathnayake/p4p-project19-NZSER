import dis
import numpy as np
from model import train_model
from utils.jl import display_jl_quadrant_chart, move_jl_wav_files, test_jl, load_jl_results, transform_jl_dataset
from utils.recola import display_recola_quadrant_chart, recola_dataset, test_recola
from utils.semaine import interpolate_semaine_annotations, load_semaine_datasets, map_semaine_annotations, plot_semaine_results, test_semaine, transform_semaine_dataset, move_semaine_annotation_files, move_semaine_wav_files

# train_dataset, test_dataset = transform_jl_dataset()
# move_jl_wav_files()

# Finetune semaine and then finetune on JL
# trainDataset, testDataset = load_jl_datasets_disk()
# trainDataset, testDataset = load_semaine_datasets()
# train_model(trainDataset=trainDataset, testDataset=testDataset, datasetName="jl")

# interpolate_semaine_annotations()
# display_recola_quadrant_chart(0, 21)

plot_semaine_results(0, 2)
# processor, model = load_model()
# train_model(train_dataset, test_dataset)

# test_msp(processor, model)
# test_recola(processor, model)
# test_semaine(processor, model)

# test_jl(processor, model)
# test_jl_sentence(processor, model)
# load_jl_sentence()
# load_jl_results()

# display_jl_quadrant_chart('jl_results_f1', 10, 20)
# display_jl_quadrant_chart_sentence()

# print(process_func(signal_dummy, sampling_rate))
#  Arousal    dominance valence
# [[0.5460759 0.6062269 0.4043165]]

# process_func(signals, SAMPLING_RATE, embeddings=True)
# Pooled hidden states of last transformer layer
# [[-0.00752167  0.0065819  -0.00746339 ...  0.00663631  0.00848747
#   0.00599209]]
