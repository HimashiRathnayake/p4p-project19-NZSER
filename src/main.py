import numpy as np

import glob
from utils.display_utils import map_w2v_to_quadrant
from utils.jl_utils import load_jl
from utils.msp_utils import load_msp, test_msp
from utils.recola_utils import load_recola
from utils.metrics import ccc
from scipy.io.wavfile import read
from librosa import load

from utils.semaine_utils import load_semaine, prune_semaine

# load_jl()
# load_recola()
# load_semaine()
# load_msp()
test_msp()

# Create lists for storing annotations
true_val = []
true_aro = []

pred_val = []
pred_aro = []

# for i, file in enumerate(files):
#     current_signal = load(file, sr=SAMPLING_RATE)
#     current_signal = [current_signal[0]]
#     signals[i].append(current_signal)
    # chunks = get_audio_chunks(
    #     signal=current_signal[0], frame_size=200, sampling_rate=16000, is_jl=False)
    # for chunk in chunks:
    #     chunk = [[np.array(chunk, dtype=np.float32)]]
    #     result = process_func(chunk, SAMPLING_RATE)
    #     result[0][0], result[0][2], result[0][1] = map_w2v_to_quadrant(
    #         result[0][0], result[0][2], result[0][1])
    #     pred_aro.append(result[0][0])
    #     pred_val.append(result[0][2])
    # break

# true_val = [-0.6, -0.6, -0.62, -0.62, -0.58, -0.58, -0.6, -0.6]
# true_aro = [0.86, 0.84, 0.84, 0.85, 0.8, 0.81, 0.85, 0.85]

# true_val = np.array(true_val)
# true_aro = np.array(true_aro)
# pred_val = np.array(pred_val)
# pred_aro = np.array(pred_aro)

# ccc_val = ccc(pred_val, true_val)
# ccc_aro = ccc(pred_aro, true_aro)
# print("Pred val:", pred_val)
# print("Pred aro:", pred_aro)
# print(f'CCC for arousal: {ccc_aro} \nCCC for valence: {ccc_val}')


# results = [[] for i in range(len(files))]

# # process loaded signals
# for j, signal in enumerate(signals):
#     results[j].append(process_func(signal, SAMPLING_RATE))
#     print(results[j][0])


# print(process_func(signal_dummy, sampling_rate))
#  Arousal    dominance valence
# [[0.5460759 0.6062269 0.4043165]]

# process_func(signals, SAMPLING_RATE, embeddings=True)
# Pooled hidden states of last transformer layer
# [[-0.00752167  0.0065819  -0.00746339 ...  0.00663631  0.00848747
#   0.00599209]]
