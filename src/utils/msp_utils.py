import sys
import os
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from librosa import load
from model import process_func, load_model
from utils.display_utils import map_msp_to_quadrant
from utils.metrics import ccc

SAMPLING_RATE = 16000
msp_data = []

def load_msp() -> list[pd.DataFrame, np.ndarray]:
    # Obtain root file paths
    file_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    root = os.path.dirname(os.path.dirname(file_path))
    msp_path = root + "/data/msp"
    msp_data = []
    
    # Load csv file with annotations
    full_df = pd.read_csv(msp_path + "/labels_concensus.csv", delimiter=',')
    full_df = full_df.drop(['EmoClass', 'EmoDom', 'SpkrID', 'Gender', 'Split_Set'], axis=1)

    full_list_annot = full_df.values.tolist()
    # Merge list of annotations and wav files in list
    # for row in full_list_annot:
    for i in range(0, 19):
        # wav_file = load(msp_path + '/' + row[0], sr=SAMPLING_RATE)
        wav_file = load(msp_path + '/' + full_list_annot[i][0], sr=SAMPLING_RATE)
        # msp_data.append([row ,wav_file])
        msp_data.append([full_list_annot[i] ,wav_file])

    
    # Return merged list of annotations and wav files
    return msp_data

def test_msp(processor, model):
    # Load msp data
    msp_data = load_msp()
    load_model()

    # Create lists for storing annotations
    true_val = []
    true_aro = []
    pred_val = []
    pred_aro = []
    # Feed msp data to model and generate predictions
    for i, clip in enumerate(msp_data):
        pred_vals = process_func([[clip[1][0]]], sampling_rate=SAMPLING_RATE, model=model, processor=processor)
        mapped_true_vals = map_msp_to_quadrant(clip[0][1], clip[0][2])
        # print(f'Filename: {clip[0][0]}\nPredicted arousal: {pred_vals[0][0]:.2f}, Predicted valence: {pred_vals[0][2]:.2f}')
        # print(f'True arousal: {mapped_true_vals[0]:.2f}, True valence: {mapped_true_vals[1]:.2f}')
        true_val.append(mapped_true_vals[1])
        true_aro.append(mapped_true_vals[0])
        pred_val.append(pred_vals[0][2])
        pred_aro.append(pred_vals[0][0])
    # Calculate CCC for arousal and valence
    ccc_aro = ccc(np.array(true_aro), np.array(pred_aro))
    ccc_val = ccc(np.array(true_val), np.array(pred_val))
    print(f'CCC arousal: {ccc_aro:.2f}, CCC valence: {ccc_val:.2f}')
    return ccc_aro, ccc_val