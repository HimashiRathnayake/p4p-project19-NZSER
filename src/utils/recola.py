import os
import glob
import numpy as np
import pandas as pd
from librosa import load

from model import process_func
from utils.audio import get_audio_chunks_recola
from utils.calc import ccc, map_arrays_to_w2v

# Constants
SAMPLING_RATE = 16000


def load_recola():
    """
    Load the Recola dataset and annotations
    """
    file_path = os.path.realpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__)))
    root = os.path.dirname(os.path.dirname(file_path))

    # Load csv files with annotations
    aro_csv_files = glob.glob(
        root + "/data/recola/RECOLA-Annotation/emotional_behaviour/arousal/*.csv")
    val_csv_files = glob.glob(
        root + "/data/recola/RECOLA-Annotation/emotional_behaviour/valence/*.csv")

    aro_dfs = []
    for csv_file in aro_csv_files:
        aro_df = pd.read_csv(csv_file, delimiter=';')

        aro_df.insert(0, 'bundle', os.path.basename(
            csv_file).split('.')[0])  # Add filename to dataframe
        # Drop FF3 annotator due to lack of range
        aro_df = aro_df.drop('FF3', axis=1)
        # Rename columns for arousal
        aro_df = aro_df.rename(
            columns={'FM1 ': 'A1', 'FM2 ': 'A2', 'FM3 ': 'A3', 'FF1 ': 'A4', 'FF2 ': 'A5'})

        aro_dfs.append(aro_df)

    # Load valence csv files and concatentate arousal and valence dataframes
    mer_dfs = []
    for i, csv_file in enumerate(val_csv_files):
        val_df = pd.read_csv(csv_file, delimiter=';')

        val_df = val_df.drop('FF3', axis=1)
        # Rename columns for valence
        val_df = val_df.rename(
            columns={'FM1 ': 'V1', 'FM2 ': 'V2', 'FM3 ': 'V3', 'FF1 ': 'V4', 'FF2 ': 'V5'})

        mer_dfs.append(pd.concat([aro_dfs[i], val_df], axis=1))

    # Load wav files
    wav_files = glob.glob(root + "/data/recola/RECOLA-Audio-recordings/*.wav")

    # Merge wav files and annotations
    recola_data = []
    for i, wav_file in enumerate(wav_files):
        sig = load(wav_file, sr=SAMPLING_RATE)
        recola_data.append([mer_dfs[i], sig])
    print("Finished loading Recola data")
    return recola_data


def test_recola(processor, model):
    # Load the recola dataset
    recola_data = load_recola()

    # Create lists for storing the true and predicted arousal and valence values
    true_aro = []
    pred_aro = []
    true_val = []
    pred_val = []

    # Create lists for storing the CCC values
    ccc_aro = []
    ccc_val = []

    # Feed the recola data to the model generate predictions and store them in the lists

    for i, data in enumerate(recola_data):
        print("Processing file " + str(i) + " out of " + str(len(recola_data)))
        # Get the true arousal and valence values
        true_values = data[0]

        # Averaeg the true values from the dataframe
        true_values['aro_average'] = true_values[[ 'A1', 'A2', 'A3', 'A4', 'A5']].mean(axis=1)
        true_values['val_average'] = true_values[[ 'V1', 'V2', 'V3', 'V4', 'V5']].mean(axis=1)

        # Add true values to lists
        true_aro_mapped = []
        true_val_mapped = []
        true_aro_mapped, true_val_mapped = map_arrays_to_w2v(true_values['aro_average'], true_values['val_average'])
        true_aro.append(true_aro_mapped)
        true_val.append(true_val_mapped)

        signal = [[data[1][0]]]  # format for process func

        chunks = get_audio_chunks_recola(
            signal=signal, frame_size=40, sampling_rate=SAMPLING_RATE)

        # Store each files predicted arousal and valence
        file_pred_aro = []
        file_pred_val = []

        for chunk in chunks:
            chunk = np.array(chunk, dtype=np.float32)
            results = process_func([[chunk]], SAMPLING_RATE, model=model, processor=processor)
            file_pred_aro.append(results[0][0])
            file_pred_val.append(results[0][2])

        min_array_length = min(len(true_aro[i]), len(file_pred_aro))
        file_pred_aro = file_pred_aro[:min_array_length]
        file_pred_val = file_pred_val[:min_array_length]
        true_aro[i] = true_aro[i][:min_array_length]
        true_val[i] = true_val[i][:min_array_length]

        pred_aro.append(file_pred_aro)
        pred_val.append(file_pred_val)

        # Calculate the CCC for the predicted arousal and valence values
        ccc_aro.append(ccc(np.array(true_aro[i]), np.array(file_pred_aro)))
        ccc_val.append(ccc(np.array(true_val[i]), np.array(file_pred_val)))
        print(f'Session {i} - CCC arousal: {ccc_aro[i]:.2f}, CCC valence: {ccc_val[i]:.2f}')

    # Calculate the CCC and print it
    print(f'Avg. Semaine Acc. per session - CCC arousal: {np.mean(ccc_aro):.2f}, CCC valence: {np.mean(ccc_val):.2f}')
    return ccc_aro, ccc_val
