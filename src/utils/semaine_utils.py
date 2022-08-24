import os
import glob
import pandas as pd
import numpy as np
from librosa import load
from utils.metrics import ccc
from model import process_func
from utils.audio_utils import get_audio_chunks_semaine
from utils.display_utils import map_arrays_to_w2v

semaine_data = []
SAMPLING_RATE = 16000
SEMAINE_FRAME_SIZE = 40
''' 
One time use only.
'''
# Renames folders in semaine that dont have at least 2 text files inside with a "NA"


def prune_semaine():
    # Obtain root file paths
    file_path = os.path.realpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__)))
    root = os.path.dirname(os.path.dirname(file_path))
    sessions_path = root + "/data/semaine/sessions"

    # Iterate through folders for renaming
    for folder in os.listdir(sessions_path):
        if len(glob.glob(sessions_path + "/" + folder + "/*.txt")) < 2:
            os.rename(sessions_path + "/" + folder,
                      sessions_path + "/" + folder + "_NA")


def load_semaine():
    # Obtain root file paths
    file_path = os.path.realpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__)))
    root = os.path.dirname(os.path.dirname(file_path))
    sessions_path = root + "/data/semaine/sessions"
    annotations_path = root + "/data/semaine/TrainingInput"

    dfs = []
    signals = []

    # Iterate through TrainingInput folder for CSV annotations
    for folder in os.listdir(annotations_path):
        df = pd.read_csv(annotations_path + "/" + folder + '/TU_VA.csv')
        df.insert(0, 'bundle', folder)  # Add filename to dataframe
        dfs.append(df)

    # Iterate through session folders and load wav files for each session
    for folder in os.listdir(sessions_path):
        if not folder.endswith("_NA"):
            for file in glob.glob(sessions_path + "/" + folder + "/*.wav"):
                if (file.find("User HeadMounted") > 0):
                    signals.append(load(file, sr=SAMPLING_RATE))

    # Merge WAV files and their corresponding annotations
    for i, df in enumerate(dfs):
        semaine_data.append([df, signals[i]])
    print('Finished loading Semaine')
    return semaine_data


def test_semaine(processor, model):
    # Load the semaine dataset
    semaine_data = load_semaine()
    # Create lists for storing the true and predicted arousal and valence values
    true_aro = []
    pred_aro = []
    true_val = []
    pred_val = []
    # Iterate through the semaine dataset and store the true and predicted arousal and valence values

    for i, data in enumerate(semaine_data):
        print("Processing session " + str(i) + " out of " + str(len(semaine_data)))
        true_values = data[0]
        true_values = true_values.iloc[1::2] # Drop every alternate row as model has minimum input of 25 ms frames (Semaine is 20ms)
        signal = [[data[1][0]]]  # format for input to wav2vec

        chunks = get_audio_chunks_semaine(
            signal=signal, frame_size=SEMAINE_FRAME_SIZE, sampling_rate=SAMPLING_RATE)

        true_aro.append(true_values['Arousal'])
        true_val.append(true_values['Valence'])

        # Call map_arrays_to_w2v to map the true arousal and valence to the wav2vec model
        true_aro_mapped, true_val_mapped = map_arrays_to_w2v(
            true_aro, true_val)

        for j, chunk in enumerate(chunks):
            # Process the chunk and get the predicted arousal and valence values
            if j % 100 == 0:
                print("Processing chunk " + str(j) + " out of " + str(len(chunks)))
            chunk = np.array(chunk, dtype=np.float32)
            results = process_func(
                [[chunk]], SAMPLING_RATE, model=model, processor=processor)

            pred_aro.append(results[0][0])
            pred_val.append(results[0][2])

    # Calculate the CCC for arousal and valence and print it
    ccc_aro = ccc(np.array(true_aro_mapped), np.array(true_val_mapped))
    ccc_val = ccc(np.array(true_val), np.array(pred_val))
    print(f'CCC arousal: {ccc_aro:.2f}, CCC valence: {ccc_val:.2f}')
    return ccc_aro, ccc_val
