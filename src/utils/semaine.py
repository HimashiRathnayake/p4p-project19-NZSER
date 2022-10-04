import os
import glob
import pandas as pd
import numpy as np
from librosa import load
from model import process_func
from utils.audio import get_audio_chunks_semaine
from utils.calc import ccc, map_arrays_to_w2v, pearson
from datetime import datetime


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

    dbgO1 = []
    dbgO2 = []

    # Iterate through TrainingInput folder for CSV annotations
    for folder in os.listdir(annotations_path):
        df = pd.read_csv(annotations_path + "/" + folder + '/TU_VA.csv')
        df.insert(0, 'bundle', folder)  # Add filename to dataframe
        dfs.append(df)
        dbgO1.append(folder)

    # Iterate through session folders and load wav files for each session
    for folder in os.listdir(sessions_path):
        if not folder.endswith("_NA"):
            dbgO2.append(folder)
            for file in glob.glob(sessions_path + "/" + folder + "/*.wav"):
                if (file.find("User HeadMounted") > 0):
                    signals.append(load(file, sr=SAMPLING_RATE))

    print(dbgO1)
    print(dbgO2)
    # Merge WAV files and their corresponding annotations
    for i, df in enumerate(dfs):
        semaine_data.append([df, signals[i]])
    print('Finished loading Semaine')
    return semaine_data


def test_semaine(processor, model):
    # Load the semaine dataset
    semaine_data = load_semaine()

    # Create lists for storing the true and predicted arousal and valence values
    true_aro_mapped = []
    pred_aro = []
    true_val_mapped = []
    pred_val = []

    # Create lists for storing CCC values of each session
    ccc_aro = []
    ccc_val = []
    pearson_aro = []
    pearson_val = []
    with open('semaine_results.txt', 'w', encoding='utf-8') as f:
        f.write(f"True arousal,Predicted arousal,True valence,Predicted valence\n")
        df = pd.DataFrame(columns=[
                          'bundle', 'pearson arousal', 'pearson valence', 'ccc arousal', 'ccc valence'])
        # Iterate through the semaine dataset and store the true and predicted arousal and valence values
        for i, data in enumerate(semaine_data):
            print("Processing session " + str(i) +
                  " out of " + str(len(semaine_data)))
            # Write the current file name to the results file
            f.write(f"Session {i}\n")

            # Get the true arousal and valence values
            true_values = data[0]
            # Find average of every two frames as model has minimum input of 25 ms frames (Semaine frame is 20ms)
            true_values = true_values.groupby(
                np.arange(len(true_values))//2).mean()

            true_aro = true_values['Arousal']
            true_val = true_values['Valence']

            # Call map_arrays_to_w2v to map the true arousal and valence to the wav2vec model
            curr_true_aro_mapped, curr_true_val_mapped = map_arrays_to_w2v(
                true_aro, true_val)
            true_aro_mapped.append(curr_true_aro_mapped)
            true_val_mapped.append(curr_true_val_mapped)

            # Load the audio file and process it
            signal = [[data[1][0]]]  # format for input to wav2vec
            chunks = get_audio_chunks_semaine(
                signal=signal, frame_size=SEMAINE_FRAME_SIZE, sampling_rate=SAMPLING_RATE)

            # Store each session's predicted arousal and valence values
            sess_pred_aro = []
            sess_pred_val = []
            for j, chunk in enumerate(chunks):
                # Process the chunk and get the predicted arousal and valence values
                if j % 100 == 0:
                    print("Processing chunk " + str(j) +
                          " out of " + str(len(chunks)))
                chunk = np.array(chunk, dtype=np.float32)
                results = process_func(
                    [[chunk]], SAMPLING_RATE, model=model, processor=processor)
                sess_pred_aro.append(results[0][0])
                sess_pred_val.append(results[0][2])
                if j >= len(true_aro_mapped[0]):
                    break
                f.write(
                    f"{true_aro_mapped[0][j]},{results[0][0]},{true_val_mapped[0][j]},{results[0][2]}\n")

            # Append the true and predicted arousal and valence values to the lists after matching the size of the lists
            min_array_len = min(len(true_aro_mapped[i]), len(sess_pred_aro))
            sess_pred_aro = sess_pred_aro[:min_array_len]
            sess_pred_val = sess_pred_val[:min_array_len]
            true_aro_mapped[i] = true_aro_mapped[i][:min_array_len]
            true_val_mapped[i] = true_val_mapped[i][:min_array_len]
            pred_aro.append(sess_pred_aro)
            pred_val.append(sess_pred_val)

            # Calculate the CCC for the predicted arousal and valence values
            ccc_aro.append(
                ccc(np.array(true_aro_mapped[i]), np.array(pred_aro[i])))
            ccc_val.append(
                ccc(np.array(true_val_mapped[i]), np.array(pred_val[i])))
            pearson_aro.append(
                pearson(np.array(true_aro_mapped[i]), np.array(pred_aro[i])))
            pearson_val.append(
                pearson(np.array(true_val_mapped[i]), np.array(pred_val[i])))
            print(
                f'Session {i} - CCC arousal: {ccc_aro[i]:.2f}, CCC valence: {ccc_val[i]:.2f}')
            print(
                f'Session {i} - Pearson arousal: {pearson_aro[i]:.2f}, Pearson valence: {pearson_val[i]:.2f}')

            # Concat ccc and pearson values of each file to dataframe
            df = df.append({'bundle': data[0]['bundle'][0], 'pearson arousal': pearson_aro[i],
                           'pearson valence': pearson_val[i], 'ccc arousal': ccc_aro[i], 'ccc valence': ccc_val[i]}, ignore_index=True)
            # Print the average ccc and pearsons accuracy for each bundle of the dataframe and save to the results file
            print(df.groupby('bundle').mean())
            f.write(
                f"Session {i} average accuracy:\n{df.groupby('bundle').mean()}\n")

        # Calculate the CCC for arousal and valence and print it
        # print(
        #     f'Avg. Semaine Acc. per session - CCC arousal: {np.mean(ccc_aro):.2f}, CCC valence: {np.mean(ccc_val):.2f}')
        # print(
        #     f'Avg. Semaine Acc. per session - Pearson arousal: {np.mean(pearson_aro):.2f}, Pearson valence: {np.mean(pearson_val):.2f}')

        filename = f'seamine_results_ - {datetime.now().strftime("%H-%M_%d-%m-%Y")}.csv'
        df.to_csv(filename, index=False)

    return ccc_aro, ccc_val
