import os
import glob
import shutil
import pandas as pd
import numpy as np
from librosa import load
from model import process_func
from utils.audio import get_audio_chunks_semaine
from utils.calc import ccc, map_arrays_to_w2v, pearson, map_arrays_to_quadrant
from datetime import datetime
import matplotlib.pyplot as plt

from utils.display import quadrant_chart


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
            df = pd.concat([df, pd.DataFrame({'bundle': data[0]['bundle'][0], 'pearson arousal': pearson_aro[i], 'pearson valence': pearson_val[i],
                                              'ccc arousal': ccc_aro[i], 'ccc valence': ccc_val[i]}, index=[0])], ignore_index=True)

            # Print the average ccc and pearsons accuracy for each bundle of the dataframe and save to the results file
            print(df.groupby('bundle').mean())
        # Calculate the CCC for arousal and valence and print it
        # print(
        #     f'Avg. Semaine Acc. per session - CCC arousal: {np.mean(ccc_aro):.2f}, CCC valence: {np.mean(ccc_val):.2f}')
        # print(
        #     f'Avg. Semaine Acc. per session - Pearson arousal: {np.mean(pearson_aro):.2f}, Pearson valence: {np.mean(pearson_val):.2f}')

        filename = f'semaine_results_ - {datetime.now().strftime("%H-%M_%d-%m-%Y")}.csv'
        df.to_csv(filename, index=False)

    return ccc_aro, ccc_val


def load_semaine_results():
    # Load in the results of the semaine corpus txt file
    semaine_results = []
    file_results = []

    with open('semaine_results.txt', 'r', encoding='utf-8') as f:
        for line in f:
            # Skip the first line

            if line == 'True arousal,Predicted arousal,True valence,Predicted valence\n':
                continue

            # Store the results of each prev file into a pandas dataframe
            if line.startswith('Session'):
                # if this is the first file then skip
                if len(file_results) != 0:
                    df = pd.DataFrame(file_results, columns=[
                                      'true_aro', 'pred_aro', 'true_val', 'pred_val'])
                    semaine_results.append(df)
                    file_results = []
                continue
            else:
                # Remove the newline character from the end of the line and append to file_results
                list_results = list(map(float, line.strip().split(',')))
                file_results.append(list_results)

    return semaine_results


def plot_semaine_results(start_index: int, end_index: int):
    # Load in the results of the semaine corpus txt file
    semaine_results = load_semaine_results()
    file_path = os.path.realpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__)))

    if(start_index < 0 or start_index >= len(semaine_results)):
        print('Invalid start index')
        return

    if(end_index < start_index or end_index >= len(semaine_results)):
        print('Invalid end index')
        return

    # Iterate through each file and calculate the arousal and valence values
    for i in range(start_index, end_index + 1):

        results = semaine_results[i]
        # Retrieve true arousal and valence values from dataframe
        true_aro = results['true_aro'].values.tolist()
        true_val = results['true_val'].values.tolist()

        # Retrieve predicted arousal and valence values from dataframe
        pred_aro = results['pred_aro'].values.tolist()
        pred_val = results['pred_val'].values.tolist()

        # Map true arousal and valence values from [0, 1] -> [-1, 1]
        pred_aro, pred_val = map_arrays_to_quadrant(pred_aro, pred_val)
        true_aro, true_val = map_arrays_to_quadrant(true_aro, true_val)

        # Take every 500th value of the true and predicted arousal and valence values
        true_aro = true_aro[::500]
        true_val = true_val[::500]
        pred_aro = pred_aro[::500]
        pred_val = pred_val[::500]

        # Plot the results using the quadrant chart function
        quadrant_chart(pred_val, pred_aro, true_val, true_aro)
        plt.figure(i)
        plt.title('Arousal vs Valence', fontsize=16)
        plt.ylabel('Arousal', fontsize=14)
        plt.xlabel('Valence', fontsize=14)
        plt.grid(True, animated=True, linestyle='--', alpha=0.5)
        # Save the plot as a png file with the index of the file as the filename
        plt.savefig(f'{file_path}/semaine_plts/semaine_results_{i}.png')


# Move all semaine wav files into a single folder and name them to the session number
def move_semaine_wav_files():
    file_path = os.path.realpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__)))
    root = os.path.dirname(os.path.dirname(file_path))
    sessions_path = root + "/data/semaine/sessions"
    destination_path = root + "/data/semaine/semaine_all_files/"

    # Create a folder to store all the wav files
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    # Iterate through session folders and load wav files for each session
    for folder in os.listdir(sessions_path):
        if not folder.endswith("_NA"):
            for file in glob.glob(sessions_path + "/" + folder + "/*.wav"):
                if (file.find("User HeadMounted") > 0):
                    # Rename the file to have the session number before copying
                    new_file_name = folder + ".wav"
                    shutil.copy(file, destination_path)
                    # Rename file to the new file name
                    os.rename(destination_path + os.path.basename(file),
                              destination_path + new_file_name)


# Move semaine annotatoin files into a single folder and name them to the session number
def move_semaine_annotation_files():
    file_path = os.path.realpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__)))
    root = os.path.dirname(os.path.dirname(file_path))
    annotations_path = root + "/data/semaine/TrainingInput"
    destination_path = root + "/data/semaine/semaine_all_files/"

    # Create a folder to store all the annotation files
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    # Iterate through TrainingInput folder for CSV annotations
    for folder in os.listdir(annotations_path):
        file_name = annotations_path + "/" + folder + '/TU_VA.csv'
        new_file_name = folder + ".csv"
        shutil.copy(file_name, destination_path)
        # Rename file to the new file name
        os.rename(destination_path + os.path.basename(file_name),
                  destination_path + new_file_name)

# Iterates through semaine annotations and creates new mapped columns for arousal and valence, and adds start end times
def map_semaine_annotations():
    file_path = os.path.realpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__)))
    root = os.path.dirname(os.path.dirname(file_path))
    annotations_path = root + "/data/semaine/semaine_all_files/"
    # Create a folder to store all the annotation files
    if not os.path.exists(annotations_path):
        os.makedirs(annotations_path)

    # Iterate through TrainingInput folder for CSV annotations
    for folder in os.listdir(annotations_path):
        if folder.endswith(".csv"):
            df = pd.read_csv(annotations_path + folder)
            df['arousal_mapped'] = df['Arousal'].apply(
                lambda x: map_annotation(x))
            df['valence_mapped'] = df['Valence'].apply(
                lambda x: map_annotation(x))
            df['start'] = df['Time']
            df['end'] = df['Time'] + 0.02
            df.to_csv(annotations_path + folder, index=False)
