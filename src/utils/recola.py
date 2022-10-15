from datetime import datetime
import os
import glob
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from librosa import load
from datasets import load_dataset, Audio, Dataset
from torch.utils.data import DataLoader

from model import process_func
from utils.audio import get_audio_chunks_recola
from utils.calc import ccc, map_arrays_to_quadrant, map_arrays_to_w2v, pearson
from utils.display import quadrant_chart

# Constants
SAMPLING_RATE = 16000


def load_recola():
    """
    Load the RECOLA dataset and annotations.
    Returns: A list of tuples containing the audio file path and the annotations

    This function is called before testing the model on the RECOLA dataset.
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
    '''
    Test the model on the RECOLA dataset.
    The arguments should be the processor and model retrieved from the load_model() function.
    The arousal and valence predictions are stored in a txt file.
    The CCC and Pearsons values are stored in a csv file.

    The results files are saved in the path below:
    /[REPO_DIRECTORY]/results/recola_results/recola_results_[DATE_TIME].txt
    /[REPO_DIRECTORY]/results/recola_results/[FILENAME].csv

    Returns: The CCC for arousal as a list and the CCC for valence as a list
    '''
    file_path = os.path.realpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__)))
    root = os.path.dirname(os.path.dirname(file_path))
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
    with open(f'{root}/results/recola_results/recola_results - {datetime.now().strftime("%H-%M_%d-%m-%Y")}.txt', 'w', encoding='utf-8') as f:
        df = pd.DataFrame(columns=[
                          'bundle', 'pearson arousal', 'pearson valence', 'ccc arousal', 'ccc valence'])
        for i, data in enumerate(recola_data):
            f.write(
                f"True arousal,Predicted arousal,True valence,Predicted valence\n")
            print("Processing file " + str(i) +
                  " out of " + str(len(recola_data)))
            # Get the true arousal and valence values
            true_values = data[0]

            # Write the current filename to the results file
            f.write(f"{true_values['bundle'][0]}\n")

            # Average the true values from the dataframe
            true_values['aro_average'] = true_values[[
                'A1', 'A2', 'A3', 'A4', 'A5']].mean(axis=1)
            true_values['val_average'] = true_values[[
                'V1', 'V2', 'V3', 'V4', 'V5']].mean(axis=1)

            # Add true values to lists
            true_aro_mapped = []
            true_val_mapped = []
            true_aro_mapped, true_val_mapped = map_arrays_to_w2v(
                true_values['aro_average'], true_values['val_average'])
            true_aro.append(true_aro_mapped)
            true_val.append(true_val_mapped)

            signal = [[data[1][0]]]  # format for process func

            chunks = get_audio_chunks_recola(
                signal=signal, frame_size=40, sampling_rate=SAMPLING_RATE)

            # Store each files predicted arousal and valence
            file_pred_aro = []
            file_pred_val = []

            for j, chunk in enumerate(chunks):
                chunk = np.array(chunk, dtype=np.float32)
                results = process_func(
                    [[chunk]], SAMPLING_RATE, model=model, processor=processor)
                file_pred_aro.append(results[0][0])
                file_pred_val.append(results[0][2])
                # Write the true and predicted arousal and valence values to the results file
                f.write(
                    f"{true_aro_mapped[j]},{results[0][0]},{true_val_mapped[j]},{results[0][2]}\n")

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
            pearson_aro = pearson(
                np.array(true_aro[i]), np.array(file_pred_aro))
            pearson_val = pearson(
                np.array(true_val[i]), np.array(file_pred_val))

            # Concat ccc and pearson values to dataframe
            df = pd.concat([df, pd.DataFrame({'bundle': data[0]['bundle'][0], 'pearson arousal': pearson_aro, 'pearson valence': pearson_val,
                                              'ccc arousal': ccc_aro[i], 'ccc valence': ccc_val[i]}, index=[0])], ignore_index=True)

            print(
                f'Session {i} - CCC arousal: {ccc_aro[i]:.2f}, CCC valence: {ccc_val[i]:.2f}')
            print(
                f'Session {i} - Pearson arousal: {pearson_aro:.2f}, Pearson valence: {pearson_val:.2f}')

        filename = f'{root}/results/recola_results/recola_results_ - {datetime.now().strftime("%H-%M_%d-%m-%Y")}.csv'
        df.to_csv(filename, index=False)
    # Calculate the CCC and print it
    print(
        f'Avg. Semaine Acc. per session - CCC arousal: {np.mean(ccc_aro):.2f}, CCC valence: {np.mean(ccc_val):.2f}')
    return ccc_aro, ccc_val


def load_recola_results():
    '''
    Loads the results of the RECOLA model for the specified txt file from the following directory:
    [REPO_DIRECTORY]/results/recola_results/

    Returns: A dataframe containing the results of the RECOLA model

    This function is called when creating the quadrant charts for the RECOLA corpus
    '''
    # Load in the results of the recola corpus txt file
    recola_results = []
    file_results = []
    file_path = os.path.realpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__)))
    root = os.path.dirname(os.path.dirname(file_path))
    recola_results_path = root + '/recola_results/recola_results.txt'
    with open(recola_results_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Skip the first line
            if line == 'True arousal,Predicted arousal,True valence,Predicted valence\n':
                continue
            # Store the results of each prev file into a pandas dataframe
            if line.startswith('P'):
                # if this is the first file then skip
                if len(file_results) != 0:
                    df = pd.DataFrame(file_results, columns=[
                                      'true_aro', 'pred_aro', 'true_val', 'pred_val'])
                    recola_results.append(df)
                    file_results = []
                continue
            else:
                # Remove the newline character from the end of the line and append to file_results
                list_results = list(map(float, line.strip().split(',')))
                file_results.append(list_results)

    return recola_results


def display_recola_quadrant_chart(start_index: int, end_index: int):
    '''
    Display the quadrant chart for the RECOLA corpus for the given start and end file numbers
    Note max end index is 150 as there are 150 files for each speaker in the RECOLA corpus
    The chart is saved as a png in the following directory:
    [REPO_DIRECTORY]/results/recola_results/recola_plts/
    '''
    # Load in the results of the JL-corpus test text file
    recola_results = load_recola_results()
    file_path = os.path.realpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__)))
    root = os.path.dirname(os.path.dirname(file_path))
    recola_plots = root + '/recola_results/recola_plts'

    if(start_index < 0 or start_index >= len(recola_results)):
        print('Invalid start index')
        return

    if(end_index < start_index or end_index >= len(recola_results)):
        print('Invalid end index')
        return

    # Iterate through each file and calculate the arousal and valence values
    for i in range(start_index, end_index + 1):  # Python exclusive of end index

        results = recola_results[i]
        # Retrieve true arousal and valence values from dataframe
        true_aro = results['true_aro'].values.tolist()
        true_val = results['true_val'].values.tolist()

        # Retrieve predicted arousal and valence values from dataframe
        pred_aro = results['pred_aro'].values.tolist()
        pred_val = results['pred_val'].values.tolist()

        # Map true arousal and valence values from [0, 1] -> [-1, 1]
        pred_aro, pred_val = map_arrays_to_quadrant(pred_aro, pred_val)
        true_aro, true_val = map_arrays_to_quadrant(true_aro, true_val)

        # Take values from position 90 to 110
        true_aro = true_aro[90:110]
        true_val = true_val[90:110]
        pred_aro = pred_aro[90:110]
        pred_val = pred_val[90:110]

        # Plot the results using the quadrant chart function
        quadrant_chart(pred_val, pred_aro, true_val, true_aro,)
        # plt.figure(i)
        plt.title('Arousal vs Valence', fontsize=16)
        plt.ylabel('Arousal', fontsize=14)
        plt.xlabel('Valence', fontsize=14)
        plt.grid(True, animated=True, linestyle='--', alpha=0.5)
        # Save the plot as a png file with the index of the file as the filename
        plt.savefig(f'{recola_plots}/recola_results_{i}.png')
        plt.close()


def recola_dataset():
    '''
    This function is now deprecated.
    Initial attempt at transforming the RECOLA dataset into a format that can be used by the model.
    '''
    # Create dictionary for loading in Recola datasets for load_dataset

    # Find all Recola audio file names
    file_path = os.path.realpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__)))
    root = (os.path.dirname(os.path.dirname(file_path)))
    print(root)
    wav_files = glob.glob(root + "/data/recola/RECOLA-Audio-recordings/*.wav")

    # Create dictionary
    audio_paths = []
    for i, wav_file in enumerate(wav_files):
        audio_paths.append(wav_file)

    # Load in the Recola annotations
    print("Loading Recola annotations")
    recola_data = load_recola()
    print("Finished loading Recola annotations")
    train_dict = {
        'audio': [],
        'arousal': [],
        'valence': []
    }
    test_dict = {
        'audio': [],
        'arousal': [],
        'valence': []
    }

    # Add the recola annotations to train_dict and test_dict
    for i in range(0, 18):
        true_values = recola_data[i][0]

        # Average the true values from the dataframe
        true_aro_avg = true_values[['A1', 'A2', 'A3', 'A4', 'A5']].mean(
            axis=1).to_numpy()
        true_val_avg = true_values[['V1', 'V2', 'V3', 'V4', 'V5']].mean(
            axis=1).to_numpy()

        # Add true values to lists
        train_dict['arousal'].append(true_aro_avg)
        train_dict['valence'].append(true_val_avg)
        train_dict['audio'].append(audio_paths[i])

    for i in range(18, 23):
        true_values = recola_data[i][0]

        # Average the true values from the dataframe
        true_aro_avg = true_values[['A1', 'A2', 'A3', 'A4', 'A5']].mean(
            axis=1).to_numpy()
        true_val_avg = true_values[['V1', 'V2', 'V3', 'V4', 'V5']].mean(
            axis=1).to_numpy()

        # Add true values to lists
        test_dict['arousal'].append(true_aro_avg)
        test_dict['valence'].append(true_val_avg)
        test_dict['audio'].append(audio_paths[i])

    train_audio_dataset = Dataset.from_dict(train_dict).with_format(
        "torch").cast_column('audio', Audio(sampling_rate=SAMPLING_RATE))
    train_audio_dataloader = DataLoader(
        train_audio_dataset, batch_size=1, shuffle=False)
    test_audio_dataset = Dataset.from_dict(test_dict).with_format(
        "torch").cast_column('audio', Audio(sampling_rate=SAMPLING_RATE))
    test_audio_dataloader = DataLoader(
        test_audio_dataset, batch_size=1, shuffle=False)

    return train_audio_dataset, train_audio_dataloader, test_audio_dataset, test_audio_dataloader
