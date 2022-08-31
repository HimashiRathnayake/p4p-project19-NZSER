import os
import glob
import pandas as pd
import numpy as np
from librosa import load
from typing import List
from model import process_func

from utils.audio_utils import get_audio_chunks_jl
from utils.metrics import ccc

# Constants
SAMPLING_RATE = 16000
JL_FRAME_SIZE = 200

# f1 = female1, m2 = male2, df = dataframe
f1 = [] # Contains wav files and annotations for female1
m2 = [] # Contains wav files and annotations for male2
f1_dfs = []
m2_dfs = []

def load_jl() -> List[List]:
    """
    Load the JL-Corpus dataset and annotations.
    Returns: List of lists containing female speaker 1's annotations and wav files.
    Returns: List of lists containing male speaker 2's annotations and wav files.
    """
    f1_dfs = []
    m2_dfs = []
    file_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    root = os.path.dirname(os.path.dirname(file_path))

    # Load csv files with annotations
    csv_files = glob.glob(root + "/data/jl/*.csv")
    
    # f1 = female1, m2 = male2, df = dataframe
    f1_aro_df = pd.read_csv(csv_files[0])
    f1_val_df = pd.read_csv(csv_files[1])
    m2_aro_df = pd.read_csv(csv_files[2])
    m2_val_df = pd.read_csv(csv_files[3])
    
    f1_aro_df = f1_aro_df[['bundle', 'labels']]
    f1_aro_df = f1_aro_df.rename(columns = {'labels': 'arousal'})
    f1_val_df = f1_val_df['labels']
    f1_val_df = f1_val_df.rename('valence')

    m2_aro_df = m2_aro_df[['bundle', 'labels']]
    m2_aro_df = m2_aro_df.rename(columns = {'labels': 'arousal'})
    m2_val_df = m2_val_df['labels']
    m2_val_df = m2_val_df.rename('valence')

    f1_mer_df = pd.concat([f1_aro_df, f1_val_df], axis=1)
    m2_mer_df = pd.concat([m2_aro_df, m2_val_df], axis=1)
    # Remove duplicate bundle names and store in list
    f1_bdls = list(dict.fromkeys(f1_mer_df['bundle'].tolist()))
    m2_bdls = list(dict.fromkeys(m2_mer_df['bundle'].tolist()))

    for bdl in f1_bdls:
        bdl_df = f1_mer_df[f1_mer_df['bundle'] == bdl]
        f1_dfs.append(bdl_df)
        
    for bdl in m2_bdls:
        bdl_df = m2_mer_df[m2_mer_df['bundle'] == bdl]
        m2_dfs.append(bdl_df)
    print('Finished loading JL-corpus CSV files.')

    for df in f1_dfs:
        sig = load(root + "/data/jl/female1_all_a_1/" + df.iloc[0]['bundle'] + '_bndl/' + df.iloc[0]['bundle'] + ".wav", sr=SAMPLING_RATE)
        f1.append([df, sig])

    for df in m2_dfs:
        sig = load(root + "/data/jl/male2_all_a_1/" + df.iloc[0]['bundle'] + '_bndl/' + df.iloc[0]['bundle'] + ".wav", sr=SAMPLING_RATE)
        m2.append([df, sig])
    print('Finished loading JL-corpus WAV files.')

    return f1, m2


    
def test_jl(processor, model):
    f1, m2 = load_jl()

    # Create lists for storing the CCC values of each wav file
    f1_aro_ccc = []
    f1_val_ccc = []
    m2_aro_ccc = []
    m2_val_ccc = []

    # Iterate through each wav file and calculate the arousal and valence values
    for i in range(len(f1)):
        print('Processing wav file ' + str(i + 1) + ' of ' + str(len(f1)))
        true_values = f1[i][0]
        signal = f1[i][1]

        # Retrieve true arousal and valence values from dataframe
        true_aro = true_values['arousal']
        true_val = true_values['valence']

        # Map true arousal and valence values from [-1, 1] -> [0, 1]
        true_aro = (true_aro + 1) / 2
        true_val = (true_val + 1) / 2

        # Load audio signal and split into segments of 0.2 seconds
        signal = [f1[i][1]]
        chunks = get_audio_chunks_jl(signal, frame_size = JL_FRAME_SIZE,
         sampling_rate=SAMPLING_RATE)
        
        if (len(chunks) != len(true_aro)):
            chunks = chunks[:len(true_aro)]

        sess_pred_aro = []
        sess_pred_val = []

        for j, chunk in enumerate(chunks):
            # Process each chunk and retrieve arousal and valence values
            chunk = np.array(chunk, dtype=np.float32)
            results = process_func([[chunk]], SAMPLING_RATE, model=model, processor=processor)
            sess_pred_aro.append(results[0][0])
            sess_pred_val.append(results[0][2])
            print('Processed chunk ' + str(j + 1) + ' of ' + str(len(chunks)))
            # Print true and predicetd arousal and valence values
            print('True arousal: ' + str(true_aro.iloc[j]))
            print('Predicted arousal: ' + str(results[0][0]))
            print('True valence: ' + str(true_val.iloc[j]))
            print('Predicted valence: ' + str(results[0][2])+'\n')

        # Calculate the CCC value of the predicted arousal and valence values for f1
        f1_aro_ccc.append(ccc(np.array(true_aro), np.array(sess_pred_aro)))
        f1_val_ccc.append(ccc(np.array(true_val), np.array(sess_pred_val)))

        print(f'Session {i} - CCC arousal: {f1_aro_ccc[i]:.2f}, CCC valence: {f1_val_ccc[i]:.2f}')
    
    # Iterate through each wav file and calculate the arousal and valence values



