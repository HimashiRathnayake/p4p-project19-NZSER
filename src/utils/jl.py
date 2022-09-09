from datetime import datetime
import os
import glob
import pandas as pd
import numpy as np
from librosa import load
import matplotlib.pyplot as plt
from typing import List
from model import process_func

from utils.audio import get_audio_chunks_jl
from utils.calc import ccc, map_arrays_to_quadrant, pearson
from utils.display import quadrant_chart

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

    results = []

    # Iterate through f1 utterances and generate and store predictions
    with open('jl_results_f1.txt', 'w', encoding='utf-8') as f:
        f.write(f"True arousal,Predicted arousal,True valence,Predicted valence\n")

        df = pd.DataFrame(columns=['bundle', 'pearson arousal', 'pearson valence', 'ccc arousal', 'ccc valence'])
        # Iterate through each wav file and calculate the arousal and valence values
        for i in range(len(f1)):
            print('Processing wav file ' + str(i + 1) + ' of ' + str(len(f1)))
            f.write(f"File: {f1[i][0].iloc[0]['bundle']}\n")
            true_values = f1[i][0]

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
                # print('Processed chunk ' + str(j + 1) + ' of ' + str(len(chunks)))
                f.write(f"{true_aro.iloc[j]},{results[0][0]},{true_val.iloc[j]},{results[0][2]}\n")
                
            # Calculate the CCC value of the predicted arousal and valence values for f1
            f1_aro_ccc.append(ccc(np.array(true_aro), np.array(sess_pred_aro)))
            f1_val_ccc.append(ccc(np.array(true_val), np.array(sess_pred_val)))
            # Calcuate the pearson correlation coefficient of the predicted arousal and valence values for f1
            f1_aro_pearson = pearson(np.array(true_aro), np.array(sess_pred_aro))
            f1_val_pearson = pearson(np.array(true_val), np.array(sess_pred_val))

            # Concat ccc and pearson values of each file to dataframe
            df = pd.concat([df, pd.DataFrame({'bundle': f1[i][0].iloc[0]['bundle'], 'pearson arousal': f1_aro_pearson, 'pearson valence': f1_val_pearson,
             'ccc arousal': f1_aro_ccc[i], 'ccc valence': f1_val_ccc[i]}, index=[0])], ignore_index=True)

            # print(f'Session {i} - CCC arousal: {f1_aro_ccc[i]:.2f}, CCC valence: {f1_val_ccc[i]:.2f}')
            # print(f'Session {i} - Pearson arousal: {f1_aro_pearson:.2f}, Pearson valence: {f1_val_pearson:.2f} \n')

        # Export the dataframe as a csv file with the timestamp as the filename
        filename = f'jl_results_f1 - {datetime.now().strftime("%H-%M_%d-%m-%Y")}.csv'        
        df.to_csv(filename, index=False)

    # Iterate through m2 utterances and generate and store predictions
    with open('jl_results_m2.txt', 'w', encoding='utf-8') as f:
        f.write(f"True arousal,Predicted arousal,True valence,Predicted valence\n")

        df = pd.DataFrame(columns=['bundle', 'pearson arousal', 'pearson valence', 'ccc arousal', 'ccc valence'])
        # Iterate through each wav file and calculate the arousal and valence values
        for i in range(len(m2)):
            print('Processing wav file ' + str(i + 1) + ' of ' + str(len(m2)))
            f.write(f"File: {m2[i][0].iloc[0]['bundle']}\n")
            true_values = m2[i][0]

            # Retrieve true arousal and valence values from dataframe
            true_aro = true_values['arousal']
            true_val = true_values['valence']

            # Map true arousal and valence values from [-1, 1] -> [0, 1]
            true_aro = (true_aro + 1) / 2
            true_val = (true_val + 1) / 2

            # Load audio signal and split into segments of 0.2 seconds
            signal = [m2[i][1]]
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
                # print('Processed chunk ' + str(j + 1) + ' of ' + str(len(chunks)))
                f.write(f"{true_aro.iloc[j]},{results[0][0]},{true_val.iloc[j]},{results[0][2]}\n")
                
            # Calculate the CCC value of the predicted arousal and valence values for m2
            m2_aro_ccc.append(ccc(np.array(true_aro), np.array(sess_pred_aro)))
            m2_val_ccc.append(ccc(np.array(true_val), np.array(sess_pred_val)))
            # Calcuate the pearson correlation coefficient of the predicted arousal and valence values for m2
            m2_aro_pearson = pearson(np.array(true_aro), np.array(sess_pred_aro))
            m2_val_pearson = pearson(np.array(true_val), np.array(sess_pred_val))

            # Concat ccc and pearson values of each file to dataframe
            df = pd.concat([df, pd.DataFrame({'bundle': m2[i][0].iloc[0]['bundle'], 'pearson arousal': m2_aro_pearson, 'pearson valence': m2_val_pearson,
                'ccc arousal': m2_aro_ccc[i], 'ccc valence': m2_val_ccc[i]}, index=[0])], ignore_index=True)
    
            # print(f'Session {i} - CCC arousal: {m2_aro_ccc[i]:.2f}, CCC valence: {m2_val_ccc[i]:.2f}')
            # print(f'Session {i} - Pearson arousal: {m2_aro_pearson:.2f}, Pearson valence: {m2_val_pearson:.2f} \n')

        # Export the dataframe as a csv file with the timestamp as the filename
        filename = f'jl_results_m2 - {datetime.now().strftime("%H-%M_%d-%m-%Y")}.csv'
        df.to_csv(filename, index=False)

def load_jl_results():
    # Load in the results of the JL-corpus test text file
    jl_results = []
    file_results = []
    filename = ''

    with open('jl_results_f1.txt', 'r', encoding='utf-8') as f:
        for line in f:
            # Skip the first line

            if line == 'True arousal,Predicted arousal,True valence,Predicted valence\n':
                continue

            # Store the results of each prev file into a pandas dataframe
            if line.startswith('File:'):
                # if this is the first file then skip
                if len(file_results) != 0:
                    df = pd.DataFrame(file_results, columns=['true_aro', 'pred_aro', 'true_val', 'pred_val'])
                    jl_results.append([filename, df])
                    filename = line.split('File: ')[1].strip()
                    file_results = []
                continue
            else:
                # Remove the newline character from the end of the line and append to file_results
                list_results = list(map(float, line.strip().split(',')))
                file_results.append(list_results)

            
    return jl_results


def display_jl_quadrant_chart(start_index: int, end_index: int):
    # Load in the results of the JL-corpus test text file
    jl_results = load_jl_results()
    
    if(start_index < 0 or start_index >= len(jl_results)):
        print('Invalid start index')
        return
    
    if(end_index < start_index or end_index >= len(jl_results)):
        print('Invalid end index')
        return
    

    # Iterate through each file and calculate the arousal and valence values
    for i in range(start_index ,end_index):

        results = jl_results[i][1]
        # Retrieve true arousal and valence values from dataframe
        true_aro = results['true_aro'].values.tolist()
        true_val = results['true_val'].values.tolist()

        # Retrieve predicted arousal and valence values from dataframe
        pred_aro = results['pred_aro'].values.tolist()
        pred_val = results['pred_val'].values.tolist()

        # Map true arousal and valence values from [0, 1] -> [-1, 1]
        pred_aro, pred_val = map_arrays_to_quadrant(pred_aro, pred_val)
        # Plot the results using the quadrant chart function
        quadrant_chart(pred_val, pred_aro, true_val, true_aro,)
        plt.figure(i)
        plt.title('Arousal vs Valence', fontsize=16)
        plt.ylabel('Arousal', fontsize=14)
        plt.xlabel('Valence', fontsize=14)
        plt.grid(True, animated=True, linestyle='--', alpha=0.5)
    plt.show()


def test_jl_sentence(processor, model):
    f1, m2 = load_jl()

    true_aro_list = []
    true_val_list = []

    pred_aro_list = []
    pred_val_list = []

    # Iterate through f1 utterances and generate and store predictions
    with open('jl_results_f1_sentence2.txt', 'w', encoding='utf-8') as f:
        f.write(f"True arousal,Predicted arousal,True valence,Predicted valence\n")

        df = pd.DataFrame(columns=['bundle', 'pearson arousal', 'pearson valence', 'ccc arousal', 'ccc valence'])
        # Iterate through each wav file and calculate the arousal and valence values
        for i in range(len(f1)):
            # print('Processing wav file ' + str(i + 1) + ' of ' + str(len(f1)))

            true_values = f1[i][0]

            # Retrieve true arousal and valence values from dataframe
            true_aro = true_values['arousal']
            true_val = true_values['valence']

            # Map true arousal and valence values from [-1, 1] -> [0, 1]
            true_aro = (true_aro + 1) / 2
            true_val = (true_val + 1) / 2

            # Average the true arousal and valence values
            true_aro_list.append(np.mean(true_aro)) 
            true_val_list.append(np.mean(true_val))

            # Load audio signal
            signal = [f1[i][1]]

            # Process signal and retrieve arousal and valence values
            signal = np.array(signal[0][0], dtype=np.float32)
            results = process_func([[signal]], SAMPLING_RATE, model=model, processor=processor)
            pred_aro_list.append(results[0][0])
            pred_val_list.append(results[0][2])
            # Print the true and predicted arousal and valence values
            f.write(f"{true_aro.iloc[0]},{results[0][0]:.4f},{true_val.iloc[0]},{results[0][2]:.4f}\n")


        # Calculate the CCC value of the predicted arousal and valence values for f1
        f1_aro_cc = ccc(np.array(true_aro_list), np.array(pred_aro_list))
        f1_val_cc = ccc(np.array(true_val_list), np.array(pred_val_list))

        # Print the CCC values for f1
        print(f'F1 - CCC arousal: {f1_aro_cc:.2f}, CCC valence: {f1_val_cc:.2f}''')
    

def load_jl_sentence():
    # Load in the results of the JL-corpus test text file
    df = pd.DataFrame(columns=['true_aro', 'pred_aro', 'true_val', 'pred_val'])

    with open('jl_results_f1_sentence2.txt', 'r', encoding='utf-8') as f:
        for line in f:
            # Skip the first line
            if line == 'True arousal,Predicted arousal,True valence,Predicted valence\n':
                continue

            list_results = list(map(float, line.strip().split(',')))
            df = pd.concat([df, pd.DataFrame({'true_aro': list_results[0], 'pred_aro': list_results[1], 'true_val': list_results[2], 'pred_val': list_results[3]}, index=[0])], ignore_index=True)

           
    return df


def display_jl_quadrant_chart_sentence():
    # Load in the results of the JL-corpus test text file
    df = load_jl_sentence()
    
    # Retrieve true arousal and valence values from dataframe
    true_aro = df['true_aro'].values.tolist()
    true_val = df['true_val'].values.tolist()

    # Retrieve predicted arousal and valence values from dataframe
    pred_aro = df['pred_aro'].values.tolist()
    pred_val = df['pred_val'].values.tolist()

    # Map true arousal and valence values from [0, 1] -> [-1, 1]
    pred_aro, pred_val = map_arrays_to_quadrant(pred_aro, pred_val)
    true_aro, true_val = map_arrays_to_quadrant(true_aro, true_val)
    # Plot the results using the quadrant chart function
    quadrant_chart(pred_val, pred_aro, true_val, true_aro,)
    plt.title('Arousal vs Valence', fontsize=16)
    plt.ylabel('Arousal', fontsize=14)
    plt.xlabel('Valence', fontsize=14)
    plt.grid(True, animated=True, linestyle='--', alpha=0.5)
    plt.show()



    




