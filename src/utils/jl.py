from datetime import datetime
import os
import glob
import shutil
import datasets
import pandas as pd
import numpy as np
from librosa import load
import matplotlib.pyplot as plt
from typing import List
from model import process_func

import audformat

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

def load_jl_as_dfs():
    file_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    root = os.path.dirname(os.path.dirname(file_path))

    # Load csv files with annotations
    csv_files = glob.glob(root + "/data/jl/*.csv")

    # f1 = female1, m2 = male2, df = dataframe
    f1_aro_df = pd.read_csv(csv_files[4])
    f1_val_df = pd.read_csv(csv_files[5])
    m2_aro_df = pd.read_csv(csv_files[6])
    m2_val_df = pd.read_csv(csv_files[7])

    f1_aro_df = f1_aro_df[['bundle', 'start', 'end', 'labels' ]]

    f1_aro_df = f1_aro_df.rename(columns = {'labels': 'arousal'})
    f1_dom_df = f1_val_df['labels']
    f1_dom_df = f1_dom_df.rename('dominance')
    f1_val_df = f1_val_df['labels']
    f1_val_df = f1_val_df.rename('valence')

    m2_aro_df = m2_aro_df[['bundle', 'start', 'end', 'labels']]
    m2_aro_df = m2_aro_df.rename(columns = {'labels': 'arousal'})
    m2_dom_df = m2_val_df['labels']
    m2_dom_df = m2_dom_df.rename('dominance')
    m2_val_df = m2_val_df['labels']
    m2_val_df = m2_val_df.rename('valence')

    f1_mer_df = pd.concat([f1_aro_df, f1_dom_df, f1_val_df], axis=1)
    m2_mer_df = pd.concat([m2_aro_df, m2_dom_df, m2_val_df], axis=1)

    return f1_mer_df, m2_mer_df

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
    file_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    root = os.path.dirname(os.path.dirname(file_path))
    
    if(start_index < 0 or start_index >= len(jl_results)):
        print('Invalid start index')
        return
    
    if(end_index < start_index or end_index >= len(jl_results)):
        print('Invalid end index')
        return
    

    # Iterate through each file and calculate the arousal and valence values
    for i in range(start_index ,end_index + 1): # Python exclusive of end index

        results = jl_results[i][1]
        # Retrieve true arousal and valence values from dataframe
        true_aro = results['true_aro'].values.tolist()
        true_val = results['true_val'].values.tolist()

        # Retrieve predicted arousal and valence values from dataframe
        pred_aro = results['pred_aro'].values.tolist()
        pred_val = results['pred_val'].values.tolist()

        # Map true arousal and valence values from [0, 1] -> [-1, 1]
        pred_aro, pred_val = map_arrays_to_quadrant(pred_aro, pred_val)
        true_aro, true_val = map_arrays_to_quadrant(true_aro, true_val)
        # Plot the results using the quadrant chart function
        quadrant_chart(pred_val, pred_aro, true_val, true_aro,)
        # plt.figure(i)
        plt.title('Arousal vs Valence', fontsize=16)
        plt.ylabel('Arousal', fontsize=14)
        plt.xlabel('Valence', fontsize=14)
        plt.grid(True, animated=True, linestyle='--', alpha=0.5)
        # Save the plot as a png file with the index of the file as the filename
        plt.savefig(f'{file_path}/jl_plts/jl_results_f1_{i}.png')
        plt.close()

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

# Move all jl wav files into a single folder
def move_jl_wav_files():
    file_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    root = os.path.dirname(os.path.dirname(file_path))

    f1_bdls, m2_bdls = load_jl_wav_files()

    # Iterate through each file and copy it to the jl_wav_files folder
    for fl_bdl in f1_bdls:
        shutil.copy(f'{root}/data/jl/female1_all_a_1/{fl_bdl}_bndl/{fl_bdl}.wav', f'{root}/data/jl/jl_wav_files/')

    for m2_bdl in m2_bdls:
        shutil.copy(f'{root}/data/jl/male2_all_a_1/{m2_bdl}_bndl/{m2_bdl}.wav', f'{root}/data/jl/jl_wav_files/')

def load_jl_wav_files():
    file_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    root = os.path.dirname(os.path.dirname(file_path))

    # Load csv files with annotations
    csv_files = glob.glob(root + "/data/jl/*.csv")
    
    # f1 = female1, m2 = male2, df = dataframe
    f1_aro_df = pd.read_csv(csv_files[4])
    f1_val_df = pd.read_csv(csv_files[5])
    m2_aro_df = pd.read_csv(csv_files[6])
    m2_val_df = pd.read_csv(csv_files[7])
    
    f1_aro_df = f1_aro_df[['bundle', 'start', 'end', 'labels' ]]
    f1_aro_df = f1_aro_df.rename(columns = {'labels': 'arousal'})
    f1_val_df = f1_val_df['labels']
    f1_val_df = f1_val_df.rename('valence')

    m2_aro_df = m2_aro_df[['bundle', 'labels']]
    m2_aro_df = m2_aro_df.rename(columns = {'labels': 'arousal'})
    m2_val_df = m2_val_df['labels']
    m2_val_df = m2_val_df.rename('valence')

    f1_mer_df = pd.concat([f1_aro_df, f1_val_df], axis=1)
    m2_mer_df = pd.concat([m2_aro_df, m2_val_df], axis=1)

    # Merge the two dataframes
    jl_df = pd.concat([f1_mer_df, m2_mer_df], axis=0)

    # Remove duplicate bundle names and store in list
    f1_bdls = list(dict.fromkeys(f1_mer_df['bundle'].tolist()))
    m2_bdls = list(dict.fromkeys(m2_mer_df['bundle'].tolist()))

    return f1_bdls, m2_bdls

def df_train_test_split(df, train_size=0.7):
    # Split the dataframe into train and test sets
    train_df = df.sample(frac=train_size, random_state=0)
    test_df = df.drop(train_df.index)

    return train_df, test_df

def split_dfs_by_emotion(f1_df, m2_df, train_split, emotions_filter: List = None):
    ''' Split the dataframe into separate dataframes for each emotion by checking 
        if bundle name contains the emotion name.
        
        f1 = female1, m2 = male2, df = dataframe
    '''
    f1_ang_df = f1_df[f1_df['bundle'].str.contains('angry')]  
    f1_anx_df = f1_df[f1_df['bundle'].str.contains('anxious')]
    f1_apo_df = f1_df[f1_df['bundle'].str.contains('apologetic')]
    f1_ent_df = f1_df[f1_df['bundle'].str.contains('enthusiastic')]
    f1_exc_df = f1_df[f1_df['bundle'].str.contains('excited')]
    f1_hap_df = f1_df[f1_df['bundle'].str.contains('happy')]
    f1_neu_df = f1_df[f1_df['bundle'].str.contains('neutral')]
    f1_sad_df = f1_df[f1_df['bundle'].str.contains('sad')]

    m2_ang_df = m2_df[m2_df['bundle'].str.contains('angry')]
    m2_anx_df = m2_df[m2_df['bundle'].str.contains('anxious')]
    m2_apo_df = m2_df[m2_df['bundle'].str.contains('apologetic')]
    m2_ent_df = m2_df[m2_df['bundle'].str.contains('enthusiastic')]
    m2_exc_df = m2_df[m2_df['bundle'].str.contains('excited')]
    m2_hap_df = m2_df[m2_df['bundle'].str.contains('happy')]
    m2_neu_df = m2_df[m2_df['bundle'].str.contains('neutral')]
    m2_sad_df = m2_df[m2_df['bundle'].str.contains('sad')]
    m2_con_df = m2_df[m2_df['bundle'].str.contains('confident')]
    m2_wor_df = m2_df[m2_df['bundle'].str.contains('worried')]

    # Split dfs into train and test sets with a defined split ratio from input param
    f1_ang_train, f1_ang_test = df_train_test_split(f1_ang_df, train_size=train_split)
    f1_anx_train, f1_anx_test = df_train_test_split(f1_anx_df, train_size=train_split)
    f1_apo_train, f1_apo_test = df_train_test_split(f1_apo_df, train_size=train_split)
    f1_ent_train, f1_ent_test = df_train_test_split(f1_ent_df, train_size=train_split)
    f1_exc_train, f1_exc_test = df_train_test_split(f1_exc_df, train_size=train_split)
    f1_hap_train, f1_hap_test = df_train_test_split(f1_hap_df, train_size=train_split)
    f1_neu_train, f1_neu_test = df_train_test_split(f1_neu_df, train_size=train_split)
    f1_sad_train, f1_sad_test = df_train_test_split(f1_sad_df, train_size=train_split)

    m2_ang_train, m2_ang_test = df_train_test_split(m2_ang_df, train_size=train_split)
    m2_anx_train, m2_anx_test = df_train_test_split(m2_anx_df, train_size=train_split)
    m2_apo_train, m2_apo_test = df_train_test_split(m2_apo_df, train_size=train_split)
    m2_ent_train, m2_ent_test = df_train_test_split(m2_ent_df, train_size=train_split)
    m2_exc_train, m2_exc_test = df_train_test_split(m2_exc_df, train_size=train_split)
    m2_hap_train, m2_hap_test = df_train_test_split(m2_hap_df, train_size=train_split)
    m2_neu_train, m2_neu_test = df_train_test_split(m2_neu_df, train_size=train_split)
    m2_sad_train, m2_sad_test = df_train_test_split(m2_sad_df, train_size=train_split)
    m2_con_train, m2_con_test = df_train_test_split(m2_con_df, train_size=train_split)
    m2_wor_train, m2_wor_test = df_train_test_split(m2_wor_df, train_size=train_split)
    
    # Merge emotions into one dataframe for train and test
    f1_train_df = pd.concat([f1_ang_train, f1_anx_train, f1_apo_train, f1_ent_train, f1_exc_train, f1_hap_train, f1_neu_train, f1_sad_train])
    f1_test_df = pd.concat([f1_ang_test, f1_anx_test, f1_apo_test, f1_ent_test, f1_exc_test, f1_hap_test, f1_neu_test, f1_sad_test])
    m2_train_df = pd.concat([m2_ang_train, m2_anx_train, m2_apo_train, m2_ent_train, m2_exc_train, m2_hap_train, m2_neu_train, m2_sad_train, m2_con_train, m2_wor_train])
    m2_test_df = pd.concat([m2_ang_test, m2_anx_test, m2_apo_test, m2_ent_test, m2_exc_test, m2_hap_test, m2_neu_test, m2_sad_test, m2_con_test, m2_wor_test])

    # Join train sets and test sets into one dataframe
    train_df = pd.concat([f1_train_df, m2_train_df])
    test_df = pd.concat([f1_test_df, m2_test_df])

    # Shuffle train and test sets
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)

    return train_df, test_df



def transform_jl_dataset():
    file_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    root = os.path.dirname(os.path.dirname(file_path))

    # Load csv files with annotations
    csv_files = glob.glob(root + "/data/jl/*.csv")
    
    # f1 = female1, m2 = male2, df = dataframe
    f1_aro_df = pd.read_csv(csv_files[4])
    f1_val_df = pd.read_csv(csv_files[5])
    m2_aro_df = pd.read_csv(csv_files[6])
    m2_val_df = pd.read_csv(csv_files[7])
    
    f1_aro_df = f1_aro_df[['bundle', 'start', 'end', 'labels' ]]

    f1_aro_df = f1_aro_df.rename(columns = {'labels': 'arousal'})
    f1_dom_df = f1_val_df['labels']
    f1_dom_df = f1_dom_df.rename('dominance')
    f1_val_df = f1_val_df['labels']
    f1_val_df = f1_val_df.rename('valence')

    m2_aro_df = m2_aro_df[['bundle', 'start', 'end', 'labels']]
    m2_aro_df = m2_aro_df.rename(columns = {'labels': 'arousal'})
    m2_dom_df = m2_val_df['labels']
    m2_dom_df = m2_dom_df.rename('dominance')
    m2_val_df = m2_val_df['labels']
    m2_val_df = m2_val_df.rename('valence')

    f1_mer_df = pd.concat([f1_aro_df, f1_dom_df, f1_val_df], axis=1)
    m2_mer_df = pd.concat([m2_aro_df, m2_dom_df, m2_val_df], axis=1)

    # Split data into train and test sets by emotion and merge f1 and m2 dataframes
    train_set,  test_set = split_dfs_by_emotion(f1_mer_df, m2_mer_df, train_split=0.7)

    # # Merge the two dataframes
    # jl_df = pd.concat([f1_mer_df, m2_mer_df], axis=0)

    # Store annotations only in a df
    train_annotations_df = train_set[['arousal', 'dominance', 'valence']]
    test_annotations_df = test_set[['arousal', 'dominance', 'valence']]

    # Retrieve bundle column as list from df
    # filenames = jl_df['bundle'].tolist()
    train_filenames = train_set['bundle'].tolist()
    test_filenames = test_set['bundle'].tolist()

    # Append jl_wav_files dir name to each filename along with the .wav file extension
    jl_wav_files_dir = root + '/data/jl/jl_wav_files/'
    train_files = [jl_wav_files_dir + filename + '.wav' for filename in train_filenames]
    test_files = [jl_wav_files_dir + filename + '.wav' for filename in test_filenames]

    # Store start and end times in lists
    train_start_times = train_set['start'].tolist()
    train_end_times = train_set['end'].tolist()
    test_start_times = test_set['start'].tolist()
    test_end_times = test_set['end'].tolist()

    # Convert start and end times to from milliseconds to seconds
    train_start_times = [start_time / 1000 for start_time in train_start_times]
    train_end_times = [end_time / 1000 for end_time in train_end_times]
    test_start_times = [start_time / 1000 for start_time in test_start_times]
    test_end_times = [end_time / 1000 for end_time in test_end_times]

    train_segmented_index = audformat.segmented_index(train_files, train_start_times, train_end_times)
    test_segmented_index = audformat.segmented_index(test_files, test_start_times, test_end_times)

    jl_train_series = pd.Series(
        data=train_annotations_df.values.tolist(),
        index=train_segmented_index,
        dtype=object,
        name='labels'
    )

    jl_test_series = pd.Series(
        data=test_annotations_df.values.tolist(),
        index=test_segmented_index,
        dtype=object,
        name='labels'
    )

    jl_train_series.name = 'labels'
    jl_test_series.name = 'labels'

    train_dataset_df = jl_train_series.reset_index()
    test_dataset_df = jl_test_series.reset_index()

    train_dataset_df.start = train_dataset_df.start.dt.total_seconds().astype('str')
    train_dataset_df.end = train_dataset_df.end.dt.total_seconds().astype('str')
    test_dataset_df.start = test_dataset_df.start.dt.total_seconds().astype('str')
    test_dataset_df.end = test_dataset_df.end.dt.total_seconds().astype('str')
    
    train_dataset_df['input_values'] = train_dataset_df[['file', 'start', 'end']].values.tolist()
    test_dataset_df['input_values'] = test_dataset_df[['file', 'start', 'end']].values.tolist()

    train_dataset_df = train_dataset_df[['labels', 'input_values']]
    test_dataset_df = test_dataset_df[['labels', 'input_values']]

    train_dataset = datasets.Dataset.from_pandas(train_dataset_df)
    test_dataset = datasets.Dataset.from_pandas(test_dataset_df)

    # Save jl_train and jl_test to disk
    train_dataset.save_to_disk(root + "/data/jl/train_dataset")
    test_dataset.save_to_disk(root + "/data/jl/test_dataset")
    
    return train_dataset, test_dataset

# Loads JL formatted datasets from disk 
def load_jl_datasets_disk():
    file_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    repo_path = os.path.dirname(os.path.dirname(file_path))

    train_dataset = datasets.load_from_disk(repo_path + "/data/jl/train_dataset")
    test_dataset = datasets.load_from_disk(repo_path + "/data/jl/test_dataset")
    return train_dataset, test_dataset
    

def evaluate_single_file_(filename: str, processor, model):

    file_path = os.path.realpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__)))
    root = os.path.dirname(os.path.dirname(file_path))

    # Load in the one wav file that matches the filename
    signal = load(root + "/data/jl/female1_all_a_1/" + filename +
                  '_bndl/' + filename + ".wav", sr=SAMPLING_RATE)

    # If bundle contains "male" load in male csv else load female csv
    # Load csv files with annotations
    csv_files = glob.glob(root + "/data/jl/*.csv")

    # f1 = female1, m2 = male2, df = dataframe
    if filename.__contains__("female"):
        aro_df = pd.read_csv(csv_files[0])
        val_df = pd.read_csv(csv_files[1])
    else:
        aro_df = pd.read_csv(csv_files[2])
        val_df = pd.read_csv(csv_files[3])

    aro_df = aro_df[['bundle', 'labels']]
    aro_df = aro_df.rename(columns={'labels': 'arousal'})
    val_df = val_df['labels']
    val_df = val_df.rename('valence')

    mer_df = pd.concat([aro_df, val_df], axis=1)

    # Extract the df with the bundle name that matches the filename
    mer_df = mer_df[mer_df['bundle'] == filename]

    print('Finished loading JL-corpus WAV file.')

    # Test the signal
    df = pd.DataFrame(columns=['true_aro', 'pred_aro', 'true_val', 'pred_val'])

    # Get the arousal and valence labels
    true_aro = mer_df['arousal'].values
    true_val = mer_df['valence'].values

    # Map true arousal and valence values from [-1, 1] -> [0, 1]
    true_aro = (true_aro + 1) / 2
    true_val = (true_val + 1) / 2

    # Load audio signal and split into segments of 0.2 seconds
    signal = [signal]
    chunks = get_audio_chunks_jl(
        signal, frame_size=JL_FRAME_SIZE, sampling_rate=SAMPLING_RATE)

    if (len(chunks) != len(true_aro)):
        chunks = chunks[:len(true_aro)]

    pred_aro = []
    pred_val = []

    for j, chunk in enumerate(chunks):
        # Process each chunk and retrieve arousal and valence values
        chunk = np.array(chunk, dtype=np.float32)
        results = process_func(
            [[chunk]], SAMPLING_RATE, model=model, processor=processor)
        pred_aro.append(results[0][0])
        pred_val.append(results[0][2])

    # Map true arousal and valence values from [0, 1] -> [-1, 1]
    pred_aro, pred_val = map_arrays_to_quadrant(pred_aro, pred_val)
    true_aro, true_val = map_arrays_to_quadrant(true_aro, true_val)

    # Plot the results using the quadrant chart function
    quadrant_chart(pred_val, pred_aro, true_val, true_aro)
    plt.title('Arousal vs Valence', fontsize=16)
    plt.ylabel('Arousal', fontsize=14)
    plt.xlabel('Valence', fontsize=14)
    plt.grid(True, animated=True, linestyle='--', alpha=0.5)
    # Save the plot as a png file with the index of the file as the filename
    plt.savefig(f'{file_path}/jl_plts_ft/jl_results_f1_{filename}.png')
    plt.close()
