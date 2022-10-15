from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from model import process_func

from utils.calc import ccc, map_arrays_to_quadrant
from utils.display import quadrant_chart
from utils.jl import SAMPLING_RATE, load_jl

# The JL corpus annotations were averaged for a single file and the entire file was used to evaluate the model
# This was done to see if the model was more accurate when the entire file was used to evaluate the model
# The results showed that the baseline models performance was about the same and so this was not continued


def test_jl_sentence(processor, model):
    f1, m2 = load_jl()

    true_aro_list = []
    true_val_list = []

    pred_aro_list = []
    pred_val_list = []

    # Iterate through f1 utterances and generate and store predictions
    with open('jl_results_f1_sentence2.txt', 'w', encoding='utf-8') as f:
        f.write(f"True arousal,Predicted arousal,True valence,Predicted valence\n")

        df = pd.DataFrame(columns=[
                          'bundle', 'pearson arousal', 'pearson valence', 'ccc arousal', 'ccc valence'])
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
            results = process_func(
                [[signal]], SAMPLING_RATE, model=model, processor=processor)
            pred_aro_list.append(results[0][0])
            pred_val_list.append(results[0][2])
            # Print the true and predicted arousal and valence values
            f.write(
                f"{true_aro.iloc[0]},{results[0][0]:.4f},{true_val.iloc[0]},{results[0][2]:.4f}\n")

        # Calculate the CCC value of the predicted arousal and valence values for f1
        f1_aro_cc = ccc(np.array(true_aro_list), np.array(pred_aro_list))
        f1_val_cc = ccc(np.array(true_val_list), np.array(pred_val_list))

        # Print the CCC values for f1
        print(
            f'F1 - CCC arousal: {f1_aro_cc:.2f}, CCC valence: {f1_val_cc:.2f}''')


def load_jl_sentence_results():
    # Load in the results of the JL-corpus test text file
    df = pd.DataFrame(columns=['true_aro', 'pred_aro', 'true_val', 'pred_val'])

    with open('jl_results_f1_sentence2.txt', 'r', encoding='utf-8') as f:
        for line in f:
            # Skip the first line
            if line == 'True arousal,Predicted arousal,True valence,Predicted valence\n':
                continue

            list_results = list(map(float, line.strip().split(',')))
            df = pd.concat([df, pd.DataFrame({'true_aro': list_results[0], 'pred_aro': list_results[1],
                           'true_val': list_results[2], 'pred_val': list_results[3]}, index=[0])], ignore_index=True)

    return df


def display_jl_quadrant_chart_sentence():
    # Load in the results of the JL-corpus test text file
    df = load_jl_sentence_results()

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
