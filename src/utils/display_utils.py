from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def map_w2v_to_quadrant(input_val: float, input_aro: float, input_dom: float) -> float:
    ''' Maps dimensional values from wav2vec2 (0 -> 1) to quadrant (-1 -> 1) values for plotting. '''
    output_val = (input_val - 0.5) * 2
    output_aro = (input_aro - 0.5) * 2
    output_dom = (input_dom - 0.5) * 2
    return output_val, output_aro, output_dom

def map_msp_to_w2v(input_aro: float, input_val: float) -> float:
    '''
    Maps a set of values from the range 1 to 7 (from msp) to 0 to 1 (from wav2vec)
    '''
    output_aro = (input_aro - 1) / 6
    output_val = (input_val - 1) / 6
    return output_aro, output_val

def map_arrays_to_w2v(input_aro: List[float], input_val: List[float]) -> List[float]:
    '''
    Maps each value in two lists from the range -1 to 1 to 0 to 1 for use with wav2vec
    '''
    output_aro = [ (x + 1) /2 for x in input_aro ]
    output_val = [ (x + 1) /2 for x in input_val ]
    return output_aro, output_val

def quadrant_chart(input_val: List[float], input_aro: List[float], true_val: List[float], true_aro: List[float], xtick_labels=None, ytick_labels=None, ax=None, frame_size: float =.02) -> None:
    '''
    Function to display dimensional emotion data on a quadrant chart
    Function modifed from code provided from: https://gist.githubusercontent.com/ctnormand1/d9326e1556ed199d92063b89675450ef/raw/a707f33883ff59e2f738d236d4c555de56346562/simple_quadrant_chart.py
    '''
    # Check validity of input values
    val_size = len(input_val)
    print(val_size)
    if val_size != len(input_aro) or val_size != len(true_val) or val_size != len(true_aro):
        raise ValueError("Input values must be of same length")

    data = pd.DataFrame({'pred_x': input_val, 'pred_y': input_aro, 'true_x': true_val, 'true_y': true_aro})

    # Create axes
    ax = plt.axes()

    # Set x and y limits
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    # Generate timestamps
    frame_size_ms = np.arange(0, val_size)
    
    # ax.scatter(x=data['x'], y=data['y'], c=input_aro, cmap = 'Spectral', edgecolor='darkblue', s=200, zorder=99)
    true_plot = ax.scatter(x=data['true_x'], y=data['true_y'], c=frame_size_ms, cmap='Blues', edgecolor='darkblue', s=20, zorder=99)
    pred_plot = ax.scatter(x=data['pred_x'], y=data['pred_y'], c=frame_size_ms, cmap='Reds', edgecolor='darkred', s=20, zorder=99)

    # Add annotations for reference
    ax.annotate('Alarmed', (0, 1.1), fontsize=6)
    ax.annotate('Distressed', (-.757, .757), fontsize=6)
    ax.annotate('Miserable', (-1.05, 0), fontsize=6)
    ax.annotate('Gloomy', (-.757, -.757), fontsize=6)
    ax.annotate('Sleepy', (0, -1.1), fontsize=6)
    ax.annotate('Satisfied', (.757, -.757), fontsize=6)
    ax.annotate('Happy', (1.05, 0), fontsize=6)
    ax.annotate('Excited', (.757, .757), fontsize=6)

    ax.axvline(0, c='k', lw=1)
    ax.axhline(0, c='k', lw=1)
    # plt.colorbar(true_plot, location='bottom', label='True emotion vs time')
    # plt.colorbar(pred_plot, location='bottom', label='Predicted emotion vs time')
    plt.colorbar(pred_plot, label='Predicted emotion vs time')
    plt.colorbar(true_plot, label='True emotion vs time')
    

if __name__ == '__main__':
    input_val=[-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    quadrant_chart(
        # [-1,-.707, 0, .707, 1, .707, -.707, 0, 0],
        # [0, .707, 0, -.707, 0, .707, -.707, 1, -1],
        input_val=input_val,
        input_aro=input_val,
        # input_aro=[-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        true_val=[-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        true_aro=[-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.01],
    )

    plt.title('Arousal vs Valence', fontsize=16)
    plt.ylabel('Arousal', fontsize=14)
    plt.xlabel('Valence', fontsize=14)
    plt.grid(True, animated=True, linestyle='--', alpha=0.5)
    plt.show()
    