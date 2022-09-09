from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def quadrant_chart(input_val: List[float], input_aro: List[float], true_val: List[float], true_aro: List[float], xtick_labels=None, ytick_labels=None, ax=None, frame_size: float =.02) -> None:
    '''
    Function to display dimensional emotion data on a quadrant chart
    Function modifed from code provided from: https://gist.githubusercontent.com/ctnormand1/d9326e1556ed199d92063b89675450ef/raw/a707f33883ff59e2f738d236d4c555de56346562/simple_quadrant_chart.py
    '''
    # Check validity of input values
    val_size = len(input_val)
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
    