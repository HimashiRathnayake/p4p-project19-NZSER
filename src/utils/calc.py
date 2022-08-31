# Utility functions used for calculations and mapping
from typing import List
import numpy as np

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


def ccc(x, y):
    ''' Concordance Correlation Coefficient'''
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc