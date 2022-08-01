import os
import glob
import pandas as pd
from librosa import load

SAMPLING_RATE = 16000
msp_data = []

def load_msp():
    # Obtain root file paths
    file_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    root = os.path.dirname(os.path.dirname(file_path))
    msp_path = root + "/data/msp"
    msp_data = []
    
    # Load csv file with annotations
    full_df = pd.read_csv(msp_path + "/labels_concensus.csv", delimiter=',')
    full_df = full_df.drop(['EmoClass', 'EmoDom', 'SpkrID', 'Gender', 'Split_Set'], axis=1)

    full_list_annot = full_df.values.tolist()
    # Merge list of annotations and wav files in list
    for row in full_list_annot:
        wav_file = load(msp_path + '/' + row[0], sr = SAMPLING_RATE)
        msp_data.append([row ,wav_file])
