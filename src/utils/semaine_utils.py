import os
import glob
import pandas as pd
from librosa import load

semaine_data = []
SAMPLING_RATE = 16000
''' 
One time use only.
'''
# Rename folders in semaine that dont have at least 2 text files inside with a "NA"
def prune_semaine():
    # Obtain root file paths
    file_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    root = os.path.dirname(os.path.dirname(file_path))
    sessions_path = root + "/data/semaine/sessions"
    
    # Iterate through folders for renaming
    for folder in os.listdir(sessions_path):
        if len(glob.glob(sessions_path + "/" + folder + "/*.txt")) < 2:
            os.rename(sessions_path + "/" + folder, sessions_path + "/" + folder + "_NA")


def load_semaine():
    # Obtain root file paths
    file_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    root = os.path.dirname(os.path.dirname(file_path))
    sessions_path = root + "/data/semaine/sessions"
    annotations_path = root + "/data/semaine/TrainingInput"
    
    dfs = []
    signals = []

    # Iterate through TrainingInput folder for CSV annotations
    for folder in os.listdir(annotations_path):
        df = pd.read_csv(annotations_path + "/" + folder + '/TU_VA.csv')
        df.insert(0, 'bundle', folder) # Add filename to dataframe
        dfs.append(df)

    # Iterate through session folders and load wav files for each session
    for folder in os.listdir(sessions_path):
        if not folder.endswith("_NA"):
            for file in glob.glob(sessions_path + "/" + folder + "/*.wav"): 
                if (file.find("User HeadMounted") > 0):
                    signals.append(load(file, sr=SAMPLING_RATE))

    # Merge WAV files and their corresponding annotations
    for i, df in enumerate(dfs):
        semaine_data.append([df, signals[i]])
