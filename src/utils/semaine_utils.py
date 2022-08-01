import os
import glob
import re
import pandas as pd
from librosa import load

signals = []
SAMPLING_RATE = 16000
''' 
One time use only.
'''
# Rename folders in semaine that dont have at least 2 text files inside with a "NA"
def prune_semaine():
    # Obtain root file paths
    file_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    root = os.path.dirname(os.path.dirname(file_path))
    semaine_path = root + "/data/semaine/sessions"
    
    # Iterate through folders for renaming
    for folder in os.listdir(semaine_path):
        if len(glob.glob(semaine_path + "/" + folder + "/*.txt")) < 2:
            os.rename(semaine_path + "/" + folder, semaine_path + "/" + folder + "_NA")


def load_semaine():
    # Obtain root file paths
    file_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    root = os.path.dirname(os.path.dirname(file_path))
    semaine_path = root + "/data/semaine/sessions"
    
    # Iterate through session folders and load wav files for each session
    for folder in os.listdir(semaine_path):
        if not folder.endswith("_NA"):
            for file in glob.glob(semaine_path + "/" + folder + "/*.wav"): 
                if (file.find("User HeadMounted") > 0):
                    signals.append(load(file, sr=SAMPLING_RATE))
                    