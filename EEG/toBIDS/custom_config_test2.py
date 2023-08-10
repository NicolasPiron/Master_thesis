# Default settings for data processing and analysis.

from typing import Optional, Union, Iterable, List, Tuple, Dict, Callable, Literal

from numpy.typing import ArrayLike

import mne
import argparse
import sys
from mne_bids import BIDSPath
import numpy as np

from mne_bids_pipeline.typing import PathLike, ArbitraryContrast



###############################################################################
# Config parameters
# -----------------

study_name = "PULSATION"
bids_root = "/Users/caroleguedj/Desktop/PULSATION_ANAL/ToBIDS/BIDS_data/sourcedata"
deriv_root = "/Users/caroleguedj/Desktop/PULSATION_ANAL/ToBIDS/BIDS_data/derivatives"

task = "N2pc"
sessions = "all"
runs = ["01", "02"]
subjects = ["02"]
data_type = 'eeg'
ch_types = ['eeg']

interactive = False

raw_resample_sfreq = 500

eeg_bipolar_channels = {'HEOG': ('EXG4', 'EXG3'),
                               'VEOG': ('EXG2', 'EXG1')}
eog_channels = ["VEOG","HEOG","EXG1", "EXG2", "EXG3", "EXG4", "EXG5", "EXG6"]
drop_channels = ["EXG7", "EXG8"]

eeg_template_montage = mne.channels.make_standard_montage("biosemi64")



l_freq = 1
h_freq = 30
notch_freq = 50

epochs_tmin = -0.2
epochs_tmax = 1
baseline = (None, 0)        

conditions = ['stim/dis_top/target_l', 'stim/dis_top/target_r',
                        'stim/no_dis/target_l', 'stim/no_dis/target_r',
                        'stim/dis_bot/target_l', 'stim/dis_bot/target_r',
                        'stim/dis_right/target_l', 'stim/dis_left/target_r']
                        
contrasts = [("stim/target_r", "stim/target_l")]


