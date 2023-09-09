import os
import re
import glob
import mne
from mne_bids import BIDSPath, write_raw_bids
from mne_bids import make_report

origin_folder ='/Users/nicolaspiron/Documents/PULSATION/Data/EEG/Restingstate'

bids_root = '/Users/nicolaspiron/Documents/Master_thesis/EEG/toBIDS/BIDS_data/sourcedata'

files = glob.glob(os.path.join(origin_folder, '*.bdf'))

for file in files:
    if file.endswith('bdf'):
        basename = os.path.basename(file).split(".")[0]
        print(basename)
        prefix, task = basename.split('_')
        subject = re.split('(\d+)',prefix)[1]
        print(subject)
        print(task)
        session=1
        bids_path = BIDSPath(subject=subject,
                             session =f"{int(session):02d}",
                             task=task,
                             datatype='eeg',
                             root=bids_root)
        raw = mne.io.read_raw_bdf(file)
        write_raw_bids(raw, bids_path=bids_path, overwrite=True)

print(make_report(bids_root))