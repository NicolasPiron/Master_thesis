import os
import re
import mne
import glob
from mne_bids import BIDSPath, write_raw_bids
from mne_bids import make_report
import sys

origin_folder ='/Users/nicolaspiron/Documents/Master_thesis/EEG/data'

bids_root = '/Users/nicolaspiron/Documents/test_BIDS'

def single_subj_to_bids(subject_id, origin_folder, bids_root):

    subject_id = 'S'+str(subject_id)
    subject_files = glob.glob(os.path.join(origin_folder, subject_id + '*.bdf'))
    print(subject_files)
    
    for file in subject_files:
        basename = os.path.basename(file).split(".")[0]
        #print(file)
        prefix, task, suffix = basename.split('_')
        subject = re.split('(\d+)',prefix)[1] # split before digits 
        #print(subject)
        print(task)
        run = suffix.split('r')[1]
        run =f"{int(run):02d}"
        #print(run)
        session = 1

        bids_path = BIDSPath(subject=subject,
                                session =f"{int(session):02d}",
                                task=task,
                                run=run,
                                datatype='eeg',
                                root=bids_root)
        raw = mne.io.read_raw_bdf(file)
        e_list = mne.find_events(raw, stim_channel='Status', mask=0b11111111)
        #annot = mne.annotations_from_events(e_list, raw.info['sfreq'])
        #raw = raw.set_annotations(annot)
        
        if task == 'N2pc':

            event_id = {'stim/dis_top/target_l':1, 'stim/dis_top/target_r':2,
                        'stim/no_dis/target_l':3, 'stim/no_dis/target_r':4,
                        'stim/dis_bot/target_l':5, 'stim/dis_bot/target_r':6,
                        'stim/dis_right/target_l':7, 'stim/dis_left/target_r':8,
                        'response/correct':128, 'response/incorrect':129,
                        'stim/bug':12
                        
                }
            # there is a stim 12 in S22 somehow, but I don't find it in the event list'

        elif task == 'Alpheye':
        
            event_id = {'stim/Landscape':2, 'stim/Human':4, 'stim/question':10,
                        'response/correct':128, 'response/incorrect':129}

        write_raw_bids(raw, bids_path=bids_path, events_data=e_list, event_id=event_id, overwrite=True)   


single_subj_to_bids('S01', origin_folder, bids_root)

if __name__ == "__main__":
    # Check if at least one command-line argument is provided
    if len(sys.argv) < 2:
        print("Usage: python to_BIDS_single_subj.py <subject_id>")
    else:
        # Get the first command-line argument
        subject_id = sys.argv[1]
        single_subj_to_bids(subject_id, origin_folder, bids_root)

        #print(make_report(bids_root))