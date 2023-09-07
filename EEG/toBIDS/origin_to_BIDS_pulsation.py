import os
import re
import mne
from mne_bids import BIDSPath, write_raw_bids
from mne_bids import make_report

origin_folder = fr'/Users/nicolaspiron/Documents/Master_thesis/EEG/data'
print(os.listdir(origin_folder))

bids_root = fr'/Users/nicolaspiron/Documents/Master_thesis/EEG/toBIDS/BIDS_data/sourcedata'
for root, dirs, files in os.walk(origin_folder):
     for file in files:
        if file.endswith('.bdf'):
            file = os.path.join(root, file)
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
                    }

            elif task == 'Alpheye':
            
                event_id = {'stim/Landscape':2, 'stim/Human':4, 'stim/question':10,
                            'response/correct':128, 'response/incorrect':129}

            write_raw_bids(raw, bids_path=bids_path, events_data=e_list, event_id=event_id, overwrite=True)   

print(make_report(bids_root))