import mne
import numpy as np
import os

def invert_sides(subject_id, condition, input_dir):

    epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'cleaned_epochs',
                                         f'sub-{subject_id}-cleaned_epochs-{condition}.fif'))
    epochs.info['bads'] = []
    data = epochs.get_data()

    ch_names = epochs.ch_names
    left_idx = []
    right_idx = []
    for ch in ch_names:
        if ch == 'Status':
            pass
        elif 'z' in ch or 'EXG' in ch:
            pass
        elif int(ch[-1])%2 == 0:
            right_idx.append(ch_names.index(ch))
        elif int(ch[-1])%2 != 0:
            left_idx.append(ch_names.index(ch))

    left_data = data[:,left_idx,:]
    right_data = data[:,right_idx,:]
    inverted_data = data.copy()
    inverted_data[:,right_idx,:] = left_data
    inverted_data[:,left_idx,:] = right_data

    inverted_epochs = mne.EpochsArray(inverted_data, epochs.info)

    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'inverted_epochs')):
        os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'inverted_epochs'))
    inverted_epochs.save(os.path.join(input_dir, f'sub-{subject_id}', condition,
                                         'inverted_epochs', f'sub-{subject_id}-cleaned_epochs-{condition}.fif'), overwrite=True)
