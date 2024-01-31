import mne
import os
import numpy as np
import matplotlib.pyplot as plt
from set_paths import get_paths
from mne import read_labels_from_annot
from mne.datasets import fetch_fsaverage

def create_stc_epochs(subject_id):
    '''
    Create the source time course epochs for a given subject.

    Parameters
    ----------
    subject_id : str
        The subject id. 2 digits format, e.g. '01'.

    Returns
    -------
    None
    '''

    input_dir, o = get_paths()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if 'nicolaspiron/Documents' in script_dir:
        subjects_dir = fetch_fsaverage(verbose=True)
    elif 'shared_PULSATION' in script_dir:
        subjects_dir = '/home/nicolasp/shared_PULSATION/MNE-fsaverage-data/fsaverage'
    else:
        raise Exception('Please specify the path to the fsaverage directory in the create_stc_epochs function.')
    
    for condition in ['RESTINGSTATECLOSE', 'RESTINGSTATEOPEN']: 

        print(f'========== WORKING ON {condition} - {subject_id}')

        epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-{condition}.fif'))
        info = epochs.info
        info['bads'] = []

        # the same forward for every subject becauses no MRI scans
        if not os.path.exists(os.path.join(input_dir, 'sub-01', 'fwd', 'sub-01-fwd.fif')):
            raise Exception('Please compute the forward solution for subject 01 before running this function.')

        # create stc epochs directory
        for path in [os.path.join(input_dir, f'sub-{subject_id}', condition, 'stc_epochs'),
                    os.path.join(input_dir, f'sub-{subject_id}', condition, 'stc_epochs', 'cov'),
                    os.path.join(input_dir, f'sub-{subject_id}', condition, 'stc_epochs', 'inv'),]:
            if not os.path.exists(path):
                os.makedirs(path)

        # compute noise covariance matrix and inverse operator
        fwd = mne.read_forward_solution(os.path.join(input_dir, 'sub-01', 'fwd', 'sub-01-fwd.fif'))
        cov = mne.compute_covariance(epochs)
        inverse_operator = mne.minimum_norm.make_inverse_operator(info, fwd, cov)

        # save the cov and inverse operator
        mne.write_cov(os.path.join(input_dir, f'sub-{subject_id}', condition, 'stc_epochs', 'cov', f'sub-{subject_id}-cov.fif'), cov)
        mne.minimum_norm.write_inverse_operator(os.path.join(input_dir, f'sub-{subject_id}', condition, 'stc_epochs', 'inv', f'sub-{subject_id}-inv.fif'), inverse_operator)

        # apply inverse operator
        lambda2 = 1. / 9.
        method = 'sLORETA'
        epochs.set_eeg_reference('average', projection=True)
        epochs.apply_proj()
        stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inverse_operator, lambda2=lambda2, method=method, pick_ori=None)

        # extract labels from the aparc parcellation
        labels = read_labels_from_annot('', parc='aparc', subjects_dir=subjects_dir)
        labels = [label for label in labels if 'unknown' not in label.name]
        # get the label time course for each epoch -> n_epochs x n_labels x n_times
        label_ts_epochs = [mne.extract_label_time_course(stc, labels, fwd['src'], mode='pca_flip') for stc in stcs]
        del stcs
        label_ts_array = np.array(label_ts_epochs) 
        np.save(os.path.join(input_dir, f'sub-{subject_id}', condition, 'stc_epochs', f'sub-{subject_id}-{condition}-stc_epochs.npy'), label_ts_array)
        print(f'========== FINISHED {condition} - {subject_id}')

create_stc_epochs('01')