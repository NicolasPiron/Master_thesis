import mne
import os
import numpy as np
import matplotlib.pyplot as plt
from .set_paths import get_paths
from mne import read_labels_from_annot
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import compute_source_psd_epochs

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
    
    for condition in ['RESTINGSTATEOPEN', 'RESTINGSTATECLOSE']: 

        print(f'========== WORKING ON {condition} - {subject_id}')

        epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', condition,
                                            'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-{condition}.fif'))
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
        mne.write_cov(os.path.join(input_dir, f'sub-{subject_id}', condition, 'stc_epochs', 'cov', f'sub-{subject_id}-cov.fif'), cov, overwrite=True)
        mne.minimum_norm.write_inverse_operator(os.path.join(input_dir, f'sub-{subject_id}', condition, 'stc_epochs', 'inv', f'sub-{subject_id}-inv.fif'),
                                                 inverse_operator, overwrite=True)

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

def compute_source_psd_rs(subject_id, condition):
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

    print(f'========== WORKING ON {condition} - {subject_id}')

    epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', condition,
                                            'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-{condition}.fif'))
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
    mne.write_cov(os.path.join(input_dir, f'sub-{subject_id}', condition, 'stc_epochs', 'cov', f'sub-{subject_id}-cov.fif'), cov, overwrite=True)
    mne.minimum_norm.write_inverse_operator(os.path.join(input_dir, f'sub-{subject_id}', condition, 'stc_epochs', 'inv', f'sub-{subject_id}-inv.fif'),
                                                inverse_operator, overwrite=True)


    # extract labels from the aparc parcellation
    labels = read_labels_from_annot('', parc='aparc', subjects_dir=subjects_dir)
    labels = [label for label in labels if 'unknown' not in label.name]

    def get_psd_from_label(label):

        name = label.name
        # apply inverse operator
        snr = 3.0  # use smaller SNR for raw data
        lambda2 = 1.0 / snr**2
        method = 'sLORETA'

        fmin, fmax = 0.0, 30.0
        bandwidth = 2.0  # bandwidth of the windows in Hz   

        stcs = compute_source_psd_epochs(
                epochs,
                inverse_operator,
                lambda2=lambda2,
                method=method,
                fmin=fmin,
                fmax=fmax,
                bandwidth=bandwidth,
                label=label,
                return_generator=True,
                verbose=True,
            )

        psd_avg = 0.0
        for i, stc in enumerate(stcs):
            psd_avg += stc.data
        psd_avg /= len(epochs)
        freqs = stc.times  # the frequencies are stored here
        stc.data = psd_avg

        for path in ['freqs', 'spectrum']:
            if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'source-level', 'psd-data', path)):
                os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'source-level', 'psd-data', path))

        np.save(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'source-level', 'psd-data', 'spectrum',
                                f'sub-{subject_id}-{condition}-{name}-psd.npy'), psd_avg)
        np.save(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'source-level', 'psd-data', 'freqs',
                                f'sub-{subject_id}-{condition}-{name}-freqs.npy'), freqs)
        print(f'========== FINISHED {condition} - {subject_id}')

    labels_ROI_idx = [4, 5, 14, 15, 22, 23]
    for idx in labels_ROI_idx:
        get_psd_from_label(labels[idx])

if __name__ == '__main__':

    compute_source_psd_rs('01', 'RESTINGSTATEOPEN')