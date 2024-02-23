import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from .src_rec_rs import compute_source_psd_rs
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import compute_source_psd_epochs
from mne import read_labels_from_annot

def get_paths():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if 'nicolaspiron/Documents' in script_dir:
        input_dir = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/output_preproc'
        output_dir = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/output_preproc'
    elif 'shared_PULSATION' in script_dir:
        input_dir = '/home/nicolasp/shared_PULSATION/derivative'
        output_dir = '/home/nicolasp/shared_PULSATION/derivative'

    return input_dir, output_dir

def get_psd(epochs):
    '''compute the power spectral density of the epochs and return the spectra and the frequencies.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs object.
    
    Returns
    -------
    spectra : np.array
        The power spectral density of the epochs.
    freqs : np.array
        The frequencies. (x-axis of the PSD plot in Hz, very important for the FOOOF fit!)
    
    '''
    epochs.info['bads'] = []
    psd = epochs.compute_psd(fmin=0, fmax=30, method='multitaper')
    freqs = psd.freqs
    spectra = psd.get_data().mean(axis=0)
    return spectra, freqs

def save_psd(subject_id, condition):
    '''save the power spectral density of the epochs as a .npy file.

    Parameters
    ----------
    subject_id : str
        The subject id.
    condition : str
        The condition (e.g. 'RESTINGSTATEOPEN' or 'RESTINGSTATECLOSE').
    
    Returns
    -------
    None
    '''

    input_dir, _ = get_paths()
    epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'preprocessing',
                                          'additional', '07-epochs-10s', f'sub-{subject_id}-{condition}-10s-epo.fif'))
    psd, freqs = get_psd(epochs)

    for path in ['spectrum', 'freqs']:
        if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'sensor-level', 'psd-data', path)):
            os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'sensor-level', 'psd-data', path))

    np.save(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'sensor-level', 'psd-data', 'spectrum', f'sub-{subject_id}-{condition}-psd.npy'), psd)
    np.save(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'sensor-level', 'psd-data', 'freqs', f'sub-{subject_id}-{condition}-freqs.npy'), freqs)

def get_data(subject_id, condition):

    input_dir, _ = get_paths()

    #if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'sensor-level', 'psd-data', 'spectrum',
    #                                   f'sub-{subject_id}-{condition}-psd.npy')):
    save_psd(subject_id, condition)
    spectra = np.load(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'sensor-level', 'psd-data', 'spectrum',
                                      f'sub-{subject_id}-{condition}-psd.npy'))
    freqs = np.load(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'sensor-level', 'psd-data', 'freqs',
                                        f'sub-{subject_id}-{condition}-freqs.npy'))

    return spectra, freqs

def get_freq_bands():
    freq_bands = {
        'theta': [4, 8],
        'alpha': [8, 13],
        'low_beta':[13, 16],
        'high_beta': [16, 30]
    }
    return freq_bands

def get_roi_dict():

    ROI_dict = {'frontal_l':[3, 4, 5, 8, 9, 10],
            'frontal_r':[38, 39, 40, 43, 44, 45],
            'parietal_l':[16, 17, 18, 19, 20, 21],
            'parietal_r':[53, 54, 55, 56, 57, 58],
            'occipital_l':[24, 25, 26],
            'occipital_r':[61, 62, 63],
            'all':list(range(64))
}
    return ROI_dict

def get_freq_ind(freq_band):

    lower_bound = freq_band[0]*10
    upper_bound = freq_band[1]*10
    
    return lower_bound, upper_bound

def extract_power(spectrum, freq_band, ROI):

    lower_bound, upper_bound = get_freq_ind(freq_band)
    spectrum = spectrum[ROI].mean(axis=0)
    power = spectrum[lower_bound:upper_bound].mean(axis=0)
    return power

def get_all_power(subject_id, condition):

    spectra, _ = get_data(subject_id, condition)
    freq_bands = get_freq_bands()
    ROI_dict = get_roi_dict()
    powers = {}
    for freq_name in freq_bands.keys():
        powers[freq_name] = {}
        for ROI in ROI_dict:
            powers[freq_name][ROI] = extract_power(spectra, freq_bands[freq_name], ROI_dict[ROI])

    return powers

def save_all_powers(subject_id, condition):

    input_dir, _ = get_paths()
    powers = get_all_power(subject_id, condition)

    df = pd.DataFrame.from_dict(powers, orient='index').T
    
    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'sensor-level', 'power-df')):
        os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'sensor-level', 'power-df'))

    df.to_csv(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'sensor-level', 'power-df', f'sub-{subject_id}-{condition}-power.csv'))


############################################################################################################
# source-level

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
                                            'preprocessing', 'additional', '07-epochs-10s', f'sub-{subject_id}-{condition}-10s-epo.fif'))
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

def get_psd_src(subject_id, condition, ROI):

    input_dir, _ = get_paths()

    #if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'source-level', 'psd-data', 'spectrum',
    #                                   f'sub-{subject_id}-{condition}-{ROI}-psd.npy')):
    compute_source_psd_rs(subject_id, condition)
    spectra = np.load(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'source-level', 'psd-data', 'spectrum',  f'sub-{subject_id}-{condition}-{ROI}-psd.npy'))
    freqs = np.load(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'source-level', 'psd-data', 'freqs', f'sub-{subject_id}-{condition}-{ROI}-freqs.npy'))
    spectrum = spectra.mean(axis=0)

    return spectrum, freqs

def extract_power_src(spectrum, freq_band):

    lower_bound, upper_bound = get_freq_ind(freq_band)
    power = spectrum[lower_bound:upper_bound].mean(axis=0)
    return power

def get_all_power_src(subject_id, condition):

    rois = ['caudalmiddlefrontal-lh', 'caudalmiddlefrontal-rh', 'inferiorparietal-lh', 'inferiorparietal-rh',
            'lateraloccipital-lh', 'lateraloccipital-rh']

    freq_bands = get_freq_bands()
    powers = {}
    for freq_name in freq_bands.keys():
        powers[freq_name] = {}
        for ROI in rois:
            spectrum, freqs = get_psd_src(subject_id, condition, ROI)
            powers[freq_name][ROI] = extract_power_src(spectrum, freq_bands[freq_name])

    return powers

def save_all_powers_src(subject_id, condition):

    input_dir, _ = get_paths()
    powers = get_all_power_src(subject_id, condition)

    df = pd.DataFrame.from_dict(powers, orient='index').T
    
    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'source-level', 'power-df')):
        os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'source-level', 'power-df'))

    df.to_csv(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'source-level', 'power-df', f'sub-{subject_id}-{condition}-power.csv'))

