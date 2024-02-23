import os
import mne
import fooof
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .src_rec_rs import compute_source_psd_rs
#from set_paths import get_paths # having trouble importing this function so I define the paths here. 

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

def get_fooof_params(subject_id, condition):
    '''Get the power spectral density and the frequencies of the epochs.
    If the .npy files do not exist, compute the PSD and save them.

    Parameters
    ----------
    subject_id : str
        The subject id.
    condition : str
        The condition (e.g. 'RESTINGSTATEOPEN' or 'RESTINGSTATECLOSE').
    
    Returns
    ------- 
    freqs : np.array
        The frequencies.
    spectra : np.array
        The power spectral density of the epochs.
    freq_range : list
        The frequency range for the FOOOF fit.
    '''

    input_dir, _ = get_paths()

    #if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'sensor-level', 'psd-data', 'spectrum', f'sub-{subject_id}-{condition}-psd.npy')):
    save_psd(subject_id, condition)
    spectra = np.load(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'sensor-level', 'psd-data', 'spectrum', f'sub-{subject_id}-{condition}-psd.npy'))
    freqs = np.load(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'sensor-level', 'psd-data', 'freqs', f'sub-{subject_id}-{condition}-freqs.npy'))
    freq_range = [2, 30]

    return freqs, spectra, freq_range

def extract_ROI_spectrum(spectra, ROI):
    '''Extract the power spectral density of a specific region of interest.

    Parameters
    ----------
    spectra : np.array
        The power spectral density of the epochs.
    ROI : str
        The region of interest (e.g. 'frontal_l', 'frontal_r', 'parietal_l', 'parietal_r', 'occipital_l', 'occipital_r', 'all').

    Returns
    -------
    spectrum : np.array
        The power spectral density of the region of interest.
    '''

    ROI_dict = {'frontal_l':[3, 4, 5, 8, 9, 10],
                'frontal_r':[38, 39, 40, 43, 44, 45],
                'parietal_l':[16, 17, 18, 19, 20, 21],
                'parietal_r':[53, 54, 55, 56, 57, 58],
                'occipital_l':[24, 25, 26],
                'occipital_r':[61, 62, 63],
    }
    if not ROI == 'all':
        ROI_idx = ROI_dict[ROI]
        spectrum = spectra[ROI_idx].mean(axis=0)
    else:
        spectrum = spectra.mean(axis=0)

    return spectrum
    
def fit_ooof(freqs, spectrum, freq_range=[2, 30]):
    '''Fit the power spectral density with the FOOOF model.

    Parameters
    ----------
    freqs : np.array
        The frequencies.
    spectrum : np.array
        The power spectral density of the epochs.
    freq_range : list
        The frequency range for the FOOOF fit.

    Returns
    -------
    fm : fooof.FOOOF
        The FOOOF model.
    '''

    fm = fooof.FOOOF(peak_threshold=2, max_n_peaks=3, aperiodic_mode='knee')
    fm.fit(freqs, spectrum, freq_range)
    return fm

def save_params(subject_id, condition, ROI, fm):
    '''Save the FOOOF parameters and the fit metrics as .csv files.

    Parameters
    ----------
    subject_id : str
        The subject id.
    condition : str
        The condition (e.g. 'RESTINGSTATEOPEN' or 'RESTINGSTATECLOSE').
    ROI : str
        The region of interest (e.g. 'frontal_l', 'frontal_r', 'parietal_l', 'parietal_r', 'occipital_l', 'occipital_r', 'all').
    fm : fooof.FOOOF
        The FOOOF model.

    Returns
    -------
    None
    '''

    input_dir, _ = get_paths()

    peaks = fm.get_params('peak_params')
    peak_df = pd.DataFrame(peaks, columns=['CF', 'PW', 'BW'])

    res = fm.get_results()
    ap_df = pd.DataFrame(res.aperiodic_params.reshape(1,3), columns=['offset', 'knee', 'exponent'])
    fit_metrics = np.array([res.r_squared, res.error]).reshape(1,2)
    fit_df = pd.DataFrame(fit_metrics, columns=['r_squared', 'error'])

    for path in ['peaks', 'ap', 'fit']:
        if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'sensor-level', 'fooof', 'metrics', path)):
            os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'sensor-level', 'fooof', 'metrics', path))
    peak_df.to_csv(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'sensor-level', 'fooof', 'metrics', 'peaks', f'sub-{subject_id}-{condition}-{ROI}-peaks.csv'))
    ap_df.to_csv(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'sensor-level', 'fooof', 'metrics', 'ap', f'sub-{subject_id}-{condition}-{ROI}-ap.csv'))
    fit_df.to_csv(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'sensor-level', 'fooof', 'metrics', 'fit', f'sub-{subject_id}-{condition}-{ROI}-fit.csv'))

def save_fooof_plot(subject_id, condition, ROI, fm):
    '''Save the FOOOF plot as a .png file.

    Parameters
    ----------
    subject_id : str
        The subject id.
    condition : str
        The condition (e.g. 'RESTINGSTATEOPEN' or 'RESTINGSTATECLOSE').
    ROI : str
        The region of interest (e.g. 'frontal_l', 'frontal_r', 'parietal_l', 'parietal_r', 'occipital_l', 'occipital_r', 'all').
    fm : fooof.FOOOF
        The FOOOF model.

    Returns
    -------
    None
    '''

    input_dir, _ = get_paths()
    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'sensor-level', 'fooof', 'plots')):
        os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'sensor-level', 'fooof', 'plots'))
    file_name = os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'sensor-level', 'fooof', 'plots', f'sub-{subject_id}-{condition}-{ROI}-fooof.png')
    fm.plot(plot_peaks='shade', save_fig=True, file_name=file_name)
    plt.close()

def single_subj_pipeline(subject_id, condition, ROI):
    '''Run the entire pipeline for a single subject.

    Parameters
    ----------
    subject_id : str
        The subject id.
    condition : str
        The condition (e.g. 'RESTINGSTATEOPEN' or 'RESTINGSTATECLOSE').
    ROI : str
        The region of interest (e.g. 'frontal_l', 'frontal_r', 'parietal_l', 'parietal_r', 'occipital_l', 'occipital_r', 'all').

    Returns
    -------
    None
    '''

    freqs, spectra, freq_range = get_fooof_params(subject_id, condition)
    spectrum = extract_ROI_spectrum(spectra, ROI)
    fm = fit_ooof(freqs, spectrum, freq_range)
    save_params(subject_id, condition, ROI, fm)
    save_fooof_plot(subject_id, condition, ROI, fm)


####################################################################################################
# FOOOF in a label
    
def get_fooof_params_src(subject_id, condition, ROI):

    input_dir, _ = get_paths()

    #if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'source-level', 'psd-data', 'spectrum',
    #                                   f'sub-{subject_id}-{condition}-{ROI}-psd.npy')):
    compute_source_psd_rs(subject_id, condition)
    spectra = np.load(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'source-level', 'psd-data', 'spectrum',  f'sub-{subject_id}-{condition}-{ROI}-psd.npy'))
    freqs = np.load(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'source-level', 'psd-data', 'freqs', f'sub-{subject_id}-{condition}-{ROI}-freqs.npy'))
    spectrum = spectra.mean(axis=0)
    freq_range = [2, 30]

    return freqs, spectrum, freq_range

def save_params_src(subject_id, condition, ROI, fm):

    input_dir, _ = get_paths()

    peaks = fm.get_params('peak_params')
    peak_df = pd.DataFrame(peaks, columns=['CF', 'PW', 'BW'])

    res = fm.get_results()
    ap_df = pd.DataFrame(res.aperiodic_params.reshape(1,3), columns=['offset', 'knee', 'exponent'])
    fit_metrics = np.array([res.r_squared, res.error]).reshape(1,2)
    fit_df = pd.DataFrame(fit_metrics, columns=['r_squared', 'error'])

    for path in ['peaks', 'ap', 'fit']:
        if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'source-level', 'fooof', 'metrics', path)):
            os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'source-level', 'fooof', 'metrics', path))
    peak_df.to_csv(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'source-level', 'fooof', 'metrics', 'peaks', f'sub-{subject_id}-{condition}-{ROI}-peaks.csv'))
    ap_df.to_csv(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'source-level', 'fooof', 'metrics', 'ap', f'sub-{subject_id}-{condition}-{ROI}-ap.csv'))
    fit_df.to_csv(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'source-level', 'fooof', 'metrics', 'fit', f'sub-{subject_id}-{condition}-{ROI}-fit.csv'))

def save_fooof_plot_src(subject_id, condition, ROI, fm):

    input_dir, _ = get_paths()
    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'source-level', 'fooof', 'plots')):
        os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'source-level', 'fooof', 'plots'))
    file_name = os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'source-level', 'fooof', 'plots', f'sub-{subject_id}-{condition}-{ROI}-fooof.png')
    fm.plot(plot_peaks='shade', save_fig=True, file_name=file_name)
    plt.close()

def single_subj_pipeline_src(subject_id, condition, ROI):

    freqs, spectrum, freq_range = get_fooof_params_src(subject_id, condition, ROI)
    fm = fit_ooof(freqs, spectrum, freq_range)
    save_params_src(subject_id, condition, ROI, fm)
    save_fooof_plot_src(subject_id, condition, ROI, fm)

