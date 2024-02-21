import os
import mne
import fooof
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    epochs.info['bads'] = []
    psd = epochs.compute_psd(fmin=0, fmax=30, method='multitaper')
    freqs = psd.freqs
    spectra = psd.get_data().mean(axis=0)
    return spectra, freqs

def save_psd(subject_id, condition):

    input_dir, _ = get_paths()
    epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'preprocessing',
                                          'additional', '07-epochs-10s', f'sub-{subject_id}-{condition}-10s-epo.fif'))
    psd, freqs = get_psd(epochs)

    for path in ['spectrum', 'freqs']:
        if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'psd-data', path)):
            os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'psd-data', path))

    np.save(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'psd-data', 'spectrum', f'sub-{subject_id}-{condition}-psd.npy'), psd)
    np.save(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'psd-data', 'freqs', f'sub-{subject_id}-{condition}-freqs.npy'), freqs)

def get_fooof_params(subject_id, condition):

    input_dir, _ = get_paths()

    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'psd-data', 'spectrum', f'sub-{subject_id}-{condition}-psd.npy')):
        save_psd(subject_id, condition)
    spectra = np.load(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'psd-data', 'spectrum', f'sub-{subject_id}-{condition}-psd.npy'))
    freqs = np.load(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'psd-data', 'freqs', f'sub-{subject_id}-{condition}-freqs.npy'))
    freq_range = [1, 30]

    return freqs, spectra, freq_range

def extract_ROI_spectrum(spectra, ROI):

    ROI_dict = {'frontal_l':[3, 4, 5, 8, 9, 10],
                'frontal_r':[38, 39, 40, 43, 44, 45],
                'parietal_l':[16, 17, 18, 19, 20, 21],
                'parietal_r':[53, 54, 55, 56, 57, 58],
                'occipital_l':[24, 25, 26],
                'occipital_r':[61, 62, 63],
    }
    ROI_idx = ROI_dict[ROI]
    spectrum = spectra[ROI_idx].mean(axis=0)

    return spectrum
    

def fit_ooof(freqs, spectrum, freq_range=[1, 30]):

    fm = fooof.FOOOF(peak_threshold=2, max_n_peaks=5, aperiodic_mode='knee')
    fm.fit(freqs, spectrum, freq_range)
    return fm

def save_params(subject_id, condition, ROI, fm):

    input_dir, _ = get_paths()

    peaks = fm.get_params('peak_params')
    peak_df = pd.DataFrame(peaks, columns=['CF', 'PW', 'BW'])

    res = fm.get_results()
    ap_df = pd.DataFrame(res.aperiodic_params.reshape(1,3), columns=['offset', 'knee', 'exponent'])
    fit_metrics = np.array([res.r_squared, res.error]).reshape(1,2)
    fit_df = pd.DataFrame(fit_metrics, columns=['r_squared', 'error'])

    for path in ['peaks', 'ap', 'fit']:
        if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'fooof', 'metrics', path)):
            os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'fooof', 'metrics', path))
    peak_df.to_csv(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'fooof', 'metrics', 'peaks', f'sub-{subject_id}-{condition}-{ROI}-peaks.csv'))
    ap_df.to_csv(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'fooof', 'metrics', 'ap', f'sub-{subject_id}-{condition}-{ROI}-ap.csv'))
    fit_df.to_csv(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'fooof', 'metrics', 'fit', f'sub-{subject_id}-{condition}-{ROI}-fit.csv'))

def save_fooof_plot(subject_id, condition, ROI, fm):

    input_dir, _ = get_paths()
    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'fooof', 'plots')):
        os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'fooof', 'plots'))
    file_name = os.path.join(input_dir, f'sub-{subject_id}', condition, 'psd', 'fooof', 'plots', f'sub-{subject_id}-{condition}-{ROI}-fooof.png')
    fm.plot(plot_peaks='shade', save_fig=True, file_name=file_name)
    plt.close()

def single_subj_pipeline(subject_id, condition, ROI):

    freqs, spectra, freq_range = get_fooof_params(subject_id, condition)
    spectrum = extract_ROI_spectrum(spectra, ROI)
    fm = fit_ooof(freqs, spectrum, freq_range)
    save_params(subject_id, condition, ROI, fm)
    save_fooof_plot(subject_id, condition, ROI, fm)
