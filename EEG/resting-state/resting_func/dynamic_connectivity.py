import os
import csv
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne_connectivity import spectral_connectivity_time
from mne.datasets import fetch_fsaverage


def get_paths():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if 'nicolaspiron/Documents' in script_dir:
        input_dir = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/output_preproc'
        output_dir = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/output_preproc'
    elif 'shared_PULSATION' in script_dir:
        input_dir = '/home/nicolasp/shared_PULSATION/derivative'
        output_dir = '/home/nicolasp/shared_PULSATION/derivative'

    return input_dir, output_dir


############################################################################################################
# ciPLV sensor-level. The code is originally adapted for several connectivity measures, hence the akward dict structure.


def sliding_window(raw, duration, overlap):
    '''Create a sliding window by epoching the raw data with overlaps. 

    Parameters
    ----------
    raw : mne.io.Raw
        The raw data.
    duration : int
        The duration of the window.
    overlap : int
        The overlap of the window.

    Returns
    -------
    overlapping_epo : mne.Epochs
        The epochs made with the sliding window.
    '''
    overlapping_events = mne.make_fixed_length_events(raw, duration=duration, overlap=overlap)
    overlapping_epo = mne.Epochs(raw, events=overlapping_events, tmin=0, tmax=10., baseline=None, preload=True)
    overlapping_epo.info['bads'] = []
    overlapping_epo.pick_types(eeg=True)

    return overlapping_epo

def compute_conn(subject_id, condition, band:list):

    input_dir, _ = get_paths()
    raw = mne.io.read_raw_fif(os.path.join(input_dir, f'sub-{subject_id}', condition, 'preprocessing', 'additional',
                                           '06-raw-annot-clean', f'sub-{subject_id}-raw-annot-clean-{condition}.fif'))

    sfreq = raw.info['sfreq']
    overlapping_epo = sliding_window(raw, duration=10, overlap=8)
    del raw

    freqs_name, freqs = band

    con = spectral_connectivity_time(
        overlapping_epo,
        method='ciplv',
        mode="multitaper",
        sfreq=sfreq,
        freqs=freqs,
        faverage=True,
        n_jobs=1,
    )

    conn_dict = {'ciplv':con.get_data()}

    for name in conn_dict.keys():
        if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'conn-data', name)):
            os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'conn-data', name))
        np.save(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'conn-data', name,
                              f'sub-{subject_id}-{condition}-{freqs_name}-{name}-conn.npy'), conn_dict[name])

    return conn_dict

def load_conn_dict(subject_id, condition, band:list):

    input_dir, _ = get_paths()
    freqs_name, _ = band

    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'conn-data', 'ciplv',
                                      f'sub-{subject_id}-{condition}-{freqs_name}-ciplv-conn.npy')):
        compute_conn(subject_id, condition, band)
    
    conn_dict = {'ciplv':np.load(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'conn-data', 'ciplv',
                                f'sub-{subject_id}-{condition}-{freqs_name}-ciplv-conn.npy'))}
    
    return conn_dict

def get_dynamic_global_plot_params(conn_dict):
    '''Get the parameters for the plot.

    Parameters
    ----------
    conn_dict : dict
        The dictionary with the connectivity.

    Returns
    -------
    plot_conn : dict
        The dictionary with the parameters for the plot.
    '''
    plot_conn_dict = {}

    for name, con in conn_dict.items():

        # remove the zeros
        masked_arr = np.ma.masked_equal(con, 0)
        sum_along_dim = masked_arr.sum(axis=1)
        count_nonzero = np.ma.count(masked_arr, axis=1)
        # average over the non-zero elements
        average = sum_along_dim / count_nonzero

        std = np.std(average.data)
        mean = np.mean(average.data)
        min = np.min(average.data)
        max = np.max(average.data)
        plot_conn_dict[name]=[average.data, std, mean, min, max, name]

    return plot_conn_dict

def save_dgc_metrics(plot_conn_dict,  subject_id, condition, band):
    '''Save the standard deviation and the mean of the global connectivity.
    '''

    input_dir, _ = get_paths()

    ciplv_std = plot_conn_dict['ciplv'][1]
    ciplv_mean = plot_conn_dict['ciplv'][2]
    ciplv_min = plot_conn_dict['ciplv'][3]
    ciplv_max = plot_conn_dict['ciplv'][4]

    # akward reorganization of the dict
    ciplv_dict = {'std':ciplv_std, 'mean':ciplv_mean, 'min':ciplv_min, 'max':ciplv_max}

    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'metrics')):
        os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'metrics'))

    # save the metrics in a .csv file
    with open(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'metrics',
                            f'sub-{subject_id}-{condition}-{band[0]}-ciplv-global-conn-metrics.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['metric', 'ciplv'])
        for key, value in ciplv_dict.items():
            writer.writerow([key, value])

def plot_dynamic_global_conn(plot_conn_dict, subject_id, condition, band):
    
    input_dir, _ = get_paths()
    freqs_name, _ = band

    length = plot_conn_dict['ciplv'][0].shape[0]
    start = 5
    stop = start + (length-1) * 2 + 1
    t = np.arange(start, stop, 2)

    if condition == 'RESTINGSTATEOPEN':
        cond_name = 'RS open'
    elif condition == 'RESTINGSTATECLOSE':
        cond_name = 'RS close'
    mu = r"$\mu$"
    sigma = r"$\sigma$"
    fig, ax = plt.subplots(figsize=(6, 4))
    for serie, std, mean, _, _, name in plot_conn_dict.values():
        ax.plot(t, serie, label=f'{name}, {mu}: {mean:.3f}, {sigma}: {std:.3f}')
        ax.fill_between(t, mean+std, mean-std, alpha=0.2)
        ax.hlines(mean, xmin=t[0], xmax=t[-1], colors='k', linestyles='--')
    ax.set_ylim(0, 0.3)
    ax.set_title(f'Global connectivity - sub {subject_id} - {cond_name} - {freqs_name} band')
    ax.set_ylabel('global connectivity')
    ax.set_xlabel('time (s)')
    ax.legend()

    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'plots', 'time-series')):
        os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'plots', 'time-series'))
    fig.savefig(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'plots', 'time-series',
                            f'sub-{subject_id}-{condition}-{freqs_name}-ciplv-global-conn.png'), dpi=300)
    plt.close()

def pipeline(subject_id, condition, band):

    conn_dict = load_conn_dict(subject_id, condition, band)
    plot_conn_dict = get_dynamic_global_plot_params(conn_dict)
    save_dgc_metrics(plot_conn_dict, subject_id, condition, band)
    plot_dynamic_global_conn(plot_conn_dict, subject_id, condition, band)


############################################################################################################
# source-level
    
def compute_src_on_sw(subject_id, condition):

    input_dir, _ = get_paths()
    src_path = os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level')
    for path in ['stc-epochs', 'cov', 'inv']:
        if not os.path.exists(os.path.join(src_path, 'src-data', path)):
            os.makedirs(os.path.join(src_path, 'src-data',path))

    raw = mne.io.read_raw_fif(os.path.join(input_dir, f'sub-{subject_id}', condition, 'preprocessing', 'additional',
                                           '06-raw-annot-clean', f'sub-{subject_id}-raw-annot-clean-{condition}.fif'))

    epochs = sliding_window(raw, duration=10, overlap=8)
    info = epochs.info
    del raw

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if 'nicolaspiron/Documents' in script_dir:
        subjects_dir = fetch_fsaverage(verbose=True)
    elif 'shared_PULSATION' in script_dir:
        subjects_dir = '/home/nicolasp/shared_PULSATION/MNE-fsaverage-data/fsaverage'
    else:
        raise Exception('Please specify the path to the fsaverage directory in the create_stc_epochs function.')

    # the same forward for every subject becauses no MRI scans
    if not os.path.exists(os.path.join(input_dir, 'sub-01', 'fwd', 'sub-01-fwd.fif')):
        raise Exception('Please compute the forward solution for subject 01 before running this function.')
    # compute noise covariance matrix and inverse operator
    fwd = mne.read_forward_solution(os.path.join(input_dir, 'sub-01', 'fwd', 'sub-01-fwd.fif'))
    cov = mne.compute_covariance(epochs)
    inverse_operator = mne.minimum_norm.make_inverse_operator(info, fwd, cov)

    # save the cov and inverse operator
    mne.write_cov(os.path.join(src_path, 'src-data', 'cov', f'sub-{subject_id}-cov.fif'), cov, overwrite=True)
    mne.minimum_norm.write_inverse_operator(os.path.join(src_path, 'src-data', 'inv', f'sub-{subject_id}-inv.fif'), inverse_operator, overwrite=True)

    lambda2 = 1. / 9.
    method = 'sLORETA'
    epochs.set_eeg_reference('average', projection=True)
    epochs.apply_proj()
    stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inverse_operator, lambda2=lambda2, method=method, pick_ori=None)

    # extract labels from the aparc parcellation
    labels = mne.read_labels_from_annot('', parc='aparc', subjects_dir=subjects_dir)
    labels = [label for label in labels if 'unknown' not in label.name]
    # get the label time course for each epoch -> n_epochs x n_labels x n_times
    label_ts_epochs = [mne.extract_label_time_course(stc, labels, fwd['src'], mode='pca_flip') for stc in stcs]
    del stcs
    label_ts_array = np.array(label_ts_epochs) 
    np.save(os.path.join(src_path, 'src-data', 'stc-epochs', f'sub-{subject_id}-{condition}-stc_epochs.npy'), label_ts_array)

def load_stc_epochs(subject_id, condition):
    
    input_dir, _ = get_paths()
    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level',
                                    'src-data', 'stc-epochs', f'sub-{subject_id}-{condition}-stc_epochs.npy')):
        compute_src_on_sw(subject_id, condition)

    stc_epochs = np.load(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level',
                                'src-data', 'stc-epochs', f'sub-{subject_id}-{condition}-stc_epochs.npy'))
    
    return stc_epochs

def compute_ft_sw_conn(data, subject_id, condition, band:list, spe_indices=True):

    input_dir, _ = get_paths()

    freq_name, freqs = band
    sfreq = 512

    if spe_indices:
        indices = ([4, 4, 4, 5, 5, 14], [5, 14, 15, 14, 15, 15])
        con = spectral_connectivity_time(
            data,
            method='ciplv',
            mode="multitaper",
            sfreq=sfreq,
            indices=indices,
            freqs=freqs,
            faverage=True,
            n_jobs=1,
        )
        ciplv=con.get_data()
        conn_dict = {'ciplv':ciplv}
    
    else:
        con = spectral_connectivity_time(
            data,
            method='ciplv',
            mode="multitaper",
            sfreq=sfreq,
            freqs=freqs,
            faverage=True,
            n_jobs=1,
        )
        ciplv=con.get_data()
        conn_dict = {'ciplv':ciplv}
    
    for name in conn_dict.keys():
        if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 
                                           'dynamic', 'source-level', 'conn-data', name)):
            os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity',
                                        'dynamic', 'source-level', 'conn-data', name))
        if spe_indices:
            np.save(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level',
                                'conn-data', name, f'sub-{subject_id}-{condition}-{freq_name}-{name}-conn.npy'), conn_dict[name])
        else:
            np.save(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level',
                                'conn-data', name, f'sub-{subject_id}-{condition}-global-{freq_name}-{name}-conn.npy'), conn_dict[name])

def load_con_values_epochs(subject_id, condition, band:list, spe_indices=True):

    input_dir, o = get_paths()
    freq_name, _ = band

    if spe_indices:
        if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level',
                                        'conn-data', 'ciplv', f'sub-{subject_id}-{condition}-{freq_name}-ciplv-conn.npy')):
            data = load_stc_epochs(subject_id, condition)
            compute_ft_sw_conn(data, subject_id, condition, band)
        ciplv_data = np.load(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level',
                                    'conn-data', 'ciplv', f'sub-{subject_id}-{condition}-{freq_name}-ciplv-conn.npy'))
    else:
        if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level',
                                        'conn-data', 'ciplv', f'sub-{subject_id}-{condition}-global-{freq_name}-ciplv-conn.npy')):
            data = load_stc_epochs(subject_id, condition)
            compute_ft_sw_conn(data, subject_id, condition, band, spe_indices=False)
        ciplv_data = np.load(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level',
                                    'conn-data', 'ciplv', f'sub-{subject_id}-{condition}-global-{freq_name}-ciplv-conn.npy'))

    
    conn_dict = {'ciplv':ciplv_data}

    return conn_dict

def get_dynamic_src_plot_params(conn_dict):
    '''Get the parameters for the plot.

    Parameters
    ----------
    conn_dict : dict
        The dictionary with the connectivity.

    Returns
    -------
    plot_conn : dict
        The dictionary with the parameters for the plot.
    '''
    plot_conn_dict = {}

    for name, con in conn_dict.items():

        average = con.mean(axis=1)

        std = np.std(average)
        mean = np.mean(average)
        min = np.min(average)
        max = np.max(average)
        plot_conn_dict[name]=[average, std, mean, min, max, name]

    return plot_conn_dict

def save_src_dc_metrics(plot_conn_dict,  subject_id, condition, band, spe_indices=True):
    '''Save the standard deviation and the mean of the global connectivity.
    '''

    input_dir, _ = get_paths()

    ciplv_std = plot_conn_dict['ciplv'][1]
    ciplv_mean = plot_conn_dict['ciplv'][2]
    ciplv_min = plot_conn_dict['ciplv'][3]
    ciplv_max = plot_conn_dict['ciplv'][4]

    # akward reorganization of the dict
    ciplv_dict = {'std':ciplv_std, 'mean':ciplv_mean, 'min':ciplv_min, 'max':ciplv_max}

    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'metrics')):
        os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'metrics'))

    # save the metrics in a .csv file
    if spe_indices:
        with open(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'metrics',
                                f'sub-{subject_id}-{condition}-{band[0]}-ciplv-global-conn-metrics.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['metric', 'ciplv'])
            for key, value in ciplv_dict.items():
                writer.writerow([key, value])
    else:
        with open(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'metrics',
                                f'sub-{subject_id}-{condition}-{band[0]}-ciplv-global-conn-metrics.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['metric', 'ciplv'])
            for key, value in ciplv_dict.items():
                writer.writerow([key, value])

def plot_dynamic_src_conn(plot_conn_dict, subject_id, condition, band, spe_indices=True):
        
        input_dir, _ = get_paths()
        freqs_name, _ = band

        if spe_indices:

            length = plot_conn_dict['ciplv'][0].shape[0]
            start = 5
            stop = start + (length-1) * 2 + 1
            t = np.arange(start, stop, 2)
        
            if condition == 'RESTINGSTATEOPEN':
                cond_name = 'RS open'
            elif condition == 'RESTINGSTATECLOSE':
                cond_name = 'RS close'
            mu = r"$\mu$"
            sigma = r"$\sigma$"
            fig, ax = plt.subplots(figsize=(6, 4))
            for serie, std, mean, _, _, name in plot_conn_dict.values():
                ax.plot(t, serie, label=f'{name}, {mu}: {mean:.3f}, {sigma}: {std:.3f}')
                ax.fill_between(t, mean+std, mean-std, alpha=0.2)
                ax.hlines(mean, xmin=t[0], xmax=t[-1], colors='k', linestyles='--')
            ax.set_title(f'fronto-parietal conn - sub {subject_id} - {cond_name} - {freqs_name} band')
            ax.set_ylabel('ft connectivity')
            ax.set_xlabel('time (s)')
            ax.legend()
        
            if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'plots', 'time-series')):
                os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'plots', 'time-series'))
            fig.savefig(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'plots', 'time-series',
                                    f'sub-{subject_id}-{condition}-{freqs_name}-ciplv-fpdy-conn.png'), dpi=300)
            plt.close()

        else:

            length = plot_conn_dict['ciplv'][0].shape[0]
            start = 5
            stop = start + (length-1) * 2 + 1
            t = np.arange(start, stop, 2)
        
            if condition == 'RESTINGSTATEOPEN':
                cond_name = 'RS open'
            elif condition == 'RESTINGSTATECLOSE':
                cond_name = 'RS close'
            mu = r"$\mu$"
            sigma = r"$\sigma$"
            fig, ax = plt.subplots(figsize=(6, 4))
            for serie, std, mean, _, _, name in plot_conn_dict.values():
                ax.plot(t, serie, label=f'{name}, {mu}: {mean:.3f}, {sigma}: {std:.3f}')
                ax.fill_between(t, mean+std, mean-std, alpha=0.2)
                ax.hlines(mean, xmin=t[0], xmax=t[-1], colors='k', linestyles='--')
            ax.set_title(f'global source conn - sub {subject_id} - {cond_name} - {freqs_name} band')
            ax.set_ylabel('ft connectivity')
            ax.set_xlabel('time (s)')
            ax.legend()
        
            if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'plots', 'time-series')):
                os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'plots', 'time-series'))
            fig.savefig(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'plots', 'time-series',
                                    f'sub-{subject_id}-{condition}-{freqs_name}-ciplv-global-conn.png'), dpi=300)
            plt.close()

def pipeline_src(subject_id, condition, band, spe_indices=False):

    conn_dict = load_con_values_epochs(subject_id, condition, band, spe_indices=spe_indices)
    plot_conn_dict = get_dynamic_src_plot_params(conn_dict)
    save_src_dc_metrics(plot_conn_dict, subject_id, condition, band, spe_indices=spe_indices)
    plot_dynamic_src_conn(plot_conn_dict, subject_id, condition, band, spe_indices=spe_indices)

############################################################################################################
# Get the FP connectivity by hemisphere
    
def get_hemi_conn(subject_id, cond, band:list):
    
    input_dir, _ = get_paths()
    band_name, _ = band
    ciplv_data = np.load(os.path.join(input_dir, f'sub-{subject_id}', cond, 'connectivity', 'dynamic', 'source-level', 'conn-data', 'ciplv',
                                     f'sub-{subject_id}-{cond}-{band_name}-ciplv-conn.npy'))
   
    left_edge = 1
    right_edge = 4
    left_ciplv_data = ciplv_data[:,left_edge,:]
    right_ciplv_data = ciplv_data[:,right_edge,:]

    conn_dict = {'left_ciplv':left_ciplv_data,
                 'right_ciplv':right_ciplv_data}

    return conn_dict

def get_dynamic_src_plot_params_hemi(conn_dict):

    plot_conn_dict = {}

    for name, con in conn_dict.items():

        std = np.std(con)
        mean = np.mean(con)
        min = np.min(con)
        max = np.max(con)
        plot_conn_dict[name]=[con, std, mean, min, max, name]

    return plot_conn_dict

def save_hemi_dc_metrics(plot_conn_dict,  subject_id, condition, band):

    input_dir, _ = get_paths()

    left_ciplv_std = plot_conn_dict['left_ciplv'][1]
    left_ciplv_mean = plot_conn_dict['left_ciplv'][2]
    right_ciplv_std = plot_conn_dict['right_ciplv'][1]
    right_ciplv_mean = plot_conn_dict['right_ciplv'][2]
    left_ciplv_min = plot_conn_dict['left_ciplv'][3]
    left_ciplv_max = plot_conn_dict['left_ciplv'][4]
    right_ciplv_min = plot_conn_dict['right_ciplv'][3]
    right_ciplv_max = plot_conn_dict['right_ciplv'][4]

        # akward reorganization of the dict
    ciplv_dict = {'left_std':left_ciplv_std, 'left_mean':left_ciplv_mean, 'left_min':left_ciplv_min, 'left_max':left_ciplv_max,
                    'right_std':right_ciplv_std, 'right_mean':right_ciplv_mean, 'right_min':right_ciplv_min, 'right_max':right_ciplv_max}

    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'metrics')):
        os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'metrics'))

    # save the metrics in a .csv file
    with open(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'metrics',
                            f'sub-{subject_id}-{condition}-{band[0]}-ciplv-hemi-conn-metrics.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['metric', 'ciplv'])
        for key, value in ciplv_dict.items():
            writer.writerow([key, value])

def hemi_pipeline(subject_id, condition, band):

    conn_dict = get_hemi_conn(subject_id, condition, band)
    plot_conn_dict = get_dynamic_src_plot_params_hemi(conn_dict)
    save_hemi_dc_metrics(plot_conn_dict, subject_id, condition, band)


############################################################################################################
# OLD CODE FOR PLV AND PLI
############################################################################################################


############################################################################################################
# sensor-level


# def compute_conn(subject_id, condition, band:list):

#     input_dir, _ = get_paths()
#     raw = mne.io.read_raw_fif(os.path.join(input_dir, f'sub-{subject_id}', condition, 'preprocessing', 'additional',
#                                            '06-raw-annot-clean', f'sub-{subject_id}-raw-annot-clean-{condition}.fif'))

#     sfreq = raw.info['sfreq']
#     overlapping_epo = sliding_window(raw, duration=10, overlap=8)
#     del raw

#     freqs_name, freqs = band

#     con_pli = spectral_connectivity_time(
#         overlapping_epo,
#         method='pli',
#         mode="multitaper",
#         sfreq=sfreq,
#         freqs=freqs,
#         faverage=True,
#         n_jobs=1,
#     )

#     con_plv = spectral_connectivity_time(
#         overlapping_epo,
#         method='plv',
#         mode="multitaper",
#         sfreq=sfreq,
#         freqs=freqs,
#         faverage=True,
#         n_jobs=1,
#     )

#     conn_dict = {'plv':con_plv.get_data(),
#             'pli':con_pli.get_data()}
    

#     for name in conn_dict.keys():
#         if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'conn-data', name)):
#             os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'conn-data', name))
#         np.save(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'conn-data', name,
#                               f'sub-{subject_id}-{condition}-{freqs_name}-{name}-conn.npy'), conn_dict[name])

#     return conn_dict

# def load_conn_dict(subject_id, condition, band:list):

#     input_dir, _ = get_paths()
#     freqs_name, _ = band

#     #if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'conn-data', 'plv',
#     #                                  f'sub-{subject_id}-{condition}-{freqs_name}-plv-conn.npy')):
#     compute_conn(subject_id, condition, band)
    
#     conn_dict = {'plv':np.load(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'conn-data', 'plv',
#                                 f'sub-{subject_id}-{condition}-{freqs_name}-plv-conn.npy')),
#                     'pli':np.load(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'conn-data', 'pli',
#                                 f'sub-{subject_id}-{condition}-{freqs_name}-pli-conn.npy'))}
#     return conn_dict

# def get_dynamic_global_plot_params(conn_dict):
#     '''Get the parameters for the plot.

#     Parameters
#     ----------
#     conn_dict : dict
#         The dictionary with the connectivity.

#     Returns
#     -------
#     plot_conn : dict
#         The dictionary with the parameters for the plot.
#     '''
#     plot_conn_dict = {}

#     for name, con in conn_dict.items():

#         # remove the zeros
#         masked_arr = np.ma.masked_equal(con, 0)
#         sum_along_dim = masked_arr.sum(axis=1)
#         count_nonzero = np.ma.count(masked_arr, axis=1)
#         # average over the non-zero elements
#         average = sum_along_dim / count_nonzero

#         std = np.std(average.data)
#         mean = np.mean(average.data)
#         min = np.min(average.data)
#         max = np.max(average.data)
#         plot_conn_dict[name]=[average.data, std, mean, min, max, name]

#     return plot_conn_dict

# def save_dgc_metrics(plot_conn_dict,  subject_id, condition, band):
#     '''Save the standard deviation and the mean of the global connectivity.
#     '''

#     input_dir, _ = get_paths()

#     plv_std = plot_conn_dict['plv'][1]
#     plv_mean = plot_conn_dict['plv'][2]
#     pli_std = plot_conn_dict['pli'][1]
#     pli_mean = plot_conn_dict['pli'][2]
#     plv_min = plot_conn_dict['plv'][3]
#     plv_max = plot_conn_dict['plv'][4]
#     pli_min = plot_conn_dict['pli'][3]
#     pli_max = plot_conn_dict['pli'][4]

#     if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'metrics')):
#         os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'metrics'))

#     # save the metrics in a .csv file
#     with open(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'metrics',
#                             f'sub-{subject_id}-{condition}-{band[0]}-global-conn-metrics.csv'), 'w') as f:
#           f.write('metric,plv,pli\n')
#           f.write('std,%.3f,%.3f\n' % (plv_std, pli_std))
#           f.write('mean,%.3f,%.3f\n' % (plv_mean, pli_mean))
#           f.write('min,%.3f,%.3f\n' % (plv_min, pli_min))
#           f.write('max,%.3f,%.3f\n' % (plv_max, pli_max))

# def plot_dynamic_global_conn(plot_conn_dict, subject_id, condition, band):
    
#     input_dir, _ = get_paths()
#     freqs_name, _ = band

#     length = plot_conn_dict['plv'][0].shape[0]
#     start = 5
#     stop = start + (length-1) * 2 + 1
#     t = np.arange(start, stop, 2)

#     if condition == 'RESTINGSTATEOPEN':
#         cond_name = 'RS open'
#     elif condition == 'RESTINGSTATECLOSE':
#         cond_name = 'RS close'
#     mu = r"$\mu$"
#     sigma = r"$\sigma$"
#     fig, ax = plt.subplots(figsize=(6, 4))
#     for serie, std, mean, _, _, name in plot_conn_dict.values():
#         ax.plot(t, serie, label=f'{name}, {mu}: {mean:.3f}, {sigma}: {std:.3f}')
#         ax.fill_between(t, mean+std, mean-std, alpha=0.2)
#         ax.hlines(mean, xmin=t[0], xmax=t[-1], colors='k', linestyles='--')
#     ax.set_ylim(0, 0.8)
#     ax.set_title(f'Global connectivity - sub {subject_id} - {cond_name} - {freqs_name} band')
#     ax.set_ylabel('global connectivity')
#     ax.set_xlabel('time (s)')
#     ax.legend()

#     if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'plots', 'time-series')):
#         os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'plots', 'time-series'))
#     fig.savefig(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'plots', 'time-series',
#                             f'sub-{subject_id}-{condition}-{freqs_name}-global-conn.png'), dpi=300)
#     plt.close()

# def pipeline(subject_id, condition, band):

#     conn_dict = load_conn_dict(subject_id, condition, band)
#     plot_conn_dict = get_dynamic_global_plot_params(conn_dict)
#     save_dgc_metrics(plot_conn_dict, subject_id, condition, band)
#     plot_dynamic_global_conn(plot_conn_dict, subject_id, condition, band)

############################################################################################################
# source-level
    
# def compute_src_on_sw(subject_id, condition):

#     input_dir, _ = get_paths()
#     src_path = os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level')
#     for path in ['stc-epochs', 'cov', 'inv']:
#         if not os.path.exists(os.path.join(src_path, 'src-data', path)):
#             os.makedirs(os.path.join(src_path, 'src-data',path))

#     raw = mne.io.read_raw_fif(os.path.join(input_dir, f'sub-{subject_id}', condition, 'preprocessing', 'additional',
#                                            '06-raw-annot-clean', f'sub-{subject_id}-raw-annot-clean-{condition}.fif'))

#     epochs = sliding_window(raw, duration=10, overlap=8)
#     info = epochs.info
#     del raw

#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     if 'nicolaspiron/Documents' in script_dir:
#         subjects_dir = fetch_fsaverage(verbose=True)
#     elif 'shared_PULSATION' in script_dir:
#         subjects_dir = '/home/nicolasp/shared_PULSATION/MNE-fsaverage-data/fsaverage'
#     else:
#         raise Exception('Please specify the path to the fsaverage directory in the create_stc_epochs function.')

#     # the same forward for every subject becauses no MRI scans
#     if not os.path.exists(os.path.join(input_dir, 'sub-01', 'fwd', 'sub-01-fwd.fif')):
#         raise Exception('Please compute the forward solution for subject 01 before running this function.')
#     # compute noise covariance matrix and inverse operator
#     fwd = mne.read_forward_solution(os.path.join(input_dir, 'sub-01', 'fwd', 'sub-01-fwd.fif'))
#     cov = mne.compute_covariance(epochs)
#     inverse_operator = mne.minimum_norm.make_inverse_operator(info, fwd, cov)

#     # save the cov and inverse operator
#     mne.write_cov(os.path.join(src_path, 'src-data', 'cov', f'sub-{subject_id}-cov.fif'), cov, overwrite=True)
#     mne.minimum_norm.write_inverse_operator(os.path.join(src_path, 'src-data', 'inv', f'sub-{subject_id}-inv.fif'), inverse_operator, overwrite=True)

#     lambda2 = 1. / 9.
#     method = 'sLORETA'
#     epochs.set_eeg_reference('average', projection=True)
#     epochs.apply_proj()
#     stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inverse_operator, lambda2=lambda2, method=method, pick_ori=None)

#     # extract labels from the aparc parcellation
#     labels = mne.read_labels_from_annot('', parc='aparc', subjects_dir=subjects_dir)
#     labels = [label for label in labels if 'unknown' not in label.name]
#     # get the label time course for each epoch -> n_epochs x n_labels x n_times
#     label_ts_epochs = [mne.extract_label_time_course(stc, labels, fwd['src'], mode='pca_flip') for stc in stcs]
#     del stcs
#     label_ts_array = np.array(label_ts_epochs) 
#     np.save(os.path.join(src_path, 'src-data', 'stc-epochs', f'sub-{subject_id}-{condition}-stc_epochs.npy'), label_ts_array)

# def load_stc_epochs(subject_id, condition):
    
#     input_dir, _ = get_paths()
#     if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level',
#                                     'src-data', 'stc-epochs', f'sub-{subject_id}-{condition}-stc_epochs.npy')):
#         compute_src_on_sw(subject_id, condition)

#     stc_epochs = np.load(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level',
#                                 'src-data', 'stc-epochs', f'sub-{subject_id}-{condition}-stc_epochs.npy'))
    
#     return stc_epochs

# def compute_ft_sw_conn(data, subject_id, condition, band:list, spe_indices=True):

#     input_dir, _ = get_paths()

#     freq_name, freqs = band
#     sfreq = 512

#     if spe_indices:
#         indices = ([4, 4, 4, 5, 5, 14], [5, 14, 15, 14, 15, 15])
#         con = spectral_connectivity_time(
#             data,
#             method=['plv', 'pli'],
#             mode="multitaper",
#             sfreq=sfreq,
#             indices=indices,
#             freqs=freqs,
#             faverage=True,
#             n_jobs=1,
#         )
#         plv=con[0].get_data()
#         pli=con[1].get_data()

#         conn_dict = {'plv':plv,
#                 'pli':pli}
    
#     else:
#         con = spectral_connectivity_time(
#             data,
#             method=['plv', 'pli'],
#             mode="multitaper",
#             sfreq=sfreq,
#             freqs=freqs,
#             faverage=True,
#             n_jobs=1,
#         )
#         plv=con[0].get_data()
#         pli=con[1].get_data()

#         conn_dict = {'plv':plv,
#                 'pli':pli}
    
#     for name in conn_dict.keys():
#         if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 
#                                            'dynamic', 'source-level', 'conn-data', name)):
#             os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity',
#                                         'dynamic', 'source-level', 'conn-data', name))
#         if spe_indices:
#             np.save(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level',
#                                 'conn-data', name, f'sub-{subject_id}-{condition}-{freq_name}-{name}-conn.npy'), conn_dict[name])
#         else:
#             np.save(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level',
#                                 'conn-data', name, f'sub-{subject_id}-{condition}-global-{freq_name}-{name}-conn.npy'), conn_dict[name])


# def load_con_values_epochs(subject_id, condition, band:list, spe_indices=True):

#     input_dir, o = get_paths()
#     freq_name, _ = band

#     if spe_indices:
#         if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level',
#                                         'conn-data', 'plv', f'sub-{subject_id}-{condition}-{freq_name}-plv-conn.npy')):
#             data = load_stc_epochs(subject_id, condition)
#             compute_ft_sw_conn(data, subject_id, condition, band)
#         plv_data = np.load(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level',
#                                     'conn-data', 'plv', f'sub-{subject_id}-{condition}-{freq_name}-plv-conn.npy'))
#         pli_data = np.load(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level',
#                                     'conn-data', 'pli', f'sub-{subject_id}-{condition}-{freq_name}-pli-conn.npy'))
#     else:
#         if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level',
#                                         'conn-data', 'plv', f'sub-{subject_id}-{condition}-global-{freq_name}-plv-conn.npy')):
#             data = load_stc_epochs(subject_id, condition)
#             compute_ft_sw_conn(data, subject_id, condition, band, spe_indices=False)
#         plv_data = np.load(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level',
#                                     'conn-data', 'plv', f'sub-{subject_id}-{condition}-global-{freq_name}-plv-conn.npy'))
#         pli_data = np.load(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level',
#                                     'conn-data', 'pli', f'sub-{subject_id}-{condition}-global-{freq_name}-pli-conn.npy'))
    
#     conn_dict = {'plv':plv_data, 'pli':pli_data}

#     return conn_dict

# def get_dynamic_src_plot_params(conn_dict):
#     '''Get the parameters for the plot.

#     Parameters
#     ----------
#     conn_dict : dict
#         The dictionary with the connectivity.

#     Returns
#     -------
#     plot_conn : dict
#         The dictionary with the parameters for the plot.
#     '''
#     plot_conn_dict = {}

#     for name, con in conn_dict.items():

#         average = con.mean(axis=1)

#         std = np.std(average)
#         mean = np.mean(average)
#         min = np.min(average)
#         max = np.max(average)
#         plot_conn_dict[name]=[average, std, mean, min, max, name]

#     return plot_conn_dict

# def save_src_dc_metrics(plot_conn_dict,  subject_id, condition, band, spe_indices=True):
#     '''Save the standard deviation and the mean of the global connectivity.
#     '''

#     input_dir, _ = get_paths()

#     plv_std = plot_conn_dict['plv'][1]
#     plv_mean = plot_conn_dict['plv'][2]
#     pli_std = plot_conn_dict['pli'][1]
#     pli_mean = plot_conn_dict['pli'][2]
#     plv_min = plot_conn_dict['plv'][3]
#     plv_max = plot_conn_dict['plv'][4]
#     pli_min = plot_conn_dict['pli'][3]
#     pli_max = plot_conn_dict['pli'][4]

#     if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'metrics')):
#         os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'metrics'))

#     # save the metrics in a .csv file
#     if spe_indices:
#         with open(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'metrics',
#                                 f'sub-{subject_id}-{condition}-{band[0]}-global-conn-metrics.csv'), 'w') as f:
#             f.write('metric,plv,pli\n')
#             f.write('std,%.3f,%.3f\n' % (plv_std, pli_std))
#             f.write('mean,%.3f,%.3f\n' % (plv_mean, pli_mean))
#             f.write('min,%.3f,%.3f\n' % (plv_min, pli_min))
#             f.write('max,%.3f,%.3f\n' % (plv_max, pli_max))
#     else:
#         with open(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'metrics',
#                                 f'sub-{subject_id}-{condition}-global-{band[0]}-global-conn-metrics.csv'), 'w') as f:
#             f.write('metric,plv,pli\n')
#             f.write('std,%.3f,%.3f\n' % (plv_std, pli_std))
#             f.write('mean,%.3f,%.3f\n' % (plv_mean, pli_mean))
#             f.write('min,%.3f,%.3f\n' % (plv_min, pli_min))
#             f.write('max,%.3f,%.3f\n' % (plv_max, pli_max))
          
# def plot_dynamic_src_conn(plot_conn_dict, subject_id, condition, band, spe_indices=True):
        
#         input_dir, _ = get_paths()
#         freqs_name, _ = band

#         if spe_indices:

#             length = plot_conn_dict['plv'][0].shape[0]
#             start = 5
#             stop = start + (length-1) * 2 + 1
#             t = np.arange(start, stop, 2)
        
#             if condition == 'RESTINGSTATEOPEN':
#                 cond_name = 'RS open'
#             elif condition == 'RESTINGSTATECLOSE':
#                 cond_name = 'RS close'
#             mu = r"$\mu$"
#             sigma = r"$\sigma$"
#             fig, ax = plt.subplots(figsize=(6, 4))
#             for serie, std, mean, _, _, name in plot_conn_dict.values():
#                 ax.plot(t, serie, label=f'{name}, {mu}: {mean:.3f}, {sigma}: {std:.3f}')
#                 ax.fill_between(t, mean+std, mean-std, alpha=0.2)
#                 ax.hlines(mean, xmin=t[0], xmax=t[-1], colors='k', linestyles='--')
#             ax.set_title(f'fronto-parietal conn - sub {subject_id} - {cond_name} - {freqs_name} band')
#             ax.set_ylabel('ft connectivity')
#             ax.set_xlabel('time (s)')
#             ax.legend()
        
#             if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'plots', 'time-series')):
#                 os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'plots', 'time-series'))
#             fig.savefig(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'plots', 'time-series',
#                                     f'sub-{subject_id}-{condition}-{freqs_name}-ftdy-conn.png'), dpi=300)
#             plt.close()

#         else:

#             length = plot_conn_dict['plv'][0].shape[0]
#             start = 5
#             stop = start + (length-1) * 2 + 1
#             t = np.arange(start, stop, 2)
        
#             if condition == 'RESTINGSTATEOPEN':
#                 cond_name = 'RS open'
#             elif condition == 'RESTINGSTATECLOSE':
#                 cond_name = 'RS close'
#             mu = r"$\mu$"
#             sigma = r"$\sigma$"
#             fig, ax = plt.subplots(figsize=(6, 4))
#             for serie, std, mean, _, _, name in plot_conn_dict.values():
#                 ax.plot(t, serie, label=f'{name}, {mu}: {mean:.3f}, {sigma}: {std:.3f}')
#                 ax.fill_between(t, mean+std, mean-std, alpha=0.2)
#                 ax.hlines(mean, xmin=t[0], xmax=t[-1], colors='k', linestyles='--')
#             ax.set_title(f'global source conn - sub {subject_id} - {cond_name} - {freqs_name} band')
#             ax.set_ylabel('ft connectivity')
#             ax.set_xlabel('time (s)')
#             ax.legend()
        
#             if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'plots', 'time-series')):
#                 os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'plots', 'time-series'))
#             fig.savefig(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'plots', 'time-series',
#                                     f'sub-{subject_id}-{condition}-{freqs_name}-global-conn.png'), dpi=300)
#             plt.close()

# def pipeline_src(subject_id, condition, band, spe_indices=True):

#     conn_dict = load_con_values_epochs(subject_id, condition, band, spe_indices=spe_indices)
#     plot_conn_dict = get_dynamic_src_plot_params(conn_dict)
#     save_src_dc_metrics(plot_conn_dict, subject_id, condition, band, spe_indices=spe_indices)
#     plot_dynamic_src_conn(plot_conn_dict, subject_id, condition, band, spe_indices=spe_indices)


# ############################################################################################################
# # Get the FP connectivity by hemisphere
    
# def get_hemi_conn(subject_id, cond, band:list):
    
#     input_dir, _ = get_paths()
#     band_name, _ = band
#     plv_data = np.load(os.path.join(input_dir, f'sub-{subject_id}', cond, 'connectivity', 'dynamic', 'source-level', 'conn-data', 'plv',
#                                      f'sub-{subject_id}-{cond}-{band_name}-plv-conn.npy'))
#     pli_data = np.load(os.path.join(input_dir, f'sub-{subject_id}', cond, 'connectivity', 'dynamic', 'source-level', 'conn-data', 'pli',
#                                         f'sub-{subject_id}-{cond}-{band_name}-pli-conn.npy'))
#     left_edge = 1
#     right_edge = 4
#     left_plv_data = plv_data[:,left_edge,:]
#     right_plv_data = plv_data[:,right_edge,:]
#     left_pli_data = pli_data[:,left_edge,:]
#     right_pli_data = pli_data[:,right_edge,:]

#     conn_dict = {'left_plv':left_plv_data,
#                  'right_plv':right_plv_data,
#                  'left_pli':left_pli_data,
#                  'right_pli':right_pli_data}

#     return conn_dict

# def get_dynamic_src_plot_params(conn_dict):

#     plot_conn_dict = {}

#     for name, con in conn_dict.items():

#         std = np.std(con)
#         mean = np.mean(con)
#         min = np.min(con)
#         max = np.max(con)
#         plot_conn_dict[name]=[con, std, mean, min, max, name]

#     return plot_conn_dict

# def save_hemi_dc_metrics(plot_conn_dict,  subject_id, condition, band):

#     input_dir, _ = get_paths()

#     left_plv_std = plot_conn_dict['left_plv'][1]
#     left_plv_mean = plot_conn_dict['left_plv'][2]
#     right_plv_std = plot_conn_dict['right_plv'][1]
#     right_plv_mean = plot_conn_dict['right_plv'][2]
#     left_pli_std = plot_conn_dict['left_pli'][1]
#     left_pli_mean = plot_conn_dict['left_pli'][2]
#     right_pli_std = plot_conn_dict['right_pli'][1]
#     right_pli_mean = plot_conn_dict['right_pli'][2]
#     left_plv_min = plot_conn_dict['left_plv'][3]
#     left_plv_max = plot_conn_dict['left_plv'][4]
#     right_plv_min = plot_conn_dict['right_plv'][3]
#     right_plv_max = plot_conn_dict['right_plv'][4]
#     left_pli_min = plot_conn_dict['left_pli'][3]
#     left_pli_max = plot_conn_dict['left_pli'][4]
#     right_pli_min = plot_conn_dict['right_pli'][3]
#     right_pli_max = plot_conn_dict['right_pli'][4]

#     if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'metrics')):
#         os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'metrics'))

#     # save the metrics in a .csv file
#     with open(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'metrics',
#                             f'sub-{subject_id}-{condition}-{band[0]}-hemi-conn-metrics.csv'), 'w') as f:
#         f.write('metric,left_plv,right_plv,left_pli,right_pli\n')
#         f.write('std,%.3f,%.3f,%.3f,%.3f\n' % (left_plv_std, right_plv_std, left_pli_std, right_pli_std))
#         f.write('mean,%.3f,%.3f,%.3f,%.3f\n' % (left_plv_mean, right_plv_mean, left_pli_mean, right_pli_mean))
#         f.write('min,%.3f,%.3f,%.3f,%.3f\n' % (left_plv_min, right_plv_min, left_pli_min, right_pli_min))
#         f.write('max,%.3f,%.3f,%.3f,%.3f\n' % (left_plv_max, right_plv_max, left_pli_max, right_pli_max))

# def hemi_pipeline(subject_id, condition, band):

#     conn_dict = get_hemi_conn(subject_id, condition, band)
#     plot_conn_dict = get_dynamic_src_plot_params(conn_dict)
#     save_hemi_dc_metrics(plot_conn_dict, subject_id, condition, band)


if __name__ == '__main__':

    pipeline_src('01', 'RESTINGSTATEOPEN', ['alpha', (8, 12)], spe_indices=False)
    pipeline_src('01', 'RESTINGSTATEOPEN', ['alpha', (8, 12)], spe_indices=True)
    hemi_pipeline('01', 'RESTINGSTATEOPEN', ['alpha', (8, 12)])