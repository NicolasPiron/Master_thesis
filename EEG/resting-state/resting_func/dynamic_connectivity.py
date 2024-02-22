import os
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
# sensor-level

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

    con_pli = spectral_connectivity_time(
        overlapping_epo,
        method='pli',
        mode="multitaper",
        sfreq=sfreq,
        freqs=freqs,
        faverage=True,
        n_jobs=1,
    )

    con_plv = spectral_connectivity_time(
        overlapping_epo,
        method='plv',
        mode="multitaper",
        sfreq=sfreq,
        freqs=freqs,
        faverage=True,
        n_jobs=1,
    )

    conn_dict = {'plv':con_plv.get_data(),
            'pli':con_pli.get_data()}
    

    for name in conn_dict.keys():
        if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'conn-data', name)):
            os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'conn-data', name))
        np.save(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'conn-data', name,
                              f'sub-{subject_id}-{condition}-{freqs_name}-{name}-conn.npy'), conn_dict[name])

    return conn_dict

def load_conn_dict(subject_id, condition, band:list):

    input_dir, _ = get_paths()
    freqs_name, _ = band

    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'conn-data', 'plv',
                                      f'sub-{subject_id}-{condition}-{freqs_name}-plv-conn.npy')):
        compute_conn(subject_id, condition, band)
    
    conn_dict = {'plv':np.load(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'conn-data', 'plv',
                                f'sub-{subject_id}-{condition}-{freqs_name}-plv-conn.npy')),
                    'pli':np.load(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'conn-data', 'pli',
                                f'sub-{subject_id}-{condition}-{freqs_name}-pli-conn.npy'))}
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

    plv_std = plot_conn_dict['plv'][1]
    plv_mean = plot_conn_dict['plv'][2]
    pli_std = plot_conn_dict['pli'][1]
    pli_mean = plot_conn_dict['pli'][2]
    plv_min = plot_conn_dict['plv'][3]
    plv_max = plot_conn_dict['plv'][4]
    pli_min = plot_conn_dict['pli'][3]
    pli_max = plot_conn_dict['pli'][4]

    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'metrics')):
        os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'metrics'))

    # save the metrics in a .csv file
    with open(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'metrics',
                            f'sub-{subject_id}-{condition}-{band[0]}-global-conn-metrics.csv'), 'w') as f:
          f.write('metric,plv,pli\n')
          f.write('std,%.3f,%.3f\n' % (plv_std, pli_std))
          f.write('mean,%.3f,%.3f\n' % (plv_mean, pli_mean))
          f.write('min,%.3f,%.3f\n' % (plv_min, pli_min))
          f.write('max,%.3f,%.3f\n' % (plv_max, pli_max))

def plot_dynamic_global_conn(plot_conn_dict, subject_id, condition, band):
    
    input_dir, _ = get_paths()
    freqs_name, _ = band

    length = plot_conn_dict['plv'][0].shape[0]
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
    ax.set_ylim(0, 0.8)
    ax.set_title(f'Global connectivity - sub {subject_id} - {cond_name} - {freqs_name} band')
    ax.set_ylabel('global connectivity')
    ax.set_xlabel('time (s)')
    ax.legend()

    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'plots', 'time-series')):
        os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'plots', 'time-series'))
    fig.savefig(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'plots', 'time-series',
                            f'sub-{subject_id}-{condition}-{freqs_name}-global-conn.png'), dpi=300)
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

def compute_ft_sw_conn(data, subject_id, condition, band:list):

    input_dir, _ = get_paths()

    freq_name, freqs = band
    sfreq = 512
    indices = ([4, 4, 4, 5, 5, 14], [5, 14, 15, 14, 15, 15])

    con = spectral_connectivity_time(
        data,
        method=['plv', 'pli'],
        mode="multitaper",
        sfreq=sfreq,
        indices=indices,
        freqs=freqs,
        faverage=True,
        n_jobs=1,
    )
    plv=con[0].get_data()
    pli=con[1].get_data()

    conn_dict = {'plv':plv,
            'pli':pli}
    
    for name in conn_dict.keys():
        if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 
                                           'dynamic', 'source-level', 'conn-data', name)):
            os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity',
                                        'dynamic', 'source-level', 'conn-data', name))
        np.save(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level',
                                'conn-data', name, f'sub-{subject_id}-{condition}-{freq_name}-{name}-conn.npy'), conn_dict[name])

def load_con_values_epochs(subject_id, condition, band:list):

    input_dir, o = get_paths()
    freq_name, _ = band
    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level',
                                    'conn-data', 'plv', f'sub-{subject_id}-{condition}-{freq_name}-plv-conn.npy')):
        data = load_stc_epochs(subject_id, condition)
        compute_ft_sw_conn(data, subject_id, condition, band)
    plv_data = np.load(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level',
                                'conn-data', 'plv', f'sub-{subject_id}-{condition}-{freq_name}-plv-conn.npy'))
    pli_data = np.load(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level',
                                'conn-data', 'pli', f'sub-{subject_id}-{condition}-{freq_name}-pli-conn.npy'))
    
    conn_dict = {'plv':plv_data, 'pli':pli_data}

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

def save_src_dc_metrics(plot_conn_dict,  subject_id, condition, band):
    '''Save the standard deviation and the mean of the global connectivity.
    '''

    input_dir, _ = get_paths()

    plv_std = plot_conn_dict['plv'][1]
    plv_mean = plot_conn_dict['plv'][2]
    pli_std = plot_conn_dict['pli'][1]
    pli_mean = plot_conn_dict['pli'][2]
    plv_min = plot_conn_dict['plv'][3]
    plv_max = plot_conn_dict['plv'][4]
    pli_min = plot_conn_dict['pli'][3]
    pli_max = plot_conn_dict['pli'][4]

    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'metrics')):
        os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'metrics'))

    # save the metrics in a .csv file
    with open(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'source-level', 'metrics',
                            f'sub-{subject_id}-{condition}-{band[0]}-global-conn-metrics.csv'), 'w') as f:
          f.write('metric,plv,pli\n')
          f.write('std,%.3f,%.3f\n' % (plv_std, pli_std))
          f.write('mean,%.3f,%.3f\n' % (plv_mean, pli_mean))
          f.write('min,%.3f,%.3f\n' % (plv_min, pli_min))
          f.write('max,%.3f,%.3f\n' % (plv_max, pli_max))
          

def plot_dynamic_src_conn(plot_conn_dict, subject_id, condition, band):
        
        input_dir, _ = get_paths()
        freqs_name, _ = band
    
        length = plot_conn_dict['plv'][0].shape[0]
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
                                f'sub-{subject_id}-{condition}-{freqs_name}-ftdy-conn.png'), dpi=300)
        plt.close()

def pipeline_src(subject_id, condition, band):

    conn_dict = load_con_values_epochs(subject_id, condition, band)
    plot_conn_dict = get_dynamic_src_plot_params(conn_dict)
    save_src_dc_metrics(plot_conn_dict, subject_id, condition, band)
    plot_dynamic_src_conn(plot_conn_dict, subject_id, condition, band)


if __name__ == '__main__':
    pipeline_src('01', 'RESTINGSTATEOPEN', ['alpha', (8, 12)])