import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne_connectivity import spectral_connectivity_time

def get_paths():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if 'nicolaspiron/Documents' in script_dir:
        input_dir = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/output_preproc'
        output_dir = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/output_preproc'
    elif 'shared_PULSATION' in script_dir:
        input_dir = '/home/nicolasp/shared_PULSATION/derivative'
        output_dir = '/home/nicolasp/shared_PULSATION/derivative'

    return input_dir, output_dir


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
        plot_conn_dict[name]=[average.data, std, mean, name]

    return plot_conn_dict

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
    for serie, std, mean, name in plot_conn_dict.values():
        ax.plot(t, serie, label=f'{name}, {mu}: {mean:.3f}, {sigma}: {std:.3f}')
        ax.fill_between(t, mean+std, mean-std, alpha=0.2)
        ax.hlines(mean, xmin=t[0], xmax=t[-1], colors='k', linestyles='--')
    ax.set_ylim(0, 0.8)
    ax.set_title(f'Global connectivity - sub {subject_id} - {cond_name} - {freqs_name} band')
    ax.set_ylabel('global connectivity')
    ax.set_xlabel('time (s)')
    ax.legend()

    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'plot', freqs_name)):
        os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'plot', freqs_name))
    fig.savefig(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'dynamic', 'sensor-level', 'plot', freqs_name,
                            f'sub-{subject_id}-{condition}-{freqs_name}-global-conn.png'), dpi=300)


def pipeline(subject_id, condition, band):

    conn_dict = load_conn_dict(subject_id, condition, band)
    plot_conn_dict = get_dynamic_global_plot_params(conn_dict)
    plot_dynamic_global_conn(plot_conn_dict, subject_id, condition, band)


if __name__ == '__main__':

    pipeline('01', 'RESTINGSTATECLOSE', ['alpha', [8, 13]])
