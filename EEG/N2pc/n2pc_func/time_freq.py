import mne
from mne.stats import permutation_cluster_test
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

############################################################################################################
# additional functions for contra-ipsi comparisons
############################################################################################################

def run_f_test_latdiff(sbj_list1: list, grpn1: str, sbj_list2: list, grpn2: str, swp_id: list, thresh: float, input_dir: str):
    ''' runs a f-test on the time-frequency representations of two groups of subjects. Adjusted for contra-ipsi comparisons.'''
    freqs = np.arange(8, 13, 1)
    tfr_l_epo1, tfr_r_epo1, times = stack_tfr_latdiff(sbj_list1, swp_id, freqs, input_dir)
    tfr_l_epo2, tfr_r_epo2, _ = stack_tfr_latdiff(sbj_list2, swp_id, freqs, input_dir)
    target_side = {'left':[tfr_l_epo1, tfr_l_epo2], 'right':[tfr_r_epo1, tfr_r_epo2]}
    figs = {}
    for side, (tfr_epo1, tfr_epo2) in target_side.items():
        F_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
            [tfr_epo1, tfr_epo2],
            out_type="mask",
            n_permutations=1000,
            threshold=thresh,
            tail=0,
            seed=np.random.default_rng(seed=8675309),
        )
        figs[side] = plot_stat_tfr_latdiff(
            tfr_epo1,
            grpn1,
            tfr_epo2,
            grpn2,
            F_obs,
            clusters,
            cluster_p_values,
            times,
            freqs,
            side,
        )

    outdir = os.path.join(input_dir, 'all_subj', 'N2pc', 'time_freq', 'stats_latdiff')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for side, fig in figs.items():
        fname = os.path.join(outdir, f'{grpn1}_VS_{grpn2}_{side}_thresh{thresh}_tfr_stat.png')
        fig.savefig(os.path.join(outdir, fname), dpi=300)
    return figs


def plot_stat_tfr_latdiff(tfr_epo1, grpn1, tfr_epo2, grpn2, F_obs, clusters, cluster_pval,  times, freqs, side):
    ''' Plots the results of the f-test on the time-frequency representations.

    Parameters
    ----------
    tfr_epo1 : np.array
        Time-frequency representation of the first group, shape (n_epochs, n_freqs, n_times).
    grpn1 : str
        Name of the first group.
    tfr_epo2 : np.array
        Time-frequency representation of the second group, shape (n_epochs, n_freqs, n_times).
    grpn2 : str
        Name of the second group.
    F_obs : np.array
        F-values.
    clusters : list of np.array
        List of clusters.
    cluster_pval : np.array
        Cluster p-values.
    times : np.array
        Array of time points for plotting.
    freqs : np.array
        Array of frequencies for plotting.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure.
    '''

    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")

    evoked_power_1 = tfr_epo1.mean(axis=0)
    evoked_power_2 = tfr_epo2.mean(axis=0)
    evoked_power_contrast = evoked_power_1 - evoked_power_2
    signs = np.sign(evoked_power_contrast)
    
    F_obs_plot = np.nan * np.ones_like(F_obs)
    for c, p_val in zip(clusters, cluster_pval):
        if p_val <= 0.05:
            F_obs_plot[c] = F_obs[c] * signs[c]

    ax.imshow(
        F_obs,
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        aspect="auto",
        origin="lower",
        cmap="gray",
    )
    max_F = np.nanmax(abs(F_obs_plot))
    ax.imshow(
        F_obs_plot,
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=-max_F,
        vmax=max_F,
    )

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Induced power {grpn1} VS {grpn2} \n target {side}")
    sns.despine()
    plt.tight_layout()
    plt.show()
    return fig

def stack_tfr_latdiff(subject_list, swp_id, freqs, input_dir):
    ''' Stacks the time-frequency representations of a list of subjects so they
    can be compared to another group.
    IMPORTANT : this time the epochs are averaged by subject before stacking.'''
    tfr_l_list = []
    tfr_r_list = []
    for subject_id in subject_list:
        if subject_id in swp_id:
            epochs = load_data(subject_id, True, input_dir)
        else:
            epochs = load_data(subject_id, False, input_dir)
        times = epochs.times * 1e3
        target_l_tfr, target_r_tfr = extract_latdiff(epochs, freqs)
        tfr_l_list.append(target_l_tfr)
        tfr_r_list.append(target_r_tfr)
    # stack -> axis 0 : subjects, axis 1 : freqs, axis 2 : times
    return np.stack(tfr_l_list), np.stack(tfr_r_list), times

def load_data(subject_id, swp, input_dir):
    ''' Loads the epochs of a single subject. If the epochs were swapped (lesion in the left hemisphere),
    the swapped epochs are loaded.'''
    if swp:
        fname = os.path.join(
            input_dir, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs',
            'swapped_epochs', f'sub-{subject_id}-cleaned_epochs-N2pc-swp.fif'
        )
    else:
        fname = os.path.join(
            input_dir, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs',
            f'sub-{subject_id}-cleaned_epochs-N2pc.fif'
        )
    epochs = mne.read_epochs(fname) 
    return epochs

def extract_latdiff(epochs, freqs):
    ''' Extracts the time-frequency representations for the contra and ipsi conditions of a single subject.
    Returns two tfr objects: one for target_l and one for target_r. 
    IMPORTANT : this time the epochs are averaged by subject.'''
    target_l = epochs[
        'dis_mid/target_l', 
        'dis_bot/target_l', 
        'no_dis/target_l', 
        'dis_right/target_l'
    ]
    target_r = epochs[
        'dis_mid/target_r', 
        'dis_bot/target_r', 
        'no_dis/target_r', 
        'dis_left/target_r'
    ]
    # contra - ipsi when target left : PO8 - PO7, when target right : PO7 - PO8
    chan_dict = {
        'target_l_ipsi' : target_l.copy().pick_channels(['PO7']),
        'target_l_contra' : target_l.copy().pick_channels(['PO8']),
        'target_r_ipsi' : target_r.copy().pick_channels(['PO8']),
        'target_r_contra' : target_r.copy().pick_channels(['PO7']),
    }
    tfr_dict = {chan: cmpt_tfr(epochs, freqs) for chan, epochs in chan_dict.items()}
    
    target_r_tfr = tfr_dict['target_r_contra'] - tfr_dict['target_r_ipsi']
    target_l_tfr = tfr_dict['target_l_contra'] - tfr_dict['target_l_ipsi']
    print(target_l_tfr.shape)

    target_l_tfr = np.mean(target_l_tfr, axis=0)
    target_r_tfr = np.mean(target_r_tfr, axis=0)
    print(target_l_tfr.shape)

    return target_l_tfr, target_r_tfr

############################################################################################################
# funcs for statistical analysis of time-frequency representations
############################################################################################################


def run_f_test_tfr(sbj_list1: list, grpn1: str, sbj_list2: list, grpn2: str, ch_name: str, swp_id: list, thresh: float, input_dir: str):
    ''' runs a f-test on the time-frequency representations of two groups of subjects.
    The test is done on a single channel.
    The results are plotted and saved in the output directory.
    
    Parameters
    ----------
    sbj_list1 : list of str
        List of subject IDs for the first group. (format '01')
    grpn1 : str
        Name of the first group.
    sbj_list2 : list of str
        List of subject IDs for the second group. (format '01')
    grpn2 : str
        Name of the second group.
    ch_name : str
        Channel name.
    swp_id : list of str
        List of subject IDs for which the epochs were swapped (lesion in the left hemisphere).
    input_dir : str
        Path to the data directory.

    Returns
    -------
    None.
    '''
    
    freqs = np.arange(8, 13, 1)

    tfr_epo1, times = stack_tfr(sbj_list1, swp_id, freqs, ch_name, input_dir)
    tfr_epo2, _ = stack_tfr(sbj_list2, swp_id, freqs, ch_name, input_dir)
    F_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
        [tfr_epo1, tfr_epo2],
        out_type="mask",
        n_permutations=1000,
        threshold=thresh,
        tail=0,
        seed=np.random.default_rng(seed=8675309),
    )
    fig = plot_stat_tfr(tfr_epo1,
        grpn1,
        tfr_epo2,
        grpn2,
        F_obs,
        clusters,
        cluster_p_values,
        times,
        freqs,
        ch_name,
    )

    outdir = os.path.join(input_dir, 'all_subj', 'N2pc', 'time_freq', 'stats')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    fname = os.path.join(outdir, f'{grpn1}_VS_{grpn2}_{ch_name}_thresh{thresh}_tfr_stat.png')
    fig.savefig(os.path.join(outdir, fname), dpi=300)
    

def stack_tfr(subject_list, swp_id, freqs, ch_name, input_dir):
    ''' Concatenates the time-frequency representations of a list of subjects so they
    can be compared to another group.

    Parameters
    ----------
    subject_list : list of str
        List of subject IDs.
    swp_id : list of str
        List of subject IDs for which the epochs were swapped (lesion in the left hemisphere).
    freqs : list of float
        List of frequencies.
    ch_name : str
        Channel name (generally PO7 or PO8).
    input_dir : str
        Path to the input directory.

    Returns
    -------
    tfr_list : np.array
        Array of time-frequency representations, shape (n_epochs, n_freqs, n_times).
    times : np.array
        Array of time points for plotting.
    '''
    tfr_list = []
    for subject_id in subject_list:
        if subject_id in swp_id:
            fname = os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs',
                                  'swapped_epochs', f'sub-{subject_id}-cleaned_epochs-N2pc-swp.fif')
        else:
            fname = os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs',
                                  f'sub-{subject_id}-cleaned_epochs-N2pc.fif')
        epochs = mne.read_epochs(fname)
        epochs.info['bads'] = []
        epochs.pick_channels([ch_name])
        times = epochs.times * 1e3
        tfr = cmpt_tfr(epochs, freqs)
        tfr_list.append(tfr)

    return np.concatenate(tfr_list, axis=0), times

def cmpt_tfr(epochs, freqs):
    ''' Computes the time-frequency representation of a single channel.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object.
    freqs : list of float
        List of frequencies.

    Returns
    -------
    tfr.data : np.array
        Time-frequency representation, shape (n_epochs, n_freqs, n_times).
    '''

    decim = 1
    n_cycles = 1.5
    tfr_kwargs = dict(
        freqs=freqs,
        n_cycles=n_cycles,
        decim=decim,
        return_itc=False,
        average=False,
    )
    tfr = mne.time_frequency.tfr_morlet(epochs, **tfr_kwargs)
    tfr.apply_baseline(baseline=(-0.2, 0), mode='ratio')

    return tfr.data[:, 0, :, :] # single channel

def f_test_tfr(tfr_epo1, tfr_epo2, thresh=None, nperm=1000):
    ''' Computes a f-test on the time-frequency representations of two groups of subjects.

    Parameters
    ----------
    tfr_epo1 : np.array
        Time-frequency representation of the first group, shape (n_epochs, n_freqs, n_times).
    tfr_epo2 : np.array
        Time-frequency representation of the second group, shape (n_epochs, n_freqs, n_times).
    thresh : float
        Threshold for the cluster test (critical F-value). Set to None for automatic selection
        to correspond to pval = .05.
    nperm : int
        Number of permutations.

    Returns
    -------
    F_obs : np.array
        F-values.
    clusters : list of np.array
        List of clusters.
    cluster_p_values : np.array
        Cluster p-values.
    H0 : np.array
        Permutation distribution.
    '''

    F_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
        [tfr_epo1, tfr_epo2],
        out_type="mask",
        n_permutations=nperm,
        threshold=thresh, # None -> automatic p-val 0.05
        tail=0,
        seed=np.random.default_rng(seed=8675309),
    )

    return F_obs, clusters, cluster_p_values, H0

def plot_stat_tfr(tfr_epo1, grpn1, tfr_epo2, grpn2, F_obs, clusters, cluster_pval,  times, freqs, ch_name):
    ''' Plots the results of the f-test on the time-frequency representations.

    Parameters
    ----------
    tfr_epo1 : np.array
        Time-frequency representation of the first group, shape (n_epochs, n_freqs, n_times).
    grpn1 : str
        Name of the first group.
    tfr_epo2 : np.array
        Time-frequency representation of the second group, shape (n_epochs, n_freqs, n_times).
    grpn2 : str
        Name of the second group.
    F_obs : np.array
        F-values.
    clusters : list of np.array
        List of clusters.
    cluster_pval : np.array
        Cluster p-values.
    times : np.array
        Array of time points for plotting.
    freqs : np.array
        Array of frequencies for plotting.
    ch_name : str
        Channel name.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure.
    '''

    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")

    evoked_power_1 = tfr_epo1.mean(axis=0)
    evoked_power_2 = tfr_epo2.mean(axis=0)
    evoked_power_contrast = evoked_power_1 - evoked_power_2
    signs = np.sign(evoked_power_contrast)
    
    F_obs_plot = np.nan * np.ones_like(F_obs)
    for c, p_val in zip(clusters, cluster_pval):
        if p_val <= 0.05:
            F_obs_plot[c] = F_obs[c] * signs[c]

    ax.imshow(
        F_obs,
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        aspect="auto",
        origin="lower",
        cmap="gray",
    )
    max_F = np.nanmax(abs(F_obs_plot))
    ax.imshow(
        F_obs_plot,
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=-max_F,
        vmax=max_F,
    )

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Induced power {grpn1} VS {grpn2} - {ch_name}")
    sns.despine()
    plt.tight_layout()

    return fig

############################################################################################################
# exploratory tfr visualization 
############################################################################################################

def get_tfr_scalp_single_subj(subject_id, input_dir, output_dir):
    '''
    Computes and saves time-frequency representations for a single subject.
    Frequencies between 1 and 30 Hz are used.

    Parameters
    ----------
    subject_id : str
        Subject ID.
    input_dir : str
        Path to the input directory.
    output_dir : str
        Path to the output directory.

    Returns
    -------
    None.
    '''
    # Load the epochs
    epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-N2pc.fif'))
    epochs.info['bads']=[]
    # divide the epochs into conditions
    event_dict = epochs.event_id
    dis_mid_target_l = mne.concatenate_epochs([epochs['dis_top/target_l'], epochs['dis_bot/target_l']])
    dis_mid_target_r = mne.concatenate_epochs([epochs['dis_top/target_r'], epochs['dis_bot/target_r']])

    epochs_dict = {}
    for event in event_dict:
        if event.startswith('dis_top') or event.startswith('dis_bot'):
            continue
        else:
            epochs_dict[event] = epochs[event]
    epochs_dict['dis_mid/target_l'] = dis_mid_target_l
    epochs_dict['dis_mid/target_r'] = dis_mid_target_r
    epochs_dict['all'] = epochs

    # Compute the time-frequency representation
    freqs = np.arange(1, 30, 1)
    n_cycles = freqs / 2.

    tfr_dict = {}
    for condition, epochs in epochs_dict.items():
        condition = condition.replace('/', '_')
        tfr = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, average=True)
        tfr_dict[condition] = tfr

    # Save the time-frequency representation
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'time_freq', 'tfr-data')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'time_freq', 'tfr-data'))

    for condition, tfr in tfr_dict.items():
        tfr.save(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'time_freq', 'tfr-data', f'sub-{subject_id}-{condition}-tfr.hdf5'), overwrite=True)
        print(f'================= {condition} data tfr saved for {subject_id}')

    return None

def get_tfr_scalp_population(input_dir, output_dir, subject_list, population):
    '''
    Fetch the tfr data for each subject of the population and computes the mean tfr for each condition.

    Parameters
    ----------
    input_dir : str
        Path to the input directory.
    output_dir : str
        Path to the output directory.
    subject_list : list of str
        List of subject IDs.
    population : str
        Population name.


    Returns
    -------
    None.
    '''

    all = []
    dis_mid_target_l = []
    dis_mid_target_r = []
    dis_right_target_l = []
    dis_left_target_r = []
    no_dis_target_l = []
    no_dis_target_r = []
    all_lists = [all, dis_mid_target_l, dis_mid_target_r, dis_right_target_l, dis_left_target_r, no_dis_target_l, no_dis_target_r]

    for subject_id in subject_list:
        # Load the time-frequency representation
        tfr_dict = {}
        for condition in ['all', 'dis_mid_target_l', 'dis_mid_target_r', 'dis_right_target_l', 'dis_left_target_r', 'no_dis_target_l', 'no_dis_target_r']:
            tfr = mne.time_frequency.read_tfrs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'time_freq', 'tfr-data', f'sub-{subject_id}-{condition}-tfr.hdf5'))[0]
            tfr_dict[condition] = tfr

        # Append the data to the lists
        for i, condition in enumerate(tfr_dict):
            tfr = tfr_dict[condition]
            all_lists[i].append(tfr.data)

        info = tfr.info
        times = tfr.times
        freqs = tfr.freqs

    # Compute the mean tfr for each condition
    mean_tfr_dict = {}
    for condition, tfr_list in zip(['all', 'dis_mid_target_l', 'dis_mid_target_r', 'dis_right_target_l', 'dis_left_target_r', 'no_dis_target_l', 'no_dis_target_r'], all_lists):
        mean_tfr_dict[condition] = np.mean(tfr_list, axis=0)
    
    # recreate the mne.time_frequency.tfr object / nave = 500 is completely arbitrary
    mean_tfr_dict = {condition: mne.time_frequency.tfr.AverageTFR(info=info, data=tfr, times=times, freqs=freqs, nave=500) for condition, tfr in mean_tfr_dict.items()}

    # Save the time-frequency representation
    if not os.path.exists(os.path.join(output_dir, 'all_subj', 'N2pc', 'time_freq', 'tfr-data', population)):
        os.makedirs(os.path.join(output_dir, 'all_subj', 'N2pc', 'time_freq', 'tfr-data', population))

    for condition, tfr in mean_tfr_dict.items():
        tfr.save(os.path.join(output_dir, 'all_subj', 'N2pc', 'time_freq', 'tfr-data', population, f'{population}-{condition}-tfr.hdf5'), overwrite=True)
        print(f'================= {condition} data tfr saved for {population}')

    return None

def plot_tfr_single_subj(subject_id, input_dir, output_dir):
    '''
    Plots time-frequency representations for a single subject.

    Parameters
    ----------
    subject_id : str
        Subject ID.
    input_dir : str
        Path to the input directory.
    output_dir : str
        Path to the output directory.

    Returns
    -------
    None.
    '''

    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'time_freq', 'tfr-data', f'sub-{subject_id}-all-tfr.hdf5')):
        print(f'================= tfr data not found for {subject_id}, computing it now...')
        get_tfr_scalp_single_subj(subject_id, input_dir, output_dir)

    # Load the tfr objects
    tfr_dict = {}
    for condition in ['all', 'dis_mid_target_l', 'dis_mid_target_r', 'dis_right_target_l', 'dis_left_target_r', 'no_dis_target_l', 'no_dis_target_r']:
        tfr = mne.time_frequency.read_tfrs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'time_freq', 'tfr-data', f'sub-{subject_id}-{condition}-tfr.hdf5'))[0]
        tfr_dict[condition] = tfr

    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'time_freq', 'tfr-plots', 'joint')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'time_freq', 'tfr-plots', 'joint'))
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'time_freq', 'tfr-plots', 'topo')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'time_freq', 'tfr-plots', 'topo'))
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'time_freq', 'tfr-plots', 'occip')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'time_freq', 'tfr-plots', 'occip'))

    # Plot the time-frequency representation
    for condition, tfr in tfr_dict.items():
        fig1 = tfr.plot_joint(timefreqs=((0.1, 10),(0.2,10),(0.3,10)), tmax=0.6, fmin=8, fmax=30, vmin=0, vmax=0.0000000005,
                               topomap_args={'vlim': (0,0.0000000015)},
                               title=f'sub {subject_id} - {condition}', colorbar=True, show=False)
        fig2 = tfr.plot_topo(title=f'sub-{subject_id}', tmax=0.6, fmin=8, fmax=30, vmin=0, vmax=0.0000000005, show=False)
        fig3 = tfr.plot(picks=['PO3', 'P7', 'PO7', 'P9', 'O1'], combine='mean', title=f'{subject_id} - {condition} - occip left', fmin=8, fmax=30, vmin=0, vmax=0.0000000009, show=False)[0]
        fig4 = tfr.plot(picks=['PO4', 'P8', 'PO8', 'P10', 'O2'], combine='mean', title=f'{subject_id} - {condition} - occip right', fmin=8, fmax=30, vmin=0, vmax=0.0000000009, show=False)[0]
        fig5 = tfr.plot_joint(timefreqs=((0.1, 18),(0.3,18),(0.5,18)), tmax=0.6, fmin=8, fmax=30, vmin=0, vmax=0.0000000005,
                               topomap_args={'vlim': (0,0.0000000015)},
                               title=f'sub {subject_id} - {condition}', colorbar=True, show=False)    
        fig6 = tfr.plot(picks=['PO3', 'P7', 'PO7', 'P9', 'O1',
                                 'PO4', 'P8', 'PO8', 'P10', 'O2'], combine='mean', title=f'{subject_id} - {condition} - occip', fmin=8, fmax=30, vmin=0, vmax=0.0000000009, show=False)[0]
        fig1.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'time_freq', 'tfr-plots', 'joint', f'sub-{subject_id}-{condition}-joint-tfr-plt.png'))
        fig2.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'time_freq', 'tfr-plots', 'topo', f'sub-{subject_id}-{condition}-topo-plot.png'))
        fig3.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'time_freq', 'tfr-plots', 'occip', f'sub-{subject_id}-{condition}-occip-left.png'))
        fig4.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'time_freq', 'tfr-plots', 'occip', f'sub-{subject_id}-{condition}-occip-right.png'))
        fig5.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'time_freq', 'tfr-plots', 'joint', f'sub-{subject_id}-{condition}-joint-tfr-plt-beta.png'))
        fig6.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'time_freq', 'tfr-plots', 'occip', f'sub-{subject_id}-{condition}-occip.png'))
        plt.close('all')
        print(f'================= {condition} plots done for {subject_id}')
    
    return None

def plot_tfr_population(input_dir, output_dir, subject_list, population):

    if not os.path.exists(os.path.join(output_dir, 'all_subj', 'N2pc', 'time_freq', 'tfr-data', population, f'{population}-all-tfr.hdf5')):
        print(f'================= tfr data not found for {population}, computing it now...')
        get_tfr_scalp_population(input_dir, output_dir, subject_list, population)

    # Load the tfr objects
    tfr_dict = {}
    for condition in ['all', 'dis_mid_target_l', 'dis_mid_target_r', 'dis_right_target_l', 'dis_left_target_r', 'no_dis_target_l', 'no_dis_target_r']:
        tfr = mne.time_frequency.read_tfrs(os.path.join(input_dir, 'all_subj', 'N2pc', 'time_freq', 'tfr-data', population, f'{population}-{condition}-tfr.hdf5'))[0]
        tfr_dict[condition] = tfr

    if not os.path.exists(os.path.join(output_dir, 'all_subj', 'N2pc', 'time_freq', 'tfr-plots', 'joint', population)):
        os.makedirs(os.path.join(output_dir, 'all_subj', 'N2pc', 'time_freq', 'tfr-plots', 'joint', population))
    if not os.path.exists(os.path.join(output_dir, 'all_subj', 'N2pc', 'time_freq', 'tfr-plots', 'topo', population)):
        os.makedirs(os.path.join(output_dir, 'all_subj', 'N2pc', 'time_freq', 'tfr-plots', 'topo', population))
    if not os.path.exists(os.path.join(output_dir, 'all_subj', 'N2pc', 'time_freq', 'tfr-plots', 'occip', population)):
        os.makedirs(os.path.join(output_dir, 'all_subj', 'N2pc', 'time_freq', 'tfr-plots', 'occip', population))
    
    # Plot the time-frequency representation
    for condition, tfr in tfr_dict.items():
        fig1 = tfr.plot_joint(timefreqs=((0.1, 10),(0.2,10),(0.3,10)), tmax=0.6, fmin=8, fmax=30, vmin=0, vmax=0.0000000005,
                               topomap_args={'vlim': (0,0.0000000015)},
                               title=f'{population} - {condition}', colorbar=True, show=False)
        fig5 = tfr.plot_joint(timefreqs=((0.1, 18),(0.3,18),(0.5,18)), tmax=0.6, fmin=8, fmax=30, vmin=0, vmax=0.0000000005,
                               topomap_args={'vlim': (0,0.0000000015)},
                               title=f'{population} - {condition}', colorbar=True, show=False)       
        fig2 = tfr.plot_topo(title=f'{population} - {condition}', tmax=0.6, fmin=8, fmax=30, vmin=0, vmax=0.0000000005, show=False)
        fig3 = tfr.plot(picks=['PO3', 'P7', 'PO7', 'P9', 'O1'], combine='mean', title=f'{population} - {condition} - occip left', fmin=8, fmax=30, vmin=0, vmax=0.0000000009, show=False)[0]
        fig4 = tfr.plot(picks=['PO4', 'P8', 'PO8', 'P10', 'O2'], combine='mean', title=f'{population} - {condition} - occip right', fmin=8, fmax=30, vmin=0, vmax=0.0000000009, show=False)[0]
        fig6 = tfr.plot(picks=['PO3', 'P7', 'PO7', 'P9', 'O1',
                               'PO4', 'P8', 'PO8', 'P10', 'O2'], combine='mean', title=f'{population} - {condition} - occip', fmin=8, fmax=30, vmin=0, vmax=0.0000000009, show=False)[0]
        fig1.savefig(os.path.join(output_dir, 'all_subj', 'N2pc', 'time_freq', 'tfr-plots', 'joint', population, f'{population}-{condition}-joint-tfr-plt.png'))
        fig2.savefig(os.path.join(output_dir, 'all_subj', 'N2pc', 'time_freq', 'tfr-plots', 'topo', population, f'{population}-{condition}-topo-plot.png'))
        fig3.savefig(os.path.join(output_dir, 'all_subj', 'N2pc', 'time_freq', 'tfr-plots', 'occip', population, f'{population}-{condition}-occip-left.png'))
        fig4.savefig(os.path.join(output_dir, 'all_subj', 'N2pc', 'time_freq', 'tfr-plots', 'occip', population, f'{population}-{condition}-occip-right.png'))
        fig5.savefig(os.path.join(output_dir, 'all_subj', 'N2pc', 'time_freq', 'tfr-plots', 'joint', population, f'{population}-{condition}-joint-tfr-plt-beta.png'))
        fig6.savefig(os.path.join(output_dir, 'all_subj', 'N2pc', 'time_freq', 'tfr-plots', 'occip', population, f'{population}-{condition}-occip.png'))
        plt.close('all')
        print(f'================= {condition} plots done for {population}')

    return None