import mne
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

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

    # Plot the time-frequency representation
    for condition, tfr in tfr_dict.items():
        fig1 = tfr.plot_joint(timefreqs=((0.1, 10),(0.2,10),(0.3,10)), tmin=0, tmax=0.6, title=f'sub-{subject_id} - {condition}', colorbar=True, show=False)
        fig2 = tfr.plot_topo(title=f'sub-{subject_id}', tmin=0, show=False)
        fig1.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'time_freq', 'tfr-plots', 'joint', f'sub-{subject_id}-{condition}-joint-tfr-plt.png'))
        fig2.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'time_freq', 'tfr-plots', 'topo', f'sub-{subject_id}-{condition}-topo-plot.png'))
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
        fig1 = tfr.plot_joint(timefreqs=((0.1, 10),(0.2,10),(0.3,10)), tmin=0, tmax=0.6, fmin=8, fmax=30, vmin=-0.0000000005, vmax=0.0000000005,
                               topomap_args={'vlim': (-0.0000000015,0.0000000015)},
                               title=f'{population} - {condition}', colorbar=True, show=False)
        fig2 = tfr.plot_topo(title=f'{population} - {condition}', tmin=0, tmax=0.6, fmin=8, fmax=30, vmin=-0.0000000005, vmax=0.0000000005, show=False)
        fig3 = tfr.plot(picks=['P7', 'PO7', 'P9', 'O1'], combine='mean', title=f'{population} - {condition} - occip left', fmin=8, fmax=30, vmin=-0.0000000009, vmax=0.0000000009, show=False)[0]
        fig4 = tfr.plot(picks=['P8', 'PO8', 'P10', 'O2'], combine='mean', title=f'{population} - {condition} - occip right', fmin=8, fmax=30, vmin=-0.0000000009, vmax=0.0000000009, show=False)[0]
        fig1.savefig(os.path.join(output_dir, 'all_subj', 'N2pc', 'time_freq', 'tfr-plots', 'joint', population, f'{population}-{condition}-joint-tfr-plt.png'))
        fig2.savefig(os.path.join(output_dir, 'all_subj', 'N2pc', 'time_freq', 'tfr-plots', 'topo', population, f'{population}-{condition}-topo-plot.png'))
        fig3.savefig(os.path.join(output_dir, 'all_subj', 'N2pc', 'time_freq', 'tfr-plots', 'occip', population, f'{population}-{condition}-occip-left.png'))
        fig4.savefig(os.path.join(output_dir, 'all_subj', 'N2pc', 'time_freq', 'tfr-plots', 'occip', population, f'{population}-{condition}-occip-right.png'))
        plt.close('all')
        print(f'================= {condition} plots done for {population}')

    return None