import mne
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import math


################################################################################
######################### FUNCTIONS FOR ERP (AVERAGED SIGNAL) ##################
################################################################################

def get_heog_evoked(subject_id, input_dir, output_dir):
    ''' Only for N2pc task, compute the difference between ipsi and contra HEOG, saves it as a np array and returns it.

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
    dis_mid_diff : np array
        Difference between ipsi and contra HEOG for dis_mid condition.
    dis_side_diff : np array
        Difference between ipsi and contra HEOG for dis_side condition.
    no_dis_diff : np array
        Difference between ipsi and contra HEOG for no_dis condition.
    time : np array
        Time vector.
    '''

    subject_id = str(subject_id)
    # load epochs
    epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-N2pc.fif'))
    print('========== Epochs loaded ==========')

    # get time vector (important for plotting)
    time = epochs.times * 1000

    dis_mid_target_l = mne.concatenate_epochs([epochs['dis_top/target_l'], epochs['dis_bot/target_l']])
    dis_mid_target_r = mne.concatenate_epochs([epochs['dis_top/target_r'], epochs['dis_bot/target_r']])
    no_dis_target_l = epochs['no_dis/target_l']
    no_dis_target_r = epochs['no_dis/target_r']
    dis_side_target_l = epochs['dis_right/target_l']
    dis_side_target_r = epochs['dis_left/target_r']

    print('========== Epochs concatenated ==========')

    # isolate HEOG data
    # EXG3: right HEOG
    # EXG4: left HEOG
    dis_mid_target_l_heog3 = dis_mid_target_l.copy().pick('EXG3')
    dis_mid_target_r_heog3 = dis_mid_target_r.copy().pick('EXG3')
    no_dis_target_l_heog3 = no_dis_target_l.copy().pick('EXG3')
    no_dis_target_r_heog3 = no_dis_target_r.copy().pick('EXG3')
    dis_side_target_l_heog3 = dis_side_target_l.copy().pick('EXG3')
    dis_side_target_r_heog3 = dis_side_target_r.copy().pick('EXG3')

    dis_mid_target_l_heog4 = dis_mid_target_l.copy().pick('EXG4')
    dis_mid_target_r_heog4 = dis_mid_target_r.copy().pick('EXG4')
    no_dis_target_l_heog4 = no_dis_target_l.copy().pick('EXG4')
    no_dis_target_r_heog4 = no_dis_target_r.copy().pick('EXG4')
    dis_side_target_l_heog4 = dis_side_target_l.copy().pick('EXG4')
    dis_side_target_r_heog4 = dis_side_target_r.copy().pick('EXG4')

    print('========== HEOG data isolated ==========')

    # extract data
    dis_mid_target_l_heog3 = dis_mid_target_l_heog3.get_data()
    dis_mid_target_r_heog3 = dis_mid_target_r_heog3.get_data()
    no_dis_target_l_heog3 = no_dis_target_l_heog3.get_data()
    no_dis_target_r_heog3 = no_dis_target_r_heog3.get_data()
    dis_side_target_l_heog3 = dis_side_target_l_heog3.get_data()
    dis_side_target_r_heog3 = dis_side_target_r_heog3.get_data()

    dis_mid_target_l_heog4 = dis_mid_target_l_heog4.get_data()
    dis_mid_target_r_heog4 = dis_mid_target_r_heog4.get_data()
    no_dis_target_l_heog4 = no_dis_target_l_heog4.get_data()
    no_dis_target_r_heog4 = no_dis_target_r_heog4.get_data()
    dis_side_target_l_heog4 = dis_side_target_l_heog4.get_data()
    dis_side_target_r_heog4 = dis_side_target_r_heog4.get_data()
    
    print('========== HEOG data extracted ==========')

    # compute evoked
    dis_mid_target_l_heog3_evk = dis_mid_target_l_heog3.mean(axis=0)
    dis_mid_target_r_heog3_evk = dis_mid_target_r_heog3.mean(axis=0)
    no_dis_target_l_heog3_evk = no_dis_target_l_heog3.mean(axis=0)
    no_dis_target_r_heog3_evk = no_dis_target_r_heog3.mean(axis=0)
    dis_side_target_l_heog3_evk = dis_side_target_l_heog3.mean(axis=0)
    dis_side_target_r_heog3_evk = dis_side_target_r_heog3.mean(axis=0)

    dis_mid_target_l_heog4_evk = dis_mid_target_l_heog4.mean(axis=0)
    dis_mid_target_r_heog4_evk = dis_mid_target_r_heog4.mean(axis=0)
    no_dis_target_l_heog4_evk = no_dis_target_l_heog4.mean(axis=0)
    no_dis_target_r_heog4_evk = no_dis_target_r_heog4.mean(axis=0)
    dis_side_target_l_heog4_evk = dis_side_target_l_heog4.mean(axis=0)
    dis_side_target_r_heog4_evk = dis_side_target_r_heog4.mean(axis=0)

    # define ipsi and contra depending on the side of the target
    # dis mid
    dis_mid_target_l_ipsi = dis_mid_target_l_heog4_evk
    dis_mid_target_l_contra = dis_mid_target_l_heog3_evk
    dis_mid_target_r_ipsi = dis_mid_target_r_heog3_evk
    dis_mid_target_r_contra = dis_mid_target_r_heog4_evk

    # dis side 
    dis_side_target_l_ipsi = dis_side_target_l_heog4_evk
    dis_side_target_l_contra = dis_side_target_l_heog3_evk
    dis_side_target_r_ipsi = dis_side_target_r_heog3_evk
    dis_side_target_r_contra = dis_side_target_r_heog4_evk

    # no dis
    no_dis_target_l_ipsi = no_dis_target_l_heog4_evk
    no_dis_target_l_contra = no_dis_target_l_heog3_evk
    no_dis_target_r_ipsi = no_dis_target_r_heog3_evk
    no_dis_target_r_contra = no_dis_target_r_heog4_evk

    # compute the difference between ipsi and contra
    # dis mid
    dis_mid_target_l_diff = dis_mid_target_l_ipsi - dis_mid_target_l_contra
    dis_mid_target_r_diff = dis_mid_target_r_ipsi - dis_mid_target_r_contra

    # dis side
    dis_side_target_l_diff = dis_side_target_l_ipsi - dis_side_target_l_contra
    dis_side_target_r_diff = dis_side_target_r_ipsi - dis_side_target_r_contra

    # no dis
    no_dis_target_l_diff = no_dis_target_l_ipsi - no_dis_target_l_contra
    no_dis_target_r_diff = no_dis_target_r_ipsi - no_dis_target_r_contra

    print('========== sanity check ==========')

    # apply a filter to smooth the data
    # reshape arrays
    dis_mid_target_l_diff = dis_mid_target_l_diff.reshape(dis_mid_target_l_diff.shape[1])
    dis_mid_target_r_diff = dis_mid_target_r_diff.reshape(dis_mid_target_r_diff.shape[1])
    dis_side_target_l_diff = dis_side_target_l_diff.reshape(dis_side_target_l_diff.shape[1])
    dis_side_target_r_diff = dis_side_target_r_diff.reshape(dis_side_target_r_diff.shape[1])
    no_dis_target_l_diff = no_dis_target_l_diff.reshape(no_dis_target_l_diff.shape[1])
    no_dis_target_r_diff = no_dis_target_r_diff.reshape(no_dis_target_r_diff.shape[1])

    print('========== HEOG data reshaped ==========')

    # create butterworth filter
    sos = signal.butter(1, 5, 'lp', fs=512, output='sos')

    print('========== filter created ==========')

    # apply filter
    filtered_dis_mid_target_l_diff = signal.sosfilt(sos, dis_mid_target_l_diff)
    filtered_dis_mid_target_r_diff = signal.sosfilt(sos, dis_mid_target_r_diff)
    filtered_dis_side_target_l_diff = signal.sosfilt(sos, dis_side_target_l_diff)
    filtered_dis_side_target_r_diff = signal.sosfilt(sos, dis_side_target_r_diff)
    filtered_no_dis_target_l_diff = signal.sosfilt(sos, no_dis_target_l_diff)
    filtered_no_dis_target_r_diff = signal.sosfilt(sos, no_dis_target_r_diff)

    print('========== filter applied ==========')

    # compute the mean of the 2 differences
    # dis mid
    dis_mid_diff = (filtered_dis_mid_target_l_diff + filtered_dis_mid_target_r_diff) / 2
    #dis_mid_diff = dis_mid_diff.reshape(dis_mid_diff.shape[0], 1)

    # dis side
    dis_side_diff = (filtered_dis_side_target_l_diff + filtered_dis_side_target_r_diff) / 2
    #dis_side_diff = dis_side_diff.reshape(dis_side_diff.shape[0], 1)

    # no dis
    no_dis_diff = (filtered_no_dis_target_l_diff + filtered_no_dis_target_r_diff) / 2
    #no_dis_diff = no_dis_diff.reshape(no_dis_diff.shape[0], 1)

    # save evoked (np arrays)
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', 'evoked-heog',)):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', 'evoked-heog',))
    np.save(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', 'evoked-heog', f'sub-{subject_id}-dis_mid_diff.npy'), dis_mid_diff)
    np.save(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', 'evoked-heog', f'sub-{subject_id}-dis_side_diff.npy'), dis_side_diff)
    np.save(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', 'evoked-heog', f'sub-{subject_id}-no_dis_diff.npy'), no_dis_diff)
    print('========== Evoked saved ==========')


    return dis_mid_diff, dis_side_diff, no_dis_diff, time

def plot_heog_erp(subject_id, input_dir, output_dir):
    ''' Plots the HEOG ERP for all (3) conditions.

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
    None
    '''

    subject_id = str(subject_id)

    # get HEOG averaged data (across trials, groupes by condition)
    dis_mid_diff, dis_side_diff, no_dis_diff, time = get_heog_evoked(subject_id, input_dir, output_dir)
    print('========== HEOG evoked data loaded ==========')

    # plot the HEOG ERP
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, dis_mid_diff, label='dis_mid')
    ax.plot(time, dis_side_diff, label='dis_side')
    ax.plot(time, no_dis_diff, label='no_dis')
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (V)')
    ax.set_title('Difference between ipsi and contra HEOG - all conditions')
    ax.legend()
    ax.grid()

    # save figure
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', 'heog-waveform', 'heog-evoked')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', 'heog-waveform', 'heog-evoked'))
    fig.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', 'heog-waveform', 'heog-evoked', f'sub-{subject_id}-heog-erp.png'))
    
    print('========== HEOG ERP plot saved ==========')

    return None

################################################################################
############# FUNCTIONS FOR EPOCHS REJECTION BASED ON HEOG AMP #################
################################################################################

def rejection_report_heog_artifact(subject_id, input_dir, output_dir):

    subject_id = str(subject_id)

    # load epochs
    epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-N2pc.fif'))

    # get time vector (important for plotting)
    time = epochs.times*1000

    # empty list to store the index of the rejected epochs
    saccades = []

    # loop through the epochs
    for i, epoch in enumerate(epochs.pick(['EXG3', 'EXG4'])):
        
        # get the data of VEOG1, VEOG2, and the diff
        veog1 = epoch[0]
        veog2 = epoch[1]
        diff = veog1-veog2
        
        # create and apply a filter to smooth the signal (EXGs are not filtered during preprocessing)
        sos = signal.butter(1, 5, 'lp', fs=512, output='sos')
        veog1 = signal.sosfilt(sos,veog1)
        veog2 = signal.sosfilt(sos,veog2)
        diff = signal.sosfilt(sos,diff)

        # create directory to save the plots
        if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', 'heog-waveform', 'individual-epochs')):
            os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', 'heog-waveform', 'individual-epochs'))

        # plot and save the 3 waveformes
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(time, veog1, c='#FAB6C6', linestyle='--', label='VEOG1')
        ax.plot(time, veog2, c='#B6F0FA', linestyle='--', label='VEOG2')
        ax.plot(time, diff, c='#FA2937', label='diff')
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
        ax.axhline(y=0, color='black', linewidth=1)
        ax.set_ylim(-0.0001, 0.0001)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (V)')
        ax.set_title(f'subject {subject_id} : epoch {i}')
        ax.legend()
        ax.grid()
        fig.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', 'heog-waveform', 'individual-epochs', f'sub-{subject_id}-heog-epoch-{i}.png'))
        plt.close(fig)

        # create a rejection threshold # here : 40 microV between 0 and 300 ms
        # tmin : 0 ms -> 0.2*epochs.info['sfreq'] because 200 ms before onset
        tmin = 0.2*epochs.info['sfreq']
        tmin = math.ceil(tmin)
        # tmax : 300ms -> 0.5*epochs.info['sfreq'] because 200 ms before onset + 300 ms after onset
        tmax = 0.5*epochs.info['sfreq']
        tmax = math.ceil(tmax)

        lower_threshold = -0.00004
        upper_threshold = 0.00004

        if diff[tmin:tmax].max() > upper_threshold or diff[tmin:tmax].min() < lower_threshold:
            print(f'saccade in epoch {i}')
            saccades.append(i)

    print(f'======= rejection based on : {lower_threshold} - {upper_threshold} volts threshold ========')
    print(f'========== {len(saccades)} epochs rejected ==========')
    # create and save a csv file with the index of the rejected epochs
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', 'rejected-epochs-list')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', 'rejected-epochs-list'))
    df = pd.DataFrame(saccades) 
    df.to_csv(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', 'rejected-epochs-list', f'sub-{subject_id}-heog-artifact.csv'), index=False, header=False)

