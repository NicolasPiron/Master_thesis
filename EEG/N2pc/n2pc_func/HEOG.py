import mne
import os
import glob
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

    # Create a dataframe based on the index of the rejected epoch, and add information about the condition of the epoch
    df = pd.DataFrame(saccades, columns=['index'])

    # Get the condition of each epoch : the full name and the category (to be able to do stats with the side of the lesion of the participants)
    # The full condirion name
    conditions = []
    for sacc in saccades:
        epoch_ = epochs[sacc]
        condition = list(epoch_.event_id.keys())[0]
        conditions.append(condition)

    # Transform the dis_top and dis_bot conditions into dis_mid, but keep the side of the target
    conditions_clean = []
    for cond in conditions:
        if 'dis_top' in cond or 'dis_bot' in cond:
            if 'target_l' in cond:
                conditions_clean.append('dis_mid/target_l')
            elif 'target_r' in cond:
                conditions_clean.append('dis_mid/target_r')
        else:
            conditions_clean.append(cond)

    # The category of the condition
    conditions_grouped = []
    for cond in conditions_clean:
        if 'dis_mid' in cond:
            conditions_grouped.append('dis_mid')
        elif 'no_dis' in cond:
            conditions_grouped.append('no_dis')
        elif 'dis_right' in cond or 'dis_left' in cond:
            conditions_grouped.append('dis_lat')

    # add them to the dataframe
    df['condition'] = conditions_clean
    df['category'] = conditions_grouped
    
    print(f'======= rejection based on : {lower_threshold} - {upper_threshold} volts threshold ========')
    print(f'========== {len(saccades)} epochs rejected ==========')
    # create and save a csv file with the index of the rejected epochs
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', 'rejected-epochs-list')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', 'rejected-epochs-list'))
    #df = pd.DataFrame(saccades) 
    df.to_csv(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', 'rejected-epochs-list', f'sub-{subject_id}-heog-artifact.csv'))

def report_heog_all_subj(input_dir, output_dir):
    ''' Take all the csv files containing the index of the rejected epochs for each subject, and create a csv file with the number of rejected epochs for each subject.

    Parameters
    ----------
    input_dir : str
        Path to the input directory.
    output_dir : str
        Path to the output directory.
    
    Returns
    -------
    df : pandas dataframe
        Dataframe containing the number of rejected epochs for each subject.    
    '''

    # define subject paths list
    subject_paths = glob.glob(os.path.join(input_dir, 'sub-*'))
    print(f'========== {len(subject_paths)} subjects found ==========')

    # re order the list
    subject_paths = sorted(subject_paths)

    # empty dict to store the number of rejected epochs for each subject
    data_dict = {}

    # loop through the subject paths
    for subject_path in subject_paths:
        subject_id = subject_path[-2:]

        try:
            # load the csv file containing the index of the rejected epochs
            df = pd.read_csv(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', 'rejected-epochs-list', f'sub-{subject_id}-heog-artifact.csv'))

            # count the number of rejected epochs for each condition
            try:
                no_dis_target_r = df['condition'].value_counts()['no_dis/target_r']
            except:
                no_dis_target_r = 0
            try:
                no_dis_target_l = df['condition'].value_counts()['no_dis/target_l']
            except:
                no_dis_target_l = 0
            try:
                dis_mid_target_r = df['condition'].value_counts()['dis_mid/target_r']
            except:
                dis_mid_target_r = 0
            try:
                dis_mid_target_l = df['condition'].value_counts()['dis_mid/target_l']
            except:
                dis_mid_target_l = 0
            try:
                dis_right_target_l = df['condition'].value_counts()['dis_right/target_l']
            except:
                dis_right_target_l = 0
            try:
                dis_left_target_r = df['condition'].value_counts()['dis_left/target_r']
            except:
                dis_left_target_r = 0

            no_dis = no_dis_target_r + no_dis_target_l
            dis_mid = dis_mid_target_r + dis_mid_target_l
            dis_lat = dis_right_target_l + dis_left_target_r
            
            total = len(df)

            # store the number of rejected epochs for each subject
            rejected_epochs = {}
            rejected_epochs['total'] = total
            rejected_epochs['no_dis/target_r'] = no_dis_target_r
            rejected_epochs['no_dis/target_l'] = no_dis_target_l
            rejected_epochs['dis_mid/target_r'] = dis_mid_target_r
            rejected_epochs['dis_mid/target_l'] = dis_mid_target_l
            rejected_epochs['dis_right/target_l'] = dis_right_target_l
            rejected_epochs['dis_left/target_r'] = dis_left_target_r
            rejected_epochs['no_dis'] = no_dis
            rejected_epochs['dis_mid'] = dis_mid
            rejected_epochs['dis_lat'] = dis_lat
            
            # store the number of rejected epochs for each condition in the dict
            data_dict[f'sub-{subject_id}'] = rejected_epochs
            print(f'========== {subject_id} done ==========')
        except:
            print(f'no csv file for subject {subject_id}')
            continue

    # Flatten the nested dictionaries into a list of dictionaries
    data_list = [{'subject': key, **values} for key, values in data_dict.items()]
    # create a dataframe with the number of rejected epochs for each subject
    df = pd.DataFrame(data_list)
    df.set_index('subject', inplace=True)

    # save the dataframe as a csv file
    if not os.path.exists(os.path.join(output_dir, 'all_subj', 'N2pc', 'heog-artifact-report')):
        os.makedirs(os.path.join(output_dir, 'all_subj', 'N2pc', 'heog-artifact-report'))
    df.to_csv(os.path.join(output_dir, 'all_subj', 'N2pc', 'heog-artifact-report', 'heog-artifact-report-allsubj.csv'))

    return df

def reject_heog_artifacts(subject_id, input_dir, output_dir):
    ''' Reject the epochs containing HEOG artifacts.

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
    epochs : mne epochs
        Epochs with HEOG artifacts rejected.
    '''

    subject_id = str(subject_id)
    print(f'========== rejecting epochs for subject {subject_id} ==========')

    # load epochs
    epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-N2pc.fif'))

    # load the csv file containing the index of the rejected epochs
    df = pd.read_csv(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', 'rejected-epochs-list', f'sub-{subject_id}-heog-artifact.csv'))

    # get the index of the rejected epochs
    rejected_epochs = df['index'].to_list()

    # reject the epochs
    epochs.drop(rejected_epochs)

    # save the epochs
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs', 'heog-artifact-rejected')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs', 'heog-artifact-rejected'))
    epochs.save(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs', 'heog-artifact-rejected', f'sub-{subject_id}-cleaned_epochs-N2pc.fif'), overwrite=True)

    print(f'========== epochs rejected for subject {subject_id} ==========')

    return epochs

################################################################################
################# FUNCTIONS FOR CLEANED EPOCHS N2PC PLOTS ######################
################################################################################

# These functions are adapted from ERP.py, that's why they are wierd and not very clean

def to_evoked(subject_id, input_dir):
    ''' This function converts the epochs to evoked objects and saves them in the subject directory.
        It saves one evoked file by condition (i.e. bin).

    Parameters
    ----------
    subject_id : str
        The subject ID to plot.
    task : str
        The task to plot.
    input_dir : str
        The path to the directory containing the input data.
    
    Returns
    -------
    None
    
    '''
    # load the epochs
    file = os.path.join(input_dir, f'sub-{subject_id}/N2pc/cleaned_epochs/heog-artifact-rejected/sub-{subject_id}-cleaned_epochs-N2pc.fif')
    epochs = mne.read_epochs(file)


    # crop the epochs to the relevant time window
    tmin = -0.2
    tmax = 0.8
    epochs.crop(tmin=tmin, tmax=tmax)
        
    # define the bins
    bins = {'bin1' : ['dis_top/target_l','dis_bot/target_l'],
            'bin2' : ['dis_top/target_r','dis_bot/target_r'],
            'bin3' : ['no_dis/target_l'],
            'bin4' : ['no_dis/target_r'],
            'bin5' : ['dis_right/target_l'],
            'bin6' : ['dis_left/target_r']}

    # create evoked for each bin
    evoked_list = [epochs[bin].average() for bin in bins.values()]

    # create evoked for all the conditions
    evoked_all = epochs.average()
    
    # rename the distractor mid conditions to simplify
    evoked_1 = evoked_list[0]
    evoked_2 = evoked_list[1]
    evoked_1.comment = 'dis_mid/target_l'
    evoked_2.comment = 'dis_mid/target_r'
    
    # replace the '/' that causes problems when saving
    for evoked in evoked_list:
        evoked.comment = evoked.comment.replace('/', '_')
    
    # save the evoked objects in subject directory
    if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', f'evoked-N2pc-clean')):
        os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', f'evoked-N2pc-clean'))
    for evoked in evoked_list:
        print(evoked.comment)
        evoked.save(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', f'evoked-N2pc-clean', f'sub-{subject_id}-{evoked.comment}-ave.fif'), overwrite=True)
    evoked_all.save(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', f'evoked-N2pc-clean', f'sub-{subject_id}-all-ave.fif'), overwrite=True)

def get_evoked(subject_id, input_dir):
    ''' This function loads the evoked files for a given subject and returns a dictionary
    
    Parameters
    ----------
    subject_id : str
        The subject ID to plot.
    input_dir : str
        The path to the directory containing the input data.
    
    Returns
    -------
    bin_evoked : dict
        A dictionary containing the evoked objects for each condition (i.e. bin).
    '''

    subject_id = str(subject_id)
    evoked_path = os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', f'evoked-N2pc-clean')
    evoked_files = glob.glob(os.path.join(evoked_path, f'sub-{subject_id}-*.fif'))
    # Load the evoked files
    evoked_list = [mne.read_evokeds(evoked_file)[0] for evoked_file in evoked_files]
    bin_dict = {'bin1' : 'dis_mid_target_l',
            'bin2' : 'dis_mid_target_r',
            'bin3' : 'no_dis_target_l',
            'bin4' : 'no_dis_target_r',
            'bin5' : 'dis_right_target_l',
            'bin6' : 'dis_left_target_r'}

    # Assign the evoked object that corresponds to the bin
    bin_evoked = {}

    for bin_name, comment in bin_dict.items():
        for evoked in evoked_list:
            if evoked.comment == comment:
                bin_evoked[bin_name] = evoked
                break 

    # Rename the keys of the dict
    prefix = 'evk_'
    # Create a new dictionary with modified keys
    bin_evoked = {prefix + key: value for key, value in bin_evoked.items()}

    return bin_evoked

def get_bins_data(subject_id, input_dir):
    ''' This function extracts the N2pc ERP values for a given subject

    Parameters
    ----------
    subject_id : str
        The subject ID OR 'GA' to get the grand average.
    input_dir : str
        The path to the directory containing the input data.

    Returns
    -------
    PO7_data_nbin1 : numpy.ndarray
        The N2pc ERP data for the Dis_Mid contra condition.
    PO7_data_nbin2 : numpy.ndarray
        The N2pc ERP data for the Dis_Mid ipsi condition.
    PO7_data_nbin3 : numpy.ndarray
        The N2pc ERP data for the No_Dis contra condition.
    PO7_data_nbin4: numpy.ndarray
        The N2pc ERP data for the No_Dis ipsi condition.
    PO7_data_nbin5 : numpy.ndarray
        The N2pc ERP data for the Dis_Contra contra condition.
    PO7_data_nbin6 : numpy.ndarray
        The N2pc ERP data for the Dis_Contra ipsi condition.
    time : numpy.ndarray
        The time axis for the ERP data.
    '''
    def get_evoked_data(subject_id, input_dir):

        bin_evoked = get_evoked(subject_id, input_dir)
        
        # Define the channel indices for left (Lch) and right (Rch) channels
        Lch = np.concatenate([np.arange(0, 27)])
        Rch = np.concatenate([np.arange(33, 36), np.arange(38, 46), np.arange(48, 64)])

        # Lateralize bins in order to be able to compute the N2pc
        evk_bin1_R = bin_evoked['evk_bin1'].copy().pick(Rch)
        evk_bin1_L = bin_evoked['evk_bin1'].copy().pick(Lch)
        evk_bin2_R = bin_evoked['evk_bin2'].copy().pick(Rch)
        evk_bin2_L = bin_evoked['evk_bin2'].copy().pick(Lch)
        evk_bin3_R = bin_evoked['evk_bin3'].copy().pick(Rch)
        evk_bin3_L = bin_evoked['evk_bin3'].copy().pick(Lch)
        evk_bin4_R = bin_evoked['evk_bin4'].copy().pick(Rch)
        evk_bin4_L = bin_evoked['evk_bin4'].copy().pick(Lch)
        evk_bin5_R = bin_evoked['evk_bin5'].copy().pick(Rch)
        evk_bin5_L = bin_evoked['evk_bin5'].copy().pick(Lch)
        evk_bin6_R = bin_evoked['evk_bin6'].copy().pick(Rch)
        evk_bin6_L = bin_evoked['evk_bin6'].copy().pick(Lch)

        # Define functions to create the new bin operations
        def bin_operator(data1, data2):
            return 0.5 * data1 + 0.5 * data2
        
        # Create the new bins
        nbin1 = bin_operator(evk_bin1_R.data, evk_bin2_L.data)
        nbin2 = bin_operator(evk_bin1_L.data, evk_bin2_R.data)
        nbin3 = bin_operator(evk_bin3_R.data, evk_bin4_L.data)
        nbin4 = bin_operator(evk_bin3_L.data, evk_bin4_R.data)
        nbin5 = bin_operator(evk_bin5_R.data, evk_bin6_L.data)
        nbin6 = bin_operator(evk_bin5_L.data, evk_bin6_R.data)
        
        # Useful to plot the data
        time = bin_evoked['evk_bin1'].times * 1000  # Convert to milliseconds

        # Define the channel indices for (P7, P9, and) PO7
        #P7_idx = bin_evoked['evk_bin1'].info['ch_names'].index('P7')
        #P9_idx = bin_evoked['evk_bin1'].info['ch_names'].index('P9')
        PO7_idx = bin_evoked['evk_bin1'].info['ch_names'].index('PO7')

        # Extract the data for (P7, P9, and) PO7 electrodes
        PO7_data_nbin1 = nbin1[PO7_idx]
        PO7_data_nbin2 = nbin2[PO7_idx]
        PO7_data_nbin3 = nbin3[PO7_idx]
        PO7_data_nbin4 = nbin4[PO7_idx]
        PO7_data_nbin5 = nbin5[PO7_idx]
        PO7_data_nbin6 = nbin6[PO7_idx]
        
        return PO7_data_nbin1, PO7_data_nbin2, PO7_data_nbin3, PO7_data_nbin4, PO7_data_nbin5, PO7_data_nbin6, time

    b1, b2, b3, b4, b5, b6, time = get_evoked_data(subject_id, input_dir)
    
    return b1, b2, b3, b4, b5, b6, time

def plot_n2pc_clean(subject_id, input_dir, output_dir):

    def create_erp_plot(subject_id, contra, ipsi, time, color, condition, output_dir):

        plt.figure(figsize=(10, 6))
        plt.plot(time, contra, color=color, label=f'{condition} (Contralateral)')
        plt.plot(time, ipsi, color=color, linestyle='dashed', label=f'{condition} (Ipsilateral)')
        plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linewidth=1)
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude (uV)')
        plt.title(f'Signal from Electrodes PO7 - {condition} Condition')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, f'sub-{subject_id}','N2pc','heog-artifact', 'n2pc-plots-clean', f'sub-{subject_id}-PO7_{condition}_clean.png'))
        plt.show(block=False)
        plt.close()

    d1, d2, d3, d4, d5, d6, time = get_bins_data(subject_id, input_dir)
        
    # Create output directory if it doesn't exist
    if os.path.exists(os.path.join(output_dir, f'sub-{subject_id}','N2pc','heog-artifact', 'n2pc-plots-clean')) == False:
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}','N2pc','heog-artifact', 'n2pc-plots-clean'))

    condition1 = 'Dis_Mid'
    condition2 = 'No_Dis'
    condition3 = 'Dis_Contra'
    create_erp_plot(subject_id, d1, d2, time, 'blue', condition1, output_dir)
    create_erp_plot(subject_id, d3, d4, time,'green', condition2, output_dir)
    create_erp_plot(subject_id, d5, d6, time, 'red', condition3, output_dir)

    pass