import mne
import numpy as np
import pandas as pd
import glob
import mne
import os
import re

### ================================================================================================
### ==================================== N2PC ALPHA POWER PER EPOCH ================================
### ============================================== FFT =============================================
### ================================================================================================

# 3rd version of the functions. This time we compute the alpha power for each individual electrode using FFT.
# For that we use the mne.time_frequency.Spectrum class. 

def get_psd(subject_id, input_dir):

    # load the epochs
    epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs',f'sub-{subject_id}-cleaned_epochs-N2pc.fif'))
    # clear bad channels
    epochs.info['bads'] = []
    # create spectrum object
    spec = epochs.compute_psd(fmin=0, fmax=35)
    print(f'========================= PSD for subject {subject_id} computed!')

    return spec

def get_mean_freq(spec, bands, picks, epoch_index):

    # Get the equivalent indices of the bands in the data (from .get_data() method)
    lower_bound = bands[0]*2
    upper_bound = bands[1]*2+1

    # Compute the mean of the epochs for each freq
    spec_mean_power = spec.get_data(picks=picks)[epoch_index:epoch_index+1, :, lower_bound:upper_bound].mean(axis=2)
    print(f'========================= Mean power for {picks} computed!')

    return spec_mean_power

def get_power_df_single_subj(subject_id, input_dir, output_dir):


    # load the epochs
    epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs',f'sub-{subject_id}-cleaned_epochs-N2pc.fif'))

    # Get the spec object
    spec = get_psd(subject_id, input_dir)

    # Define the frequency bands
    bands = {'theta' : np.arange(4, 9),
         'alpha' : np.arange(8, 13),
         'low_beta' : np.arange(12, 17),
         'high_beta' : np.arange(16, 31)
    
    }

    # Define the picks
    channels = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7',
      'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4','F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4',
        'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']

    # Define the columns of the dataframe : subject ID, epoch index, epoch dropped, reset index, condition, target side,
    # distractor position, theta Fp1, alpha Fp1, low beta Fp1, high beta Fp1, theta AF7, alpha AF7, low beta AF7, high beta AF7, etc...
    ch_list = []
    for ch in channels:
        for band in ['theta', 'alpha', 'low_beta', 'high_beta']:
            ch_list.append(f'{band}-{ch}')
    columns = ['ID', 'epoch_index', 'epoch_dropped', 'index_reset','saccade','condition', 'target_side', 'distractor_position'] + ch_list

    # Create the dataframe
    df = pd.DataFrame(columns=columns)

    # Populate the df
    # Start by getting the epochs indices, the epochs status (dropped or not) and the reset indices

    # Load the reject log
    reject_log = np.load(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'preprocessing', 'step-06-final-reject_log', f'sub-{subject_id}-final-reject_log-N2pc.npz'))
    # Define the epochs status (rejected or not)
    epochs_status = reject_log['bad_epochs']
    
    # Create row for each epoch
    df['ID'] = [subject_id] * len(epochs_status)
    df['epoch_index'] = range(1,len(epochs_status)+1)
    df['epoch_dropped'] = epochs_status
    
    # Add a column that store the reset index of the epochs
    index_val = 0
    index_list = []

    # Iterate through the 'epoch_dropped' column to create the reset index column
    for row_number in range(len(df)):
        if df.iloc[row_number, 2] == False:
            index_list.append(index_val)
            index_val += 1
        else:
            index_list.append(np.nan)

    # Add the reset index column to the DataFrame
    df['index_reset'] = index_list

    # Load the csv file contaning the indices of epochs with saccades
    saccades = pd.read_csv(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', 'rejected-epochs-list', f'sub-{subject_id}-heog-artifact.csv'))
    # Create a list of the indices of the epochs with saccades
    saccades_list = list(saccades['index'])
    # Add a column that specifies if the epoch contains a saccade. FALSE if no saccade, TRUE if saccade. 
    df['saccade'] = df['index_reset'].isin(saccades_list)

    # Fill the condition and the power columns
    for row_number in range(len(df)):

        if df.iloc[row_number, 2] == False:

            # add condition
            df.iloc[row_number, 5] = epochs.events[int(df['index_reset'].loc[row_number]),2]

            # add target side
            if df.iloc[row_number, 5] % 2 == 0:
                df.iloc[row_number, 6] = 'right'
            elif df.iloc[row_number, 5] % 2 != 0:
                df.iloc[row_number,6] = 'left'

            # add dis position
            if df.iloc[row_number, 5] in [1,2,5,6]:
                df.iloc[row_number, 7] = 'mid'
            elif df.iloc[row_number, 5] in [3,4]:
                df.iloc[row_number, 7] = 'nodis'
            elif df.iloc[row_number, 5] in [7,8]:
                df.iloc[row_number, 7] = 'lat'
            
            # Compute the mean power for each band and each electrode
            for i, ch in enumerate(ch_list):
                band, pick = ch.split('-')
                power = get_mean_freq(spec, bands=bands[band], picks=pick, epoch_index=int(df['index_reset'].loc[row_number]))
                df.iloc[row_number, i+8] = power[0][0] # +8 because the first 8 columns are not electrodes

    print(f'========================= Dataframe created for subject {subject_id}!')
    # save the dataframe
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'alpha-power-df')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}','N2pc', 'alpha-power-df'))
    df.to_csv(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'alpha-power-df', f'sub-{subject_id}-alpha-power-per-epoch_v2.csv'), index=False)
    print(f'========================= Dataframe saved for subject {subject_id}!')

    return df

def get_power_df_all_subj(input_dir, output_dir):

    # Get the list of all the subject directories
    subject_list = [os.path.basename(subj) for subj in glob.glob(os.path.join(input_dir, 'sub-*'))]
    # Sort the list
    subject_list.sort()

    # Create a list to store the dataframes
    df_list = []

    # Loop over the subject directories
    for subject in subject_list:

        subject_id = subject[-2:]
        # Compute alpha power and save it in a dataframe
        try:
            df = get_power_df_single_subj(subject_id, input_dir, output_dir)
            df_list.append(df)
            print(f'==================== Dataframe created and saved for subject {subject_id}! :)')
        except:
            print(f"==================== No data (epochs or reject log) for subject {subject_id}! O_o'")
            continue
    
    # Concatenate all dataframes in the list
    big_df = pd.concat(df_list)

    # Save dataframe as .csv file
    if not os.path.exists(os.path.join(output_dir,'all_subj','N2pc', 'alpha-power-allsubj')):
        os.makedirs(os.path.join(output_dir,'all_subj','N2pc', 'alpha-power-allsubj'))
    big_df.to_csv(os.path.join(output_dir,'all_subj','N2pc', 'alpha-power-allsubj', 'alpha-power-per-epoch-allsubj_v2.csv'), index=False)

    return big_df

### ================================================================================================
### ==================================== N2PC ALPHA POWER PER EPOCH ================================
### ============================================== CWT =============================================
### ================================================================================================


def alpha_power_per_epoch(subject_id, input_dir, right_elecs=['O2', 'PO4', 'PO8'], left_elecs=['O1', 'PO3', 'PO7']):
    ''' Compute the alpha power for each epoch and return a list of alpha power values for the right side of the head and a list of alpha power values for the left side of the head.

    Parameters:
    ----------
    epochs : mne.Epochs
        The epochs object to be sorted.
    
    Returns
    ----------
    right_power_list : list of float
        a list of alpha band mean power values (8-12Hz) for the right side of the head for each epoch
    
    left_power_list : list of float
        a list of alpha band mean power values (8-12Hz) for the left side of the head for each epoch
    '''

    epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs',f'sub-{subject_id}-cleaned_epochs-N2pc.fif'))

    freqs = np.arange(8, 13)

    n_cycles = freqs / 2.
    time_bandwidth = 4.
    n_jobs = 1  # number of parallel jobs to run
    
    right_power_list = []
    left_power_list = []

    print('========================= Computing alpha power, might take a while...')

    for i in range(len(epochs)):

        right_power = mne.time_frequency.tfr_morlet(epochs[i], freqs=freqs, n_cycles=n_cycles, picks=right_elecs,
                                                        use_fft=True, return_itc=False, decim=1,
                                                        n_jobs=n_jobs, verbose=True)
        right_power_mean = right_power.to_data_frame()[right_elecs].mean(axis=1).mean(axis=0)
        right_power_list.append(right_power_mean)

        left_power= mne.time_frequency.tfr_morlet(epochs[i], freqs=freqs, n_cycles=n_cycles, picks=left_elecs,
                                                        use_fft=True, return_itc=False, decim=1,
                                                        n_jobs=n_jobs, verbose=True)
        left_power_mean = left_power.to_data_frame()[left_elecs].mean(axis=1).mean(axis=0)
        left_power_list.append(left_power_mean)

    return right_power_list, left_power_list

def alpha_df_epoch(subject_id : str, input_dir, right_power_list, left_power_list):
    ''' Create a dataframe with the alpha power for each epoch

    Parameters:
    ----------
    epochs : mne.Epochs
        The epochs object to be sorted.

    right_power_list : list of float
        a list of alpha band mean power values (8-12Hz) for the right side of the head for each epoch

    left_power_list : list of float
        a list of alpha band mean power values (8-12Hz) for the left side of the head for each epoch

    Returns
    ----------
    df : pandas.DataFrame
        A dataframe with the alpha power score for each side for each epoch

    '''
    # Load the epochs
    epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-N2pc.fif'))
    # Load the reject log
    reject_log = np.load(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'preprocessing', 'step-06-final-reject_log', f'sub-{subject_id}-final-reject_log-N2pc.npz'))
    # Define the epochs status (rejected or not)
    epochs_status = reject_log['bad_epochs']

    # Initiate the df
    df = pd.DataFrame(columns=[['epoch_index', 'epoch_dropped', 'condition','target_side', 'distractor_position', 'alpha_power_right', 'alpha_power_left']])
    
    # Create row for each epoch
    df['epoch_index'] = range(1,len(epochs_status)+1)
    df['epoch_dropped'] = epochs_status

    # Add aa column that store the reset index of the epochs
    index_val = 0
    index_list = []

    # Iterate through the 'epoch_dropped' column to create the reset index column
    for row_number in range(len(df)):
        if df.iloc[row_number, 1] == False:
            index_list.append(index_val)
            index_val += 1
        else:
            index_list.append(np.nan)
    # Add the index column to the DataFrame
    df['index_reset'] = index_list
    
    for row_number in range(len(df)):

        if df.iloc[row_number, 1] == False:

            # add condition
            df.iloc[row_number, 2] = epochs.events[int(df['index_reset'].loc[row_number]),2]

            # add target side
            if df.iloc[row_number, 2] % 2 == 0:
                df.iloc[row_number, 3] = 'right'
            elif df.iloc[row_number, 2] % 2 != 0:
                df.iloc[row_number,3] = 'left'

            # add dis position
            if df.iloc[row_number, 2] in [1,2,5,6]:
                df.iloc[row_number, 4] = 'mid'
            elif df.iloc[row_number, 2] in [3,4]:
                df.iloc[row_number, 4] = 'nodis'
            elif df.iloc[row_number, 2] in [7,8]:
                df.iloc[row_number, 4] = 'lat'
                
            # add alpha power right
            df.iloc[row_number, 5] = right_power_list[int(df['index_reset'].loc[row_number])]
            # add alpha power left
            df.iloc[row_number, 6] = left_power_list[int(df['index_reset'].loc[row_number])]
    
    # Scientific notification because very small values
    pd.options.display.float_format = '{:.5e}'.format

    return df    

def alpha_df_epoch_3clusters(subject_id, input_dir):

    # get the alpha power for each epoch for each cluster
    cluster_dict = {'occipital': {'right':['O2', 'PO4', 'PO8'], 'left': ['O1', 'PO3', 'PO7']},
                    'parietal' : {'right':['P2', 'CP2', 'CP4'], 'left':['P1', 'CP1', 'CP3']}, 'frontal':{'right':['FC2', 'FC4', 'F2'], 'left':['FC1', 'FC3', 'F1']}}


    right_occip, left_occip = alpha_power_per_epoch(subject_id, input_dir, right_elecs=cluster_dict['occipital']['right'], left_elecs=cluster_dict['occipital']['left'])
    right_parietal, left_parietal = alpha_power_per_epoch(subject_id, input_dir, right_elecs=cluster_dict['parietal']['right'], left_elecs=cluster_dict['parietal']['left'])
    right_frontal, left_frontal = alpha_power_per_epoch(subject_id, input_dir, right_elecs=cluster_dict['frontal']['right'], left_elecs=cluster_dict['frontal']['left'])

    epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-N2pc.fif'))
    if type(epochs) == mne.epochs.EpochsFIF:
        print(f'=========== epochs found for subject {subject_id}')
    else:
        print(f'=========== epochs not found for subject {subject_id}')
    # Load the reject log
    reject_log = np.load(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'preprocessing', 'step-06-final-reject_log', f'sub-{subject_id}-final-reject_log-N2pc.npz'))
    # Define the epochs status (rejected or not)
    epochs_status = reject_log['bad_epochs']

    # Initiate the df
    df = pd.DataFrame(columns=[['ID', 'epoch_index', 'epoch_dropped', 'condition','target_side', 'distractor_position', 'alpha_power_right_occip', 'alpha_power_left_occip',
                                'alpha_power_right_parietal', 'alpha_power_left_parietal', 'alpha_power_right_frontal', 'alpha_power_left_frontal']])
    
    # Create row for each epoch
    df['ID'] = [subject_id] * len(epochs_status)
    df['epoch_index'] = range(1,len(epochs_status)+1)
    df['epoch_dropped'] = epochs_status

    # Add aa column that store the reset index of the epochs
    index_val = 0
    index_list = []

    # Iterate through the 'epoch_dropped' column to create the reset index column
    for row_number in range(len(df)):
        if df.iloc[row_number, 2] == False:
            index_list.append(index_val)
            index_val += 1
        else:
            index_list.append(np.nan)
    # Add the index column to the DataFrame
    df['index_reset'] = index_list

    for row_number in range(len(df)):

        if df.iloc[row_number,2] == False:

            # add condition
            df.iloc[row_number, 3] = epochs.events[int(df['index_reset'].loc[row_number]),2]

            # add target side
            if df.iloc[row_number, 3] % 2 == 0:
                df.iloc[row_number, 4] = 'right'
            elif df.iloc[row_number, 3] % 2 != 0:
                df.iloc[row_number,4] = 'left'

            # add dis position
            if df.iloc[row_number, 3] in [1,2,5,6]:
                df.iloc[row_number, 5] = 'mid'
            elif df.iloc[row_number, 3] in [3,4]:
                df.iloc[row_number, 5] = 'nodis'
            elif df.iloc[row_number, 3] in [7,8]:
                df.iloc[row_number, 5] = 'lat'
                
            # add alpha power right occipital
            df.iloc[row_number, 6] = right_occip[int(df['index_reset'].loc[row_number])]
            # add alpha power left occipital
            df.iloc[row_number, 7] = left_occip[int(df['index_reset'].loc[row_number])]
            # add alpha power right parietal
            df.iloc[row_number, 8] = right_parietal[int(df['index_reset'].loc[row_number])]
            # add alpha power left parietal
            df.iloc[row_number, 9] = left_parietal[int(df['index_reset'].loc[row_number])]
            # add alpha power right frontal
            df.iloc[row_number, 10] = right_frontal[int(df['index_reset'].loc[row_number])]
            # add alpha power left frontal
            df.iloc[row_number, 11] = left_frontal[int(df['index_reset'].loc[row_number])]
    
    # Scientific notification because very small values
    pd.options.display.float_format = '{:.5e}'.format

    return df

def single_subj_alpha_epoch(subject_id, input_dir, output_dir):

    subject_id = str(subject_id)

    df = alpha_df_epoch_3clusters(subject_id, input_dir)

    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'alpha-power-df')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}','N2pc', 'alpha-power-df'))
    df.to_csv(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'alpha-power-df', f'sub-{subject_id}-alpha-power-per-epoch.csv'))

    return df
    
# ======================================== ALL SUBJECTS ==========================================

def all_subj_alpha_epoch(input_dir, output_dir):
    ''' Create a dataframe for all subjects with the alpha score for each side and for each epoch
    
    Parameters:
    ----------
    Input_dir : str
        The path to the directory containing the epochs objects for each subject
    output_dir : str
        The path to the directory where the dataframe will be saved

    Returns
    ----------
    big_df : pandas.DataFrame
        A dataframe with the alpha power score for each side for each subject
    '''
    print('========================= WARNING')
    print('THIS FUNCTION SHOULD NOT BE USED, IT DOES NOT TRACK THE DROPPED EPOCHS')
    print('USE THE FUNCTION alpha_power_per_epoch INSTEAD - SEE THE FILE compute_alpha_epoch.py')
    print('========================= WARNING')

    big_df = pd.DataFrame()

    # Empty list to store the files
    all_subj_files = []

    # Loop over the directories to access the files
    directories = glob.glob(os.path.join(input_dir, 'sub*', 'N2pc'))
    for directory in directories:
        file = glob.glob(os.path.join(directory, 'cleaned_epochs', 'sub*N2pc.fif'))
        all_subj_files.append(file[0])
    all_subj_files.sort()

    # Pattern to extract the subject ID
    pattern = r'sub-(\d{2})-cleaned'

    for subject in all_subj_files:

        # Extract the subject ID
        match = re.search(pattern, subject)
        if match:
            subj_id = match.group(1)
        else:
            print("No match found in:", subject)

        print(f'========================= working on {subj_id}')

        # Load the epochs
        right, left = alpha_power_per_epoch(subject_id=subj_id, input_dir=input_dir)
        subj_df = alpha_df_epoch(subject_id=subj_id, input_dir=input_dir, right_power_list=right, left_power_list=left)

        subj_df.insert(0, 'ID', subj_id)

        big_df = pd.concat([big_df, subj_df], ignore_index=True)
    
    if not os.path.exists(os.path.join(output_dir, 'all_subj' 'alpha-power-df')):
        os.makedirs(os.path.join(output_dir, 'all_subj', 'alpha-power-df'))
    big_df.to_csv(os.path.join(output_dir, 'all-subj', 'alpha-power-df', 'allsubj_alpha_power_per_epoch.csv'))

    return big_df


### ================================================================================================
### ======================================= LEGACY CODE ============================================
### ================================================================================================

### NOT USED ANYMORE : MIGHT HAVE PROBLEMS WITH THE NEW FILE ORGANIZATION!!

def sort_epochs(subject_id : str, input_dir : str):
    ''' Takes the epochs object and returns a list of epochs objects sorted by condition.
        
        the original order of the epochs is as follows: 
        'dis_top/target_l': 1,
        'dis_top/target_r': 2,
        'no_dis/target_l': 3,
        'no_dis/target_r': 4,
        'dis_bot/target_l': 5,
        'dis_bot/target_r': 6,
        'dis_right/target_l': 7,
        'dis_left/target_r': 8

    Parameters
    ----------
    subject_id : str
        The subject ID.
    input_dir : str
        The path to the directory containing all the subject directories

    Returns
    -------
    sorted_epochs : list of mne.Epochs
        A list of epochs sorted by condition. The order of the epochs is as follows:
        ['no_dis/target_l',
        'no_dis/target_r',
        'dis_right/target_l',
        'dis_left/target_r',
        'dis_mid/target_l',
        'dis_mid/target_r']
    '''

    epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', 'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-N2pc.fif'))

    event_ids = epochs.event_id
    
    sorted_epochs = []
    
    for event in event_ids.keys():
        
        condition = epochs[event]
        sorted_epochs.append(condition)

    # concatenate the the files when the target is on the same side but the distractor is bot or top
    
    dis_vert_target_l = mne.concatenate_epochs([sorted_epochs[0], sorted_epochs[4]])
    dis_vert_target_r = mne.concatenate_epochs([sorted_epochs[1], sorted_epochs[5]])
    
    sorted_epochs.append(dis_vert_target_l)
    sorted_epochs.append(dis_vert_target_r)
    
    # remove useless files
    
    indices_to_remove = [0, 1, 4, 5]
    
    # reverse to avoid index shifting issues
    
    indices_to_remove.sort(reverse=True)
    for index in indices_to_remove:
        del sorted_epochs[index]
    
    return sorted_epochs  



def compute_alpha_by_side(sorted_epochs : list):
    ''' Takes a list of epochs objects sorted by conditions and returns 2 lists of alpha band mean power values (8-12Hz), one for each side of the head.

    Parameters
    ----------
    sorted_epochs : list of mne.Epochs
        a list of epochs objects sorted by condition. The order of the epochs is as follows:
        'no_dis/target_l',
        'no_dis/target_r',
        'dis_right/target_l',
        'dis_left/target_r',
        'dis_mid/target_l',
        'dis_mid/target_r'

    Returns
    -------
    right_power_list : list of float
        a list of alpha band mean power values (8-12Hz) for the right side of the head for each epochs object in sorted_epochs (condition)

    left_power_list : list of float
        a list of alpha band mean power values (8-12Hz) for the left side of the head for each epochs object in sorted_epochs (condition)
    '''
    freqs = np.arange(8, 13)
    right_elecs=[ 'O2', 'PO4', 'PO8']
    left_elecs=['O1', 'PO3', 'PO7']

    n_cycles = freqs / 2.
    time_bandwidth = 4.
    baseline = None  # no baseline correction
    n_jobs = 1  # number of parallel jobs to run
    
    right_power_list = []
    left_power_list = []

    # compute alpha power and append it to right/left_power_list
    for i in range(len(sorted_epochs)):
        right_power = mne.time_frequency.tfr_morlet(sorted_epochs[i], freqs=freqs, n_cycles=n_cycles, picks=right_elecs,
                                                use_fft=True, return_itc=False, decim=1,
                                                n_jobs=n_jobs, verbose=True)
        right_power_mean = right_power.to_data_frame()[right_elecs].mean(axis=1).mean(axis=0)
        right_power_list.append(right_power_mean)

        left_power= mne.time_frequency.tfr_morlet(sorted_epochs[i], freqs=freqs, n_cycles=n_cycles, picks=left_elecs,
                                                use_fft=True, return_itc=False, decim=1,
                                                n_jobs=n_jobs, verbose=True)
        left_power_mean = left_power.to_data_frame()[left_elecs].mean(axis=1).mean(axis=0)
        left_power_list.append(left_power_mean)
    
    return right_power_list, left_power_list


def extract_conditions(epochs_list : list):
    ''' Takes a list of epochs objects and returns a list of strings containing the conditions.
    
    Parameters
    ----------
    epochs_list : list of mne.Epochs
        a list of epochs objects sorted by condition. 
    
    Returns
    -------
    conditions : list of str
        a list of strings containing the conditions.
    '''
    conditions = []
    string_conditions = []

    for i in epochs_list:

        cond = list(i.event_id.keys())
        conditions.append(cond)
        
    for index, cond in enumerate(conditions):
        
        if len(cond) == 2 and 'target_l' in conditions[index][0]:
            conditions[index] = ['dis_mid/target_l']
        elif len(cond) == 2 and 'target_r' in conditions[index][0]:
            conditions[index] = ['dis_mid/target_r']
            
    string_conditions = [cond[0] for cond in conditions]
    
    return string_conditions

def alpha_power_df(conditions : list, right_power_list: list, left_power_list : list):
    ''' Takes a list of conditions, a list of alpha power values for the right side of the head and a list of alpha power values for the left side of the head and 
        returns a dataframe containing the conditions, the target side, the distractor side, the alpha side relative to the target and the mean alpha power.

    Parameters
    ----------
    conditions : list of str
        a list of strings containing the conditions.

    right_power_list : list of float
        a list of alpha band mean power values (8-12Hz) for the right side of the head for each epochs object in sorted_epochs (condition)
    
    left_power_list : list of float
        a list of alpha band mean power values (8-12Hz) for the left side of the head for each epochs object in sorted_epochs (condition)

    Returns
    -------
    df : pandas dataframe
        a dataframe containing the conditions, the target side, the distractor side, the side of recording relative to the target (ipsi or contralateral) and the mean alpha power value.
    '''
    df = pd.DataFrame(columns=[['condition','target_side', 'distractor_side', 'alpha_side', 'alpha_power']])
    
    df['condition'] = conditions * 2
    
    df ['alpha_power'] = right_power_list + left_power_list
    
    for row_number in range(len(df)):
        
        # add target side
        if 'target_l' in df.iloc[row_number, 0]:
            df.iloc[row_number, 1] = 'left'
        elif 'target_r' in df.iloc[row_number, 0]:
            df.iloc[row_number, 1] = 'right'
        
        # add distractor side
        if 'dis_mid' in df.iloc[row_number, 0]:
            df.iloc[row_number, 2] = 'mid'
        elif 'dis_right' in df.iloc[row_number, 0] or 'dis_left' in df.iloc[row_number, 0]:
            df.iloc[row_number, 2] = 'lat'
        elif 'no_dis' in df.iloc[row_number, 0]:
            df.iloc[row_number, 2] = 'nodis'

        # add alpha side
        if row_number <= 5:
            if 'target_l' in df.iloc[row_number, 0]:
                df.iloc[row_number, 3] = 'contra'
            elif 'target_r' in df.iloc[row_number, 0]:
                df.iloc[row_number,3] = 'ipsi'
        else:
            if 'target_l' in df.iloc[row_number, 0]:
                df.iloc[row_number,3] = 'ipsi'
            elif 'target_r' in df.iloc[row_number, 0]:
                df.iloc[row_number,3] = 'contra'
        
    return df

def single_subj_alpha_assymetry(subject_id : str, input_dir : str, output_dir : str):

    # Use the functions defined above to create a dataframe with the alpha power score for each side for each condition
    sorted_epochs = sort_epochs(subject_id, input_dir)
    right, left = compute_alpha_by_side(sorted_epochs)
    conditions = extract_conditions(sorted_epochs)
    df = alpha_power_df(conditions, right, left)

    df.insert(0, 'ID', subject_id)

    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'alpha-power-df')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'alpha-power-df'))
    df.to_csv(os.path.join(output_dir, f'sub-{subject_id}', 'alpha-power-df', f'sub-{subject_id}-alpha-power-assymetry.csv'))

    return df

### ======================================== ALL SUBJECTS ==========================================

def alpha_assymetry_all_subj(input_dir, output_dir):
    ''' Create a dataframe for all subjects with the alpha score for each side
    
    Parameters:
    ----------
    Input_dir : str
        The path to the directory containing the epochs objects for each subject
    output_dir : str
        The path to the directory where the dataframe will be saved
    
    Returns
    ----------
    big_df : pandas.DataFrame
        A dataframe with the alpha power score for each side for each subject
        
    '''
    big_df = pd.DataFrame()

    # Empty list to store the files
    all_subj_files = []

    # Loop over the directories to access the files
    directories = glob.glob(os.path.join(input_dir, 'sub*'))
    for directory in directories:
        file = glob.glob(os.path.join(directory, 'cleaned_epochs', 'sub*N2pc.fif'))
        all_subj_files.append(file[0])
    all_subj_files.sort()

    # Pattern to extract the subject ID
    pattern = r'sub-(\d{2})-cleaned'
    
    for subject in all_subj_files:
        
        # Extract the subject ID
        match = re.search(pattern, subject)
        if match:
            subj_id = match.group(1)
        else:
            print("No match found in:", subject)
        
        print(f'========================= working on {subj_id}')
        
        epochs = sort_epochs(subj_id, input_dir)
    
        right_power_list, left_power_list = compute_alpha_by_side(epochs)
        
        conditions = extract_conditions(epochs)
        
        subj_df = alpha_power_df(conditions, right_power_list, left_power_list)
        
        subj_df.insert(0, 'ID', str(subj_id))
        
        big_df = pd.concat([big_df, subj_df], ignore_index=True)
        
        print(f'========================= data from {subj_id} added to the dataframe :D')
    
    if not os.path.exists(os.path.join(output_dir, 'alpha-power-df')):
        os.makedirs(os.path.join(output_dir, 'alpha-power-df'))
    big_df.to_csv(os.path.join(output_dir, 'alpha-power-df', 'all_subj_alpha_power_assymetry.csv'))
    
    return big_df