import os
import mne
import numpy as np
import pandas as pd
import math
from n2pc_func.set_paths import get_paths



def get_df_n2pc_values_epoch_young(subject_id, input_dir, output_dir):
    ''' Compute the difference bewteen the ipsi and contra channels (PO7-PO8) for each epoch and save the values in a dataframe

    Parameters
    ----------
    subject_id : str
        The subject ID to plot.
    input_dir : str
        The path to the directory containing the input data.

    Returns
    -------
    df : pandas.DataFrame
        A dataframe containing the N2pc values for each condition and side.
    '''

    # load data
    file = os.path.join(input_dir, f'sub-{subject_id}/N2pc/cleaned_epochs/sub-{subject_id}-cleaned_epochs-N2pc.fif')
    epochs = mne.read_epochs(file)

    # crop epochs to relevent time window
    epochs.crop(tmin=0, tmax=0.8)
    
    # get the reeject log (preprocessing step) for the subject
    reject_log = np.load(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'preprocessing', 'step-06-final-reject_log', f'sub-{subject_id}-final-reject_log-N2pc.npz'))
    # Define the epochs status (rejected or not)
    epochs_status = reject_log['bad_epochs']

    # initialize the df
    df = pd.DataFrame(columns=['ID','epoch_index', 'epoch_dropped', 'index_reset', 'saccade', 'condition', 'target_side',
                                '180_300ms'])
    
    # create row for each epoch
    df['epoch_index'] = range(1,len(epochs_status)+1)
    df['epoch_dropped'] = epochs_status
    df['ID'] = subject_id

    # add a column that store the reset index of the epochs
    index_val = 0
    index_list = []
    n_valid = []
    # iterate through the 'epoch_dropped' column to create the reset index column
    for row_number in range(len(df)):
        if df.iloc[row_number, 2] == False:
            index_list.append(index_val)
            n_valid.append(index_val)
            index_val += 1
        else:
            index_list.append(np.nan)

    # add the index column to the DataFrame
    df['index_reset'] = index_list

    # Load the csv file contaning the indices of epochs with saccades
    saccades = pd.read_csv(os.path.join(input_dir, f'sub-{subject_id}', 'N2pc', 'heog-artifact', 'rejected-epochs-list', f'sub-{subject_id}-heog-artifact.csv'))
    # Create a list of the indices of the epochs with saccades
    saccades_list = list(saccades['index'])
    # Add a column that specifies if the epoch contains a saccade. FALSE if no saccade, TRUE if saccade. 
    df['saccade'] = df['index_reset'].isin(saccades_list)

    # iterate through the rows of the DataFrame to fill the columns
    for row_number in range(len(df)):
        
        # check if the epoch is dropped
        if df.iloc[row_number, 2] == True:
            print(f'========= epoch {row_number+1} was dropped',)
        else:
            print(f'========= epoch {row_number+1} was keeped')

            # compute the data to fill the dataframe

            # get the epoch index (after epochs rejection)
            epoch_idx = int(df['index_reset'].loc[row_number])
            
            # get the data from the channels of interest
            PO7 = epochs[epoch_idx].get_data(picks=['PO7'])
            PO8 = epochs[epoch_idx].get_data(picks=['PO8'])
            PO7 = PO7.reshape(410)
            PO8 = PO8.reshape(410)
            
            # find where is ispsilateral and contralateral to the target
            epoch_id = epochs.events[epoch_idx][2]
            if epoch_id in [1, 3, 5, 7]:
                target_side = 'left'
                ipsi = PO7
                contra = PO8
            elif epoch_id in [2, 4, 6, 8]:
                target_side = 'right'
                ipsi = PO8
                contra = PO7

            # get the difference between the channels
            diff = contra - ipsi
            
            if epoch_id in [1, 2, 5, 6]:
                cond = 'Dis_mid'
            elif epoch_id in [3, 4]:
                cond = 'No_dis'
            elif epoch_id in [7, 8]:
                cond = 'Dis_contra'

            # create the time points based on sfreq
            sfreq = epochs.info['sfreq'] 

            t_180 = sfreq * 0.180
            t_180 = math.ceil(t_180)
            t_300 = sfreq * 0.300
            t_300 = math.ceil(t_300)
            
            # slice the data into 50ms and 100ms windows
            
            diff_180_300 = diff[t_180:t_300].mean()

            # fill the dataframe with everything we just computed 
            df.iloc[row_number, 5] = cond
            df.iloc[row_number, 6] = target_side
            df.iloc[row_number, 7] = diff_180_300           
    
    print(f'========== df created for subject {subject_id}')

    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-values')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-values'))
    df.to_csv(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-values', f'sub-{subject_id}-n2pc_values_per_epoch.csv'))

    print(f'========== df saved for subject {subject_id}')


def loop_over_all_subj(input_dir, output_dir):
    
    subject_list = [70, 71, 72, 73, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87]
    #subdirectories = [f'sub-{str(sub).zfill(2)}' for sub in subject_list]

    # Loop over the subdirectories and create the dataframe for each subject
    for subject_id in subject_list:
        # Compute n2pc values and save thems in a dataframe

        try:
            get_df_n2pc_values_epoch_young(subject_id, input_dir, output_dir)
            print(f'==================== Dataframe created and saved for subject {subject_id}! :)')
        except:
            print(f"==================== No data (epochs or reject log) for subject {subject_id}! O_o'")
            continue


    df_list = []
    missing_subj = []
    for subject_id in subject_list:
        file = os.path.join(input_dir, f'sub-{subject_id}','N2pc', 'n2pc-values', f'sub-{subject_id}-n2pc_values_per_epoch.csv')
        if os.path.exists(file):
            df_list.append(pd.read_csv(file))
        else:
            print(f"==================== No dataframe for subject {subject_id}! O_o'")
            missing_subj.append(subject_id)
    # Concatenate all dataframes in the list
    df = pd.concat(df_list)
    # Save dataframe as .csv file
    if not os.path.exists(os.path.join(output_dir, 'all_subj','N2pc', 'n2pc-values', 'n2pc-values-per-epoch')):
        os.makedirs(os.path.join(output_dir, 'all_subj','N2pc', 'n2pc-values', 'n2pc-values-per-epoch'))
    df.to_csv(os.path.join(output_dir, 'all_subj','N2pc', 'n2pc-values', 'n2pc-values-per-epoch', 'n2pc_values_per_epoch-young.csv'), index=False)
    print(f'==================== Dataframe created and saved for all young subjects! :) (except for {missing_subj})')

if __name__ == '__main__':

    input_dir, output_dir = get_paths()
    loop_over_all_subj(input_dir, output_dir)