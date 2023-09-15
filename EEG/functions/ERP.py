import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import math
from functions.file_management import add_sub0


def to_evoked(subject_id, task, input_dir):
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
    file = os.path.join(input_dir, f'sub-{subject_id}/cleaned_epochs/sub-{subject_id}-cleaned_epochs-{task}.fif')
    epochs = mne.read_epochs(file)

    if task == 'N2pc':

        # crop the epochs to the relevant time window
        tmin = -0.2
        tmax = 0.4
        epochs.crop(tmin=tmin, tmax=tmax)
            
        # define the bins
        bins = {'bin1' : ['dis_top/target_l','dis_bot/target_l'],
                'bin2' : ['dis_top/target_r','dis_bot/target_r'],
                'bin3' : ['no_dis/target_l'],
                'bin4' : ['no_dis/target_r'],
                'bin5' : ['dis_right/target_l'],
                'bin6' : ['dis_left/target_r']}

        # create evoked
        evoked_list = [epochs[bin].average() for bin in bins.values()]
        
        # rename the distractor mid conditions to simplify
        evoked_1 = evoked_list[0]
        evoked_2 = evoked_list[1]
        evoked_1.comment = 'dis_mid/target_l'
        evoked_2.comment = 'dis_mid/target_r'
        
        # replace the '/' that causes problems when saving
        for evoked in evoked_list:
            evoked.comment = evoked.comment.replace('/', '_')
        

        # save the evoked objects in subject directory
        if not os.path.exists(os.path.join(input_dir, f'sub-{subject_id}', f'evoked-{task}')):
            os.makedirs(os.path.join(input_dir, f'sub-{subject_id}', f'evoked-{task}'))
        for evoked in evoked_list:
            print(evoked.comment)
            evoked.save(os.path.join(input_dir, f'sub-{subject_id}', f'evoked-{task}', f'sub-{subject_id}-{evoked.comment}-ave.fif'), overwrite=True)


def get_bins_data(subject_id, input_dir, exclude_subjects=False, excluded_subjects_list=[], population=None):
    ''' This function extracts the N2pc ERP values for a given subject, or all of them
        if subject_id = 'GA'.

    Parameters
    ----------
    subject_id : str
        The subject ID OR 'GA' to get the grand average.
    input_dir : str
        The path to the directory containing the input data.
    exclude_subjects : bool
        Whether to exclude subjects from the grand average.
    excluded_subjects_list : list
        List of subjects to be excluded from the grand average.
    population : str
        Population (control or stroke).
    
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
    
        subject_id = str(subject_id)
        evoked_path = os.path.join(input_dir, f'sub-{subject_id}', f'evoked-N2pc')
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
    
    
    if subject_id == 'GA':
        
        if exclude_subjects == True:
            
            # transform the excluded_subjects_list to match the format of the subject IDs
            transformed_list = add_sub0(excluded_subjects_list)
            print(transformed_list)

            # get the list of all subjects
            subject_list = glob.glob(os.path.join(input_dir, 'sub*'))

            # exclude the subjects in the excluded_subjects_list
            subjects_to_keep = []
            for subject in subject_list:
                print(subject[-6:])
                if subject[-6:] in transformed_list:
                    print(f'====================== Excluding {subject}')
                else:
                    subjects_to_keep.append(subject)
            subject_list = subjects_to_keep

        else:
            subject_list = glob.glob(os.path.join(input_dir, 'sub*'))
        
        if population == 'control':

            # keep only the control subjects (i.e. subject IDs < 50)
            subject_list = [subject for subject in subject_list if int(subject[-2:]) < 50]
            print(f'====================== subjects list, control population : {subject_list}')
            

        elif population == 'stroke':

            # keep only the stroke subjects (i.e. subject IDs > 50)
            subject_list = [subject for subject in subject_list if int(subject[-2:]) > 50]
            print(f'====================== subjects list, stroke population : {subject_list}')
            
        # initialize lists to store the data for each subject
        PO7_data_nbin1_list = []
        PO7_data_nbin2_list = []
        PO7_data_nbin3_list = []
        PO7_data_nbin4_list = []
        PO7_data_nbin5_list = []
        PO7_data_nbin6_list = []
        
        for subject in subject_list:
            
            sub_id = subject[-2:]
            b1, b2, b3, b4, b5, b6, time = get_evoked_data(sub_id, input_dir)
            
            PO7_data_nbin1_list.append(b1)
            PO7_data_nbin2_list.append(b2)
            PO7_data_nbin3_list.append(b3)
            PO7_data_nbin4_list.append(b4)
            PO7_data_nbin5_list.append(b5)
            PO7_data_nbin6_list.append(b6)
            print(f'====================== data collected for {subject}')
        
        # transform the lists in np.arrays
        PO7_data_nbin1_array = np.array(PO7_data_nbin1_list)
        PO7_data_nbin2_array = np.array(PO7_data_nbin2_list)
        PO7_data_nbin3_array = np.array(PO7_data_nbin3_list)
        PO7_data_nbin4_array = np.array(PO7_data_nbin4_list)
        PO7_data_nbin5_array = np.array(PO7_data_nbin5_list)
        PO7_data_nbin6_array = np.array(PO7_data_nbin6_list)
        
        # compute the mean of each array (should be length evk.info['sfreq']*0.6 (200ms basline, 400ms after 0))
        PO7_data_nbin1_mean = PO7_data_nbin1_array.mean(axis=0)
        PO7_data_nbin2_mean = PO7_data_nbin2_array.mean(axis=0)
        PO7_data_nbin3_mean = PO7_data_nbin3_array.mean(axis=0)
        PO7_data_nbin4_mean = PO7_data_nbin4_array.mean(axis=0)
        PO7_data_nbin5_mean = PO7_data_nbin5_array.mean(axis=0)
        PO7_data_nbin6_mean = PO7_data_nbin6_array.mean(axis=0)
        
        return PO7_data_nbin1_mean, PO7_data_nbin2_mean, PO7_data_nbin3_mean, PO7_data_nbin4_mean, PO7_data_nbin5_mean, PO7_data_nbin6_mean, time
            
    else:
        
        b1, b2, b3, b4, b5, b6, time = get_evoked_data(subject_id, input_dir)
        
        return b1, b2, b3, b4, b5, b6, time

def plot_n2pc(subject_id, input_dir, output_dir, exclude_subjects=False, excluded_subjects_list=[], population=None):
    ''' This function plots the N2pc ERP for a given subject, or a population if you specify subject_id = 'GA'.

    Parameters
    ----------
    subject_id : str
        The subject ID to plot. Can be 'GA' to plot the grand average.
    input_dir : str
        The path to the directory containing the input data.
    output_dir : str
        The path to the directory where the output will be saved.
    exclude_subjects : bool
        Whether to exclude subjects from the grand average.
    excluded_subjects_list : list
        List of subjects to be excluded from the grand average.
    population : str
        Population (control or stroke).

    Returns
    -------
    None
    '''
    # define a function to create the plots
    def create_erp_plot(subject_id, contra, ipsi, time, color, condition, title, output_dir):

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
        if subject_id == 'GA':
            plt.savefig(os.path.join(output_dir, 'all_subj','n2pc-plots', population, f'{title}.png'))
        else:
            plt.savefig(os.path.join(output_dir, f'sub-{subject_id}','n2pc-plots',f'sub-{subject_id}-PO7_{condition}.png'))
        plt.show(block=False)
        plt.close()
        
        
    if subject_id == 'GA':
        
        d1, d2, d3, d4, d5, d6, time = get_bins_data(subject_id, input_dir, exclude_subjects=exclude_subjects, excluded_subjects_list=excluded_subjects_list, population=population)
        # Create output directory if it doesn't exist
        if os.path.exists(os.path.join(output_dir, 'all_subj','n2pc-plots', population)) == False:
            os.makedirs(os.path.join(output_dir, 'all_subj','n2pc-plots', population))
        
        if population == 'control':
            str_excluded_subjects_list = [str(sub) for sub in excluded_subjects_list if sub < 50]
            excluded_subjects_string = '_'.join(str_excluded_subjects_list)
        elif population == 'stroke':
            str_excluded_subjects_list = [str(sub) for sub in excluded_subjects_list if sub > 50]
            excluded_subjects_string = '_'.join(str_excluded_subjects_list)
        
    else:
        
        d1, d2, d3, d4, d5, d6, time = get_bins_data(subject_id, input_dir)
        # Create output directory if it doesn't exist
        if os.path.exists(os.path.join(output_dir, f'sub-{subject_id}','n2pc-plots')) == False:
            os.makedirs(os.path.join(output_dir, f'sub-{subject_id}','n2pc-plots'))
        
        excluded_subjects_string = ''
        population = ''
            
        
    condition1 = 'Dis_Mid'
    condition2 = 'No_Dis'
    condition3 = 'Dis_Contra'
    
    if exclude_subjects == True:
        title1 = f'{population}-excluded_subjects-{excluded_subjects_string}-PO7-{condition1}'
    else:
        title1 = f'{population}-PO7-{condition1}'
    if exclude_subjects == True:
        title2 = f'{population}-excluded_subjects-{excluded_subjects_string}-PO7-{condition2}'
    else:
        title2 = f'{population}-PO7-{condition2}'
    if exclude_subjects == True:
        title3 = f'{population}-excluded_subjects-{excluded_subjects_string}-PO7-{condition3}'
    else:
        title3 = f'{population}-PO7-{condition3}'

    create_erp_plot(subject_id, d1, d2, time, 'blue', condition1, title1, output_dir)
    create_erp_plot(subject_id, d3, d4, time,'green', condition2, title2, output_dir)
    create_erp_plot(subject_id, d5, d6, time, 'red', condition3, title3, output_dir)


def get_n2pc_values(subject_id, input_dir, output_dir, exclude_subjects=False, excluded_subjects_list=[], population=None):
    '''
    This function extracts the N2pc values for a given subject and saves them in a csv file. 
    Can be used to extract the values for all subjects (subject_id = 'GA').
    
    Parameters
    ----------
    subject_id : str
        The subject ID to plot.
    input_dir : str
        The path to the directory containing the input data.
    output_dir : str
        The path to the directory where the output will be saved.
    exclude_subjects : bool
        Whether to exclude subjects from the grand average.
    excluded_subjects_list : list
        List of subjects to be excluded from the grand average.
    population : str
        Population (control or stroke).

    Returns
    -------
    df : pandas.DataFrame
        A dataframe containing the N2pc values for each condition and side.
    '''
    if subject_id == 'GA':
        b1, b2, b3, b4, b5, b6, time = get_bins_data(subject_id, input_dir, exclude_subjects=exclude_subjects, excluded_subjects_list=excluded_subjects_list, population=population)

    else:
        b1, b2, b3, b4, b5, b6, time = get_bins_data(subject_id, input_dir)

    bin_list = [b1, b2, b3, b4, b5, b6]

    # Define 50ms windows lists
    slices_150_200 = []
    slices_200_250 = []
    slices_250_300 = []
    slices_300_350 = []
    slices_350_400 = []
    # Define 100ms and 200ms windows lists
    slices_200_300 = []
    slices_300_400 = []
    slices_200_400 = []

    # 51 refers to the number of sample points in 100ms (sfreq = 512)
    t_150 = 51*3.5
    t_150 = math.ceil(t_150)
    t_200 = 51*4
    t_250 = 51*4.5
    t_250 = math.ceil(t_250)
    t_300 = 51*5
    t_350 = 51*5.5
    t_350 = math.ceil(t_350)
    t_400 = 51*6

    for bin_ in bin_list:

        # Slice the data into 50ms windows
        window_150_200 = bin_[t_150:t_200].mean()
        window_200_250 = bin_[t_200:t_250].mean()
        window_250_300 = bin_[t_250:t_300].mean()
        window_300_350 = bin_[t_300:t_350].mean()
        window_350_400 = bin_[t_350:t_400].mean()
       # Slice the data into 100ms and 200ms windows
        window_200_300 = bin_[t_200:t_300].mean()
        window_300_400 = bin_[t_300:t_400].mean()
        window_200_400 = bin_[t_200:t_400].mean()
        
        # Append the slices to the respective lists
        slices_150_200.append(window_150_200)
        slices_200_250.append(window_200_250)
        slices_250_300.append(window_250_300)
        slices_300_350.append(window_300_350)
        slices_350_400.append(window_350_400)
        slices_200_300.append(window_200_300)
        slices_300_400.append(window_300_400)
        slices_200_400.append(window_200_400)
        
    # Create the dataframe and store the values
    bin_names = ['Dis_mid (Contra)',
                'Dis_mid (Ipsi)',
                'No_dis (Contra)',
                'No_dis (Ipsi)',
                'Dis_contra (Contra)',
                'Dis_contra (Ipsi)']
    
    df = pd.DataFrame({'ID':subject_id,'condition and side':bin_names,
                       '150-200ms':slices_150_200,
                        '200-250ms':slices_200_250,
                        '250-300ms':slices_250_300,
                        '300-350ms':slices_300_350,
                        '350-400ms':slices_350_400,
                        '200-300ms':slices_200_300,
                        '300-400ms':slices_300_400,
                        'total 200-400ms':slices_200_400})
    
    pd.options.display.float_format = '{:.5e}'.format
    
    # Save the dataframe
    if subject_id == 'GA':
        if not os.path.exists(os.path.join(output_dir, 'all_subj', 'n2pc-values', population)):
            os.makedirs(os.path.join(output_dir, 'all_subj', 'n2pc-values', population))

        if population == 'control':
            str_excluded_subjects_list = [str(sub) for sub in excluded_subjects_list if sub < 50]
            excluded_subjects_string = '_'.join(str_excluded_subjects_list)
        elif population == 'stroke':
            str_excluded_subjects_list = [str(sub) for sub in excluded_subjects_list if sub > 50]
            excluded_subjects_string = '_'.join(str_excluded_subjects_list)
            
        title = f'{population}-excluded_subjects-{excluded_subjects_string}'

        df.to_csv(os.path.join(output_dir, 'all_subj', 'n2pc-values', population, f'{title}-n2pc_values.csv'))
    else:
        if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'n2pc-values')):
            os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'n2pc-values'))
        df.to_csv(os.path.join(output_dir, f'sub-{subject_id}', 'n2pc-values', f'sub-{subject_id}-n2pc_values.csv'))
    
    return df


##### LEGACY CODE #####

def get_population_files(input_dir, task, population):

    files = glob.glob(os.path.join(input_dir, 'all_subj', f'{population}_allsubj', '*'))
    files = [file for file in files if task in file]

    return files

def get_population_epochs(files, exclude_subjects=False, excluded_subjects_list=[]):

    def get_last_string(file):
        return file.split('/')[-1].split('-')[-1].split('.')[0]

    if exclude_subjects == True:
        excluded_subjects_list = [str(sub) for sub in excluded_subjects_list]
        excluded_subjects_string = '_'.join(excluded_subjects_list)
        specific_file = [file for file in files if get_last_string(file) == excluded_subjects_string]
    
    if exclude_subjects == False:
        files = files.sort()
        specific_file = files[0]

    epochs = mne.read_epochs(specific_file)
    
    return epochs, excluded_subjects_string
