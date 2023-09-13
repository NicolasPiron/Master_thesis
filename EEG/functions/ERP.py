import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import math

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

def get_bins_baseline(subject_id, input_dir, epochs_=None):
    ''' This function extracts the N2pc ERP for a given subject.

    Parameters
    ----------
    subject_id : str
        The subject ID to plot OR 'GA' to plot grand average.
    input_dir : str
        The path to the directory containing the input data.
    GA : bool
        Whether to use grand average data or not.
    
    Returns
    -------
    PO7_data_nbin1_baseline : numpy.ndarray
        The N2pc ERP data for the Dis_Mid contra condition.
    PO7_data_nbin2_baseline : numpy.ndarray
        The N2pc ERP data for the Dis_Mid ipsi condition.
    PO7_data_nbin3_baseline : numpy.ndarray
        The N2pc ERP data for the No_Dis contra condition.
    PO7_data_nbin4_baseline : numpy.ndarray
        The N2pc ERP data for the No_Dis ipsi condition.
    PO7_data_nbin5_baseline : numpy.ndarray
        The N2pc ERP data for the Dis_Contra contra condition.
    PO7_data_nbin6_baseline : numpy.ndarray
        The N2pc ERP data for the Dis_Contra ipsi condition.
    time : numpy.ndarray
        The time axis for the ERP data.
    '''
    # Load subject data and crop to the relevant time window
    if subject_id == 'GA':
        epochs = epochs_.load_data()
    else:
        epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', 'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-N2pc.fif'))
    tmin = -0.2
    tmax = 0.4
    epochs.crop(tmin=tmin, tmax=tmax)
    
    # Define the channel indices for left (Lch) and right (Rch) channels
    Lch = np.concatenate([np.arange(0, 27)])
    Rch = np.concatenate([np.arange(33, 36), np.arange(38, 46), np.arange(48, 64)])

    # Define functions to create the new bin operations
    def bin_operator(data1, data2):
        return 0.5 * data1 + 0.5 * data2
    
    bin1 = ['dis_top/target_l','dis_bot/target_l']
    bin2 = ['dis_top/target_r','dis_bot/target_r']
    bin3 = ['no_dis/target_l']
    bin4 = ['no_dis/target_r']
    bin5 = ['dis_right/target_l']
    bin6 = ['dis_left/target_r']


    # Create evoked objects for each bin
    evk_bin1_R = epochs[bin1].average(picks=Rch)
    evk_bin1_L = epochs[bin1].average(picks=Lch)
    evk_bin2_R = epochs[bin2].average(picks=Rch)
    evk_bin2_L = epochs[bin2].average(picks=Lch)
    evk_bin3_R = epochs[bin3].average(picks=Rch)
    evk_bin3_L = epochs[bin3].average(picks=Lch)
    evk_bin4_R = epochs[bin4].average(picks=Rch)
    evk_bin4_L = epochs[bin4].average(picks=Lch)
    evk_bin5_R = epochs[bin5].average(picks=Rch)
    evk_bin5_L = epochs[bin5].average(picks=Lch)
    evk_bin6_R = epochs[bin6].average(picks=Rch)
    evk_bin6_L = epochs[bin6].average(picks=Lch)

    evk_bin1 = epochs[bin1].average()
    evk_bin2 = epochs[bin2].average()
    evk_bin3 = epochs[bin3].average()
    evk_bin4 = epochs[bin4].average()
    evk_bin5 = epochs[bin5].average()
    evk_bin6 = epochs[bin6].average()

    # Create weights for the new bin operations
    #substract_weights = [1, -1]

    # Prepare Contra and Ipsi bins
    nbin1 = bin_operator(evk_bin1_R.data, evk_bin2_L.data)
    nbin2 = bin_operator(evk_bin1_L.data, evk_bin2_R.data)
    nbin3 = bin_operator(evk_bin3_R.data, evk_bin4_L.data)
    nbin4 = bin_operator(evk_bin3_L.data, evk_bin4_R.data)
    nbin5 = bin_operator(evk_bin5_R.data, evk_bin6_L.data)
    nbin6 = bin_operator(evk_bin5_L.data, evk_bin6_R.data)

    # Apply weights to the new bins
    #nbin7 = mne.combine_evoked([evk_bin1, evk_bin2], weights=substract_weights)
    #nbin8 = mne.combine_evoked([evk_bin3, evk_bin4], weights=substract_weights)
    #nbin9 = mne.combine_evoked([evk_bin5, evk_bin6], weights=substract_weights)


    # Define the baseline period in milliseconds
    baseline = (-200, 0)  

    # Define the time axis
    time = evk_bin1.times * 1000  # Convert to milliseconds

    # Define the channel indices for P7, P9, and PO7
    P7_idx = evk_bin1.info['ch_names'].index('P7')
    P9_idx = evk_bin1.info['ch_names'].index('P9')
    PO7_idx = evk_bin1.info['ch_names'].index('PO7')

    # Extract the data for P7, P9, and PO7 electrodes
    PO7_data_nbin1 = nbin1[PO7_idx]
    PO7_data_nbin2 = nbin2[PO7_idx]
    PO7_data_nbin3 = nbin3[PO7_idx]
    PO7_data_nbin4 = nbin4[PO7_idx]
    PO7_data_nbin5 = nbin5[PO7_idx]
    PO7_data_nbin6 = nbin6[PO7_idx]

    # Apply baseline correction to the ERP data
    PO7_data_nbin1_baseline = mne.baseline.rescale(PO7_data_nbin1, times=time, baseline=baseline)
    PO7_data_nbin2_baseline = mne.baseline.rescale(PO7_data_nbin2, times=time, baseline=baseline)
    PO7_data_nbin3_baseline = mne.baseline.rescale(PO7_data_nbin3, times=time, baseline=baseline)
    PO7_data_nbin4_baseline = mne.baseline.rescale(PO7_data_nbin4, times=time, baseline=baseline)
    PO7_data_nbin5_baseline = mne.baseline.rescale(PO7_data_nbin5, times=time, baseline=baseline)
    PO7_data_nbin6_baseline = mne.baseline.rescale(PO7_data_nbin6, times=time, baseline=baseline)

    return PO7_data_nbin1_baseline, PO7_data_nbin2_baseline, PO7_data_nbin3_baseline, PO7_data_nbin4_baseline, PO7_data_nbin5_baseline, PO7_data_nbin6_baseline, time

def plot_n2pc(subject_id, input_dir, output_dir, epochs_=None, title=None):
    ''' This function plots the N2pc ERP for a given subject.

    Parameters
    ----------
    subject_id : str
        The subject ID to plot.
    input_dir : str
        The path to the directory containing the input data.
    output_dir : str
        The path to the directory where the output will be saved.

    
    Returns
    -------
    None
    '''
    if subject_id == 'GA':
        PO7_data_nbin1_baseline, PO7_data_nbin2_baseline, PO7_data_nbin3_baseline, PO7_data_nbin4_baseline, PO7_data_nbin5_baseline, PO7_data_nbin6_baseline, time = get_bins_baseline(subject_id, input_dir, epochs_=epochs_)
        # Create output directory if it doesn't exist
        if os.path.exists(os.path.join(output_dir, 'all_subj','n2pc-plots')) == False:
            os.makedirs(os.path.join(output_dir, 'all_subj','n2pc-plots'))

    else:
        PO7_data_nbin1_baseline, PO7_data_nbin2_baseline, PO7_data_nbin3_baseline, PO7_data_nbin4_baseline, PO7_data_nbin5_baseline, PO7_data_nbin6_baseline, time = get_bins_baseline(subject_id, input_dir)
        # Create output directory if it doesn't exist
        if os.path.exists(os.path.join(output_dir, f'sub-{subject_id}','n2pc-plots')) == False:
            os.makedirs(os.path.join(output_dir, f'sub-{subject_id}','n2pc-plots'))

    # Create three separate plots for each condition


    def create_erp_plot(subject_id, contra, ipsi, time, color, title, output_dir):

        plt.figure(figsize=(10, 6))
        plt.plot(time, contra, color=color, label='Dis_Mid (Contralateral)')
        plt.plot(time, ipsi, color=color, linestyle='dashed', label='Dis_Mid (Ipsilateral)')
        plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
        plt.axhline(y=0, color='black', linewidth=1)
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude (uV)')
        plt.title('Signal from Electrodes PO7 - Dis_Mid Condition')
        plt.legend()
        plt.grid()
        if subject_id == 'GA':
            plt.savefig(os.path.join(output_dir, 'all_subj','n2pc-plots',f'{title}-PO7_Dis_Mid.png'))
        else:
            plt.savefig(os.path.join(output_dir, f'sub-{subject_id}','n2pc-plots',f'sub-{subject_id}-PO7_Dis_Mid.png'))
        plt.show(block=False)
        plt.close()
    
    create_erp_plot(subject_id, PO7_data_nbin1_baseline, PO7_data_nbin2_baseline, 'blue', title, output_dir)

    create_erp_plot(subject_id, PO7_data_nbin3_baseline, PO7_data_nbin4_baseline, 'green', title, output_dir)

    create_erp_plot(subject_id, PO7_data_nbin5_baseline, PO7_data_nbin6_baseline, 'red', title, output_dir)


    # Plot for Dis_Mid
    #plt.figure(figsize=(10, 6))
    #plt.plot(time, PO7_data_nbin1_baseline, color='blue', label='Dis_Mid (Contralateral)')
    #plt.plot(time, PO7_data_nbin2_baseline, color='blue', linestyle='dashed', label='Dis_Mid (Ipsilateral)')
    #plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    #plt.axhline(y=0, color='black', linewidth=1)
    #plt.xlabel('Time (ms)')
    #plt.ylabel('Amplitude (uV)')
    #plt.title('Signal from Electrodes PO7 - Dis_Mid Condition')
    #plt.legend()
    #plt.grid()
    #plt.savefig(os.path.join(output_dir, f'sub-{subject_id}','n2pc-plots',f'sub-{subject_id}-PO7_Dis_Mid.png'))
    #plt.show(block=False)
    #plt.close()

    # Plot for No_Dis
    #plt.figure(figsize=(10, 6))
    #plt.plot(time, PO7_data_nbin3_baseline, color='green', label='No_Dis (Contralateral)')
    #plt.plot(time, PO7_data_nbin4_baseline, color='green', linestyle='dashed', label='No_Dis (Ipsilateral)')
    #plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    #plt.axhline(y=0, color='black', linewidth=1)
    #plt.xlabel('Time (ms)')
    #plt.ylabel('Amplitude (uV)')
    #plt.title('Signal from Electrodes PO7 - No_Dis Condition')
    #plt.legend()
    #plt.grid()
    #plt.savefig(os.path.join(output_dir, f'sub-{subject_id}','n2pc-plots', f'sub-{subject_id}-PO7_No_Dis.png'))
    #plt.show(block=False)
    #plt.close()

    # Plot for Dis_Contra
    #plt.figure(figsize=(10, 6))
    #plt.plot(time, PO7_data_nbin5_baseline, color='red', label='Dis_Contra (Contralateral)')
    #plt.plot(time, PO7_data_nbin6_baseline, color='red', linestyle='dashed', label='Dis_Contra (Ipsilateral)')
    #plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    #plt.axhline(y=0, color='black', linewidth=1)
    #plt.xlabel('Time (ms)')
    #plt.ylabel('Amplitude (uV)')
    #plt.title('Signal from Electrodes PO7 - Dis_Contra Condition')
    #plt.legend()
    #plt.grid()
    #plt.savefig(os.path.join(output_dir, f'sub-{subject_id}','n2pc-plots', f'sub-{subject_id}-PO7_Dis_Contra.png'))
    #plt.show(block=False)
    #plt.close()

def get_n2pc_values(subject_id, input_dir, output_dir, epochs_=None, title=None):
    '''
    This function extracts the N2pc values for a given subject and saves them in a csv file.
    
    Parameters
    ----------
    subject_id : str
        The subject ID to plot.
    input_dir : str
        The path to the directory containing the input data.
    output_dir : str
        The path to the directory where the output will be saved.

    Returns
    -------
    df : pandas.DataFrame
        A dataframe containing the N2pc values for each condition and side.
    '''
    if subject_id == 'GA':
        PO7_data_nbin1_baseline, PO7_data_nbin2_baseline, PO7_data_nbin3_baseline, PO7_data_nbin4_baseline, PO7_data_nbin5_baseline, PO7_data_nbin6_baseline, time = get_bins_baseline(subject_id, input_dir, epochs_=epochs_)

    else:
        PO7_data_nbin1_baseline, PO7_data_nbin2_baseline, PO7_data_nbin3_baseline, PO7_data_nbin4_baseline, PO7_data_nbin5_baseline, PO7_data_nbin6_baseline, time = get_bins_baseline(subject_id, input_dir)

    bin_list = [PO7_data_nbin1_baseline,
                PO7_data_nbin2_baseline,
                PO7_data_nbin3_baseline,
                PO7_data_nbin4_baseline,
                PO7_data_nbin5_baseline,
                PO7_data_nbin6_baseline]
    
    #bin_list = [PO7_data_nbin1,
     #           PO7_data_nbin2,
      #          PO7_data_nbin3,
       #         PO7_data_nbin4,
        #        PO7_data_nbin5,
         #       PO7_data_nbin6]

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
        if not os.path.exists(os.path.join(output_dir, 'all_subj', 'n2pc-values')):
            os.makedirs(os.path.join(output_dir, 'all_subj', 'n2pc-values'))
        df.to_csv(os.path.join(output_dir, 'all_subj', 'n2pc-values', f'{title}-n2pc_values.csv'))
    else:
        if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'n2pc-values')):
            os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'n2pc-values'))
        df.to_csv(os.path.join(output_dir, f'sub-{subject_id}', 'n2pc-values', f'sub-{subject_id}-n2pc_values.csv'))
    
    return df
