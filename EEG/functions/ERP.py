import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import mne

def plot_n2pc(subject_id, input_dir, output_dir=None):
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
    # Load subject data and crop to the relevant time window
    epochs = mne.read_epochs(os.path.join(input_dir, f'sub-{subject_id}', 'cleaned_epochs', f'sub-{subject_id}-cleaned_epochs-N2pc.fif'))
    tmin = -0.2
    tmax = 0.4
    epochs.crop(tmin=tmin, tmax=tmax)
    # Create output directory if it doesn't exist
    if os.path.exists(os.path.join(output_dir, f'sub-{subject_id}','n2pc-plots')) == False:
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}','n2pc-plots'))
    
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
    substract_weights = [1, -1]

    # Prepare Contra and Ipsi bins
    nbin1 = bin_operator(evk_bin1_R.data, evk_bin2_L.data)
    nbin2 = bin_operator(evk_bin1_L.data, evk_bin2_R.data)
    nbin3 = bin_operator(evk_bin3_R.data, evk_bin4_L.data)
    nbin4 = bin_operator(evk_bin3_L.data, evk_bin4_R.data)
    nbin5 = bin_operator(evk_bin5_R.data, evk_bin6_L.data)
    nbin6 = bin_operator(evk_bin5_L.data, evk_bin6_R.data)

    # Apply weights to the new bins
    nbin7 = mne.combine_evoked([evk_bin1, evk_bin2], weights=substract_weights)
    nbin8 = mne.combine_evoked([evk_bin3, evk_bin4], weights=substract_weights)
    nbin9 = mne.combine_evoked([evk_bin5, evk_bin6], weights=substract_weights)


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

    # Create three separate plots for each condition

    # Plot for Dis_Mid
    plt.figure(figsize=(10, 6))
    plt.plot(time, PO7_data_nbin1_baseline, color='blue', label='Dis_Mid (Contralateral)')
    plt.plot(time, PO7_data_nbin2_baseline, color='blue', linestyle='dashed', label='Dis_Mid (Ipsilateral)')
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    plt.axhline(y=0, color='black', linewidth=1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (uV)')
    plt.title('Signal from Electrodes PO7 - Dis_Mid Condition')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, f'sub-{subject_id}','n2pc-plots',f'sub-{subject_id}-PO7_Dis_Mid.png'))
    plt.show()

    # Plot for No_Dis
    plt.figure(figsize=(10, 6))
    plt.plot(time, PO7_data_nbin3_baseline, color='green', label='No_Dis (Contralateral)')
    plt.plot(time, PO7_data_nbin4_baseline, color='green', linestyle='dashed', label='No_Dis (Ipsilateral)')
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    plt.axhline(y=0, color='black', linewidth=1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (uV)')
    plt.title('Signal from Electrodes PO7 - No_Dis Condition')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, f'sub-{subject_id}','n2pc-plots', f'sub-{subject_id}-PO7_No_Dis.png'))
    plt.show()

    # Plot for Dis_Contra
    plt.figure(figsize=(10, 6))
    plt.plot(time, PO7_data_nbin5_baseline, color='red', label='Dis_Contra (Contralateral)')
    plt.plot(time, PO7_data_nbin6_baseline, color='red', linestyle='dashed', label='Dis_Contra (Ipsilateral)')
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    plt.axhline(y=0, color='black', linewidth=1)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (uV)')
    plt.title('Signal from Electrodes PO7 - Dis_Contra Condition')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, f'sub-{subject_id}','n2pc-plots', f'sub-{subject_id}-PO7_Dis_Contra.png'))
    plt.show()


