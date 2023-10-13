import mne
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

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
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'evoked-HEOG')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'evoked-HEOG'))
    np.save(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'evoked-HEOG', f'sub-{subject_id}-dis_mid_diff.npy'), dis_mid_diff)
    np.save(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'evoked-HEOG', f'sub-{subject_id}-dis_side_diff.npy'), dis_side_diff)
    np.save(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'evoked-HEOG', f'sub-{subject_id}-no_dis_diff.npy'), no_dis_diff)
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
    if not os.path.exists(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-plots', 'heog-waveform')):
        os.makedirs(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-plots', 'heog-waveform'))
    fig.savefig(os.path.join(output_dir, f'sub-{subject_id}', 'N2pc', 'n2pc-plots', 'heog-waveform', f'sub-{subject_id}-heog-erp.png'))
    
    print('========== HEOG ERP plot saved ==========')

    return None