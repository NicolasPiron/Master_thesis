import os
import numpy as np
import pandas as pd
import mne
import glob

##def alpha_power_df(subject_id):

##############################################################################################################
############################################### LEGACY, DO NOT USE ###########################################
##############################################################################################################

print('========== WARNING ==========')
print('========== OLD CODE ==========')
print('========== NOT COHERENT WITH CURRENT FILE ORGANIZATION ==========')


#Epoch 3 runs
file_dir = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/preproc/Alpheye_out/data/raw_ica/'
out_dir = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/preproc/Alpheye_out/data/epochs/'
subject_id = 'S03'
raw_files = glob.glob(file_dir + subject_id + '*')
raw_files.sort()

for i, file in enumerate(raw_files):
    
    raw = mne.io.read_raw_fif(os.path.join(file_dir, file))
    
    e_list = mne.find_events(raw, stim_channel='Status')
    df = pd.DataFrame(e_list, columns=['timepoint', 'duration', 'stim'])
    df_stim = df[(df['stim'] == 2) | (df['stim'] == 4)].reset_index()
    df_stim = df_stim.add_prefix('img_')

    mne_events = df_stim[['img_timepoint', 'img_duration', 'img_stim']].values
    event_dict = {'Landscape':2,
            'Human':4}
    
    run = f'r{i+1}'
    file_name = subject_id+'_'+run+'_'+'epochs.fif'
    
    epochs = mne.Epochs(raw, mne_events, event_id=event_dict, tmin=-0.2, tmax=6, baseline=(-0.2, 0), event_repeated='drop', preload=True)
    epochs.set_eeg_reference(ref_channels='average')
    epochs.save(os.path.join(out_dir,file_name), overwrite=True)


#Compute and put alpha scores in a dataframe with subject, run, and trial info

file_dir = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/preproc/Alpheye_out/data/epochs/'
files = glob.glob(file_dir + subject_id + '*')
files.sort()

for index, file in enumerate(files):
    
    epochs=mne.read_epochs(file)

    freqs=np.arange(8, 13)
    n_cycles=freqs/2.
    time_bandwidth=4.
    baseline=None  # no baseline correction
    n_jobs=1  # number of parallel jobs to run

    elec_names=['Fp2', 'AF8', 'AF4','F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2', 'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1']

    #==========Create new dict to have a place to store alpha power values
    all_elec={}

    for string in elec_names:
        new_list=[]
        all_elec[string]=new_list

    #==========Compute alpha power for each electrode for each epoch and append it to the list associated with the electrodes
    epoch_indexes=range(len(epochs))
    for i in epoch_indexes:
        for elec_name, power_list in all_elec.items():
            power=mne.time_frequency.tfr_morlet(epochs[i], freqs=freqs, n_cycles=n_cycles, picks=elec_name,
                                            use_fft=True, return_itc=False, decim=1,
                                            n_jobs=n_jobs, verbose=True)
            power_list.append(power)
        #print(f'alpha power of epoch {i+1} computed')
    
    #==========Create new dict to have a place to store only the alpha power score associated to each electrode
    all_elec_scores={}

    for string in elec_names:
        scores_list=[]
        all_elec_scores[string] = scores_list    

    #==========Get only one power value and append it to each list of the new score dict
    for elec_name, power_list in all_elec.items():
        for power in power_list:
            alpha_power=power.data[:, freqs == 10, :].mean(axis=1).mean(axis=1).mean()
            all_elec_scores[elec_name].append(alpha_power)
        print(f'Alpha power values of {elec_name} retrieved')

    #==========Get the hemisphere info
    hemi_info={}

    for elec_name in all_elec.keys():
        last_digit=elec_name[-1]
        if float(last_digit)%2 == 0:
            hemi_info[elec_name]='right'
        else:
            hemi_info[elec_name]='left'
        
    #==========Create a dataframe to put all the values in order
    column_names=['subject', 'run', 'electrodes', 'hemisphere', 'trial', 'image', 'category', 'alpha_power']
    df = pd.DataFrame(columns=column_names)
    df_fix=pd.read_csv('/Users/nicolaspiron/Documents/PULSATION/Python_MNE/preproc/SESSION_preprocdata_fixations.txt', delimiter='\t')
    #I want to display very small values
    pd.set_option('display.float_format', lambda x: '%.15f' % x)

    #==========Put all data in dataframe
    #need to have the number of electrodes times the number of trials rows
    nrows = len(all_elec_scores.keys())*len(all_elec_scores['Fp2'])
    repeat_count = len(elec_names)

    #Add the subject number and run
    df = df.reindex(range(nrows))
    df['subject'] = subject_id
    df['run']=index+1

    for i in range(len(df)):
        #Add the trial index
        trial = (i // repeat_count) + 1  
        df.iloc[i,4] = trial
        #Add the electrode names
        idx = i % len(elec_names)
        df.iloc[i, 2] = elec_names[idx]
        #Add the lateralization info
        key = df.iloc[i,2]
        df.iloc[i,3] = hemi_info[key]
        #Add image number
        fix_condition = (df_fix['subject'] == subject_id) & (df_fix['run'] == int(index+1)) & (df_fix['trial'] == trial)
        fix_img = df_fix.loc[fix_condition, 'image'].values[0]
        df.iloc[i,5] = fix_img
        #Add image category
        fix_category= df_fix.loc[fix_condition, 'category'].values[0]
        df.iloc[i,6] = fix_category
    
    #Add the alpha power scores for each electrode and each trial
    for i, row in df.iterrows():
        trial = (i // repeat_count) + 1 
        key = row['electrodes']
        a_power = all_elec_scores[key]
        a_score = a_power[trial-1]
        df.iloc[i,7] = a_score
    
    out_path = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/preproc/Alpheye_out/dataframe/'
    csv_name = out_path + subject_id + '_r' + str(index+1) + '_df.csv'
    df.to_csv(csv_name, index=False)

#==========Concatenate the 3 dataframes

csv_path = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/preproc/Alpheye_out/dataframe/'
csv_files = glob.glob(csv_path + subject_id + '*')
for index, file in enumerate(csv_files):
    df1 = pd.read_csv(csv_path + subject_id + '_r1_df.csv')
    df2 = pd.read_csv(csv_path + subject_id + '_r2_df.csv')
    df3 = pd.read_csv(csv_path + subject_id + '_r3_df.csv')
    df_concat = pd.concat([df1, df2, df3], axis=0).reset_index(drop=True)
    df_concat.to_csv(csv_path + 'df_subj/' + subject_id + '_df.csv', index=False)

print('==========FILES CONCATENATED========== :D')