import glob
import os
import mne

def concat_all_subj(task='n2pc', population='control'):
    ''' Concatenate the files of all the subjects that have been preprocessed.
    
    Parameters
    ---------- 
    task : 'alpheye' or 'n2pc'
        The name of the task
    population : 'control' or 'patient'
        The type of population
    
    Returns
    ----------
    Nothing
    '''

    if task == 'alpheye':

        data_path = '/Users/nicolaspiron/Documents/PULSATION/Data/EEG/Alpheye/data-preproc-alpheye/total_epochs/'
        out_path = '/Users/nicolaspiron/Documents/PULSATION/Data/EEG/Alpheye/data-preproc-alpheye/epochs_all/'
        
        files = glob.glob(data_path+'*')
        files.sort()
        total_epoch_files = []

        if population == 'control':

            for file in files:
                if data_path + 'S' in file:
                    i = mne.read_epochs(file)
                    total_epoch_files.append(i)
        
            all_subj = mne.concatenate_epochs(total_epoch_files)
            all_subj.save(os.path.join(out_path,'S_all_subj_alpheye_.fif'), overwrite=True)
        
        elif population == 'patient':

            for file in files:
                if data_path + 'P' in file:
                    i = mne.read_epochs(file)
                    total_epoch_files.append(i)

            all_subj = mne.concatenate_epochs(total_epoch_files)
            all_subj.save(os.path.join(out_path,'P_all_subj_alpheye_.fif'), overwrite=True)


    elif task == 'n2pc':

        data_path = '/Users/nicolaspiron/Documents/PULSATION/Data/EEG/N2pc/data-preproc-n2pc/total_epochs/'
        out_path = '/Users/nicolaspiron/Documents/PULSATION/Data/EEG/N2pc/data-preproc-n2pc/epochs_all/'
        
        files = glob.glob(data_path+'*')
        files.sort()
        total_epoch_files = []

        if population == 'control':

            for file in files:
                if data_path + 'S' in file:
                    i = mne.read_epochs(file)
                    total_epoch_files.append(i)
                        
            all_subj = mne.concatenate_epochs(total_epoch_files)
            all_subj.save(os.path.join(out_path,'S_all_subj_n2pc_.fif'), overwrite=True)                        
        
        elif population == 'patient':

            for file in files:
                if data_path + 'P' in file:
                    i = mne.read_epochs(file)
                    total_epoch_files.append(i)

        all_subj = mne.concatenate_epochs(total_epoch_files)
        all_subj.save(os.path.join(out_path,'P_all_subj_n2pc_.fif'), overwrite=True)