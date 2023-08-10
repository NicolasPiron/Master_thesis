import glob
import os
import mne

def concat_all_subj(task='n2pc'):
    ''' Concatenate the files of all the subjects that have been preprocessed.
    
    Parameters
    ---------- 
    task : 'alpheye' or 'n2pc'
        The name of the task
    
    Returns
    ----------
    Nothing
    '''

    if task == 'alpheye':

        data_path = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/preproc/Alpheye_out/data/epochs/'
        out_path = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/preproc/Alpheye_out/data/all_subj/'
        files = glob.glob(data_path+'*')
        files.sort()
        total_epoch_files = []

        for i, file in enumerate(files):
            if 'total' in file:
                i = mne.read_epochs(file)
                total_epoch_files.append(i)

        all_subj = mne.concatenate_epochs(total_epoch_files)
        all_subj.save(os.path.join(out_path,'all_subj_alpheye_.fif'), overwrite=True)


    elif task == 'n2pc':

        data_path = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/preproc/n2pc_out/data/epochs/'
        out_path = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/preproc/n2pc_out/data/all_subj/'
        files = glob.glob(data_path+'*')
        files.sort()
        total_epoch_files = []

        for i, file in enumerate(files):
            if 'total' in file:
                i = mne.read_epochs(file)
                total_epoch_files.append(i)

        all_subj = mne.concatenate_epochs(total_epoch_files)
        all_subj.save(os.path.join(out_path,'all_subj_n2pc_.fif'), overwrite=True)