import glob
import os
import mne

def concat_all_subj(task='N2pc', population='control', input_dir=None, output_dir=None):
    ''' Concatenate the files of all the subjects that have been preprocessed.
    
    Parameters
    ---------- 
    task : 'N2pc' or 'Alpheye'
        The name of the task
    population : 'control' or 'stroke'
        The type of population
    
    Returns
    ----------
    Nothing
    '''

    if task == 'N2pc':

        if population == 'control':
            
            # Empty list to store the files
            control_files = []

            # Loop over the directories to access the control files
            # The controls are the subjects with a number below 50
            directories = glob.glob(os.path.join(input_dir, 'sub*'))
            for directory in directories:
                if int(directory[-2:]) < 51:
                    file = glob.glob(os.path.join(directory, 'cleaned_epochs', 'sub*N2pc.fif'))
                    control_files.append(file[0])

            # Concatenate the files
            epochs_list = []
            for file in control_files:
                epochs = mne.read_epochs(file)
                epochs_list.append(epochs)
            all_subj = mne.concatenate_epochs(epochs_list)
            if not os.path.exists(os.path.join(output_dir, f'{population}-allsubj')):
                os.makedirs(os.path.join(output_dir, f'{population}-allsubj'))
            all_subj.save(os.path.join(output_dir, f'{population}-allsubj', f'{population}-allsubj-{task}.fif'), overwrite=True) 
                
        elif population == 'stroke':

             # Empty list to store the files
            stroke_files = []

            # Loop over the directories to access the control files
            # The stroke patients are sub-51 and above
            directories = glob.glob(os.path.join(input_dir, 'sub*'))
            for directory in directories:
                if int(directory[-2:]) >= 51:
                    file = glob.glob(os.path.join(directory, 'cleaned_epochs', 'sub*N2pc.fif'))
                    stroke_files.append(file[0])

            # Concatenate the files
            epochs_list = []
            for file in control_files:
                epochs = mne.read_epochs(file)
                epochs_list.append(epochs)
            all_subj = mne.concatenate_epochs(epochs_list)
            if not os.path.exists(os.path.join(output_dir, f'{population}-allsubj')):
                os.makedirs(os.path.join(output_dir, f'{population}-allsubj'))
            all_subj.save(os.path.join(output_dir, f'{population}-allsubj', f'{population}-allsubj-{task}.fif'), overwrite=True)                  

    elif task == 'Alpheye':

        if population == 'control':
            
            # Empty list to store the files
            control_files = []

            # Loop over the directories to access the control files
            # The controls are the subjects with a number below 50
            directories = glob.glob(os.path.join(input_dir, 'sub*'))
            for directory in directories:
                if int(directory[-2:]) < 51:
                    file = glob.glob(os.path.join(directory, 'cleaned_epochs', 'sub*Alpheye.fif'))
                    control_files.append(file[0])

            # Concatenate the files
            epochs_list = []
            for file in control_files:
                epochs = mne.read_epochs(file)
                epochs_list.append(epochs)
            all_subj = mne.concatenate_epochs(epochs_list)
            if not os.path.exists(os.path.join(output_dir, f'{population}-allsubj')):
                os.makedirs(os.path.join(output_dir, f'{population}-allsubj'))
            all_subj.save(os.path.join(output_dir, f'{population}-allsubj', f'{population}-allsubj-{task}.fif'), overwrite=True) 
                
        elif population == 'stroke':

             # Empty list to store the files
            stroke_files = []

            # Loop over the directories to access the control files
            # The stroke patients are sub-51 and above
            directories = glob.glob(os.path.join(input_dir, 'sub*'))
            for directory in directories:
                if int(directory[-2:]) >= 51:
                    file = glob.glob(os.path.join(directory, 'cleaned_epochs', 'sub*Alpheye.fif'))
                    stroke_files.append(file[0])

            # Concatenate the files
            epochs_list = []
            for file in control_files:
                epochs = mne.read_epochs(file)
                epochs_list.append(epochs)
            all_subj = mne.concatenate_epochs(epochs_list)
            if not os.path.exists(os.path.join(output_dir, f'{population}-allsubj')):
                os.makedirs(os.path.join(output_dir, f'{population}-allsubj'))
            all_subj.save(os.path.join(output_dir, f'{population}-allsubj', f'{population}-allsubj-{task}.fif'), overwrite=True) 