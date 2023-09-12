import glob
import os
import mne

def concat_all_subj(task, population, input_dir, output_dir, exclude_subject=False, exclude_subject_list=[]):
    ''' Concatenates and saves the files of all the subjects that have been preprocessed.
    
    Parameters
    ---------- 
    task : 'N2pc' or 'Alpheye'
        The name of the task
    population : 'control' or 'stroke'
        The type of population
    input_dir : str
        The path to the directory where the preprocessed files are stored
    output_dir : str
        The path to the directory where the concatenated files will be stored (should be a all_subj folder)
    exclude_subject : bool
        Whether or not to exclude some subjects from the concatenation
    exclude_subject_list : list
        The list of subjects to exclude from the concatenation
    
    Returns
    ----------
    Nothing
    '''

    if exclude_subject == True:

        # Transform the list of subjects to exclude into the format 'sub-xx'
        exclude_subject_list = [f'sub-{subject}' for subject in exclude_subject_list]

        transformed_list = []
        for item in exclude_subject_list:
            # Split the string into two parts: 'sub-' and the number
            parts = item.split('-')
            if len(parts) == 2 and len(parts[1]) == 1:
                # If the number has only one digit, add a '0' before it
                transformed_list.append(f'sub-0{parts[1]}')
            else:
                transformed_list.append(item)

        # Get the list of all the subjects and remove the excluded subjects
        subject_list = glob.glob(os.path.join(input_dir, 'sub*'))
        for subject in subject_list:
            if subject[-2:] in transformed_list:
                subject_list.remove(subject)
    elif exclude_subject == False:

        subject_list = glob.glob(os.path.join(input_dir, 'sub*'))

    if task == 'N2pc':

        if population == 'control':
            
            # Empty list to store the files
            control_files = []

            # Loop over the directories to access the control files
            # The controls are the subjects with a number below 50
            for subject in subject_list:
                if int(subject[-2:]) < 51:
                    file = glob.glob(os.path.join(subject, 'cleaned_epochs', 'sub*N2pc.fif'))
                    control_files.append(file[0])

            # Concatenate the files
            epochs_list = []
            for file in control_files:
                epochs = mne.read_epochs(file)
                # reset bad channels
                epochs.info['bads'] = []
                epochs_list.append(epochs)
            all_subj = mne.concatenate_epochs(epochs_list)
            if not os.path.exists(os.path.join(output_dir, f'{population}-allsubj')):
                os.makedirs(os.path.join(output_dir, f'{population}-allsubj'))
            # name the file depending on the excluded subjects
            if exclude_subject == True:
                all_subj.save(os.path.join(output_dir, f'{population}-allsubj', f'{population}-allsubj-{task}-excluded-{exclude_subject_list}.fif'), overwrite=True) 
            elif exclude_subject == False:
                all_subj.save(os.path.join(output_dir, f'{population}-allsubj', f'{population}-allsubj-{task}.fif'), overwrite=True) 
                
        elif population == 'stroke':

            # Empty list to store the files
            stroke_files = []

            # Loop over the directories to access the control files
            # The stroke patients are sub-51 and above
            for subject in subject_list:
                if int(subject[-2:]) >= 51:
                    file = glob.glob(os.path.join(subject, 'cleaned_epochs', 'sub*N2pc.fif'))
                    stroke_files.append(file[0])

             # Concatenate the files
            epochs_list = []
            for file in stroke_files:
                epochs = mne.read_epochs(file)
                # reset bad channels
                epochs.info['bads'] = []
                epochs_list.append(epochs)
            all_subj = mne.concatenate_epochs(epochs_list)
            if not os.path.exists(os.path.join(output_dir, f'{population}-allsubj')):
                os.makedirs(os.path.join(output_dir, f'{population}-allsubj'))
            # name the file depending on the excluded subjects
            if exclude_subject == True:
                all_subj.save(os.path.join(output_dir, f'{population}-allsubj', f'{population}-allsubj-{task}-excluded-{exclude_subject_list}.fif'), overwrite=True) 
            elif exclude_subject == False:
                all_subj.save(os.path.join(output_dir, f'{population}-allsubj', f'{population}-allsubj-{task}.fif'), overwrite=True) 

    elif task == 'Alpheye':

        if population == 'control':
            
            # Empty list to store the files
            control_files = []

            # Loop over the directories to access the control files
            # The controls are the subjects with a number below 50
            for subject in subject_list:
                if int(subject[-2:]) < 51:
                    file = glob.glob(os.path.join(subject, 'cleaned_epochs', 'sub*Alpheye.fif'))
                    control_files.append(file[0])

            # Concatenate the files
            epochs_list = []
            for file in control_files:
                epochs = mne.read_epochs(file)
                epochs.info['bads'] = []
                epochs_list.append(epochs)
            all_subj = mne.concatenate_epochs(epochs_list)
            if not os.path.exists(os.path.join(output_dir, f'{population}-allsubj')):
                os.makedirs(os.path.join(output_dir, f'{population}-allsubj'))

            if exclude_subject == True:
                all_subj.save(os.path.join(output_dir, f'{population}-allsubj', f'{population}-allsubj-{task}-excluded-{exclude_subject_list}.fif'), overwrite=True) 
            elif exclude_subject == False:
                all_subj.save(os.path.join(output_dir, f'{population}-allsubj', f'{population}-allsubj-{task}.fif'), overwrite=True) 
                
        elif population == 'stroke':

            # Empty list to store the files
            stroke_files = []

            # Loop over the directories to access the control files
            # The stroke patients are sub-51 and above
            for subject in subject_list:
                if int(subject[-2:]) >= 51:
                    file = glob.glob(os.path.join(subject, 'cleaned_epochs', 'sub*Alpheye.fif'))
                    stroke_files.append(file[0])

            # Concatenate the files
            epochs_list = []
            for file in stroke_files:
                epochs = mne.read_epochs(file)
                epochs.info['bads'] = []
                epochs_list.append(epochs)
            all_subj = mne.concatenate_epochs(epochs_list)
            if not os.path.exists(os.path.join(output_dir, f'{population}-allsubj')):
                os.makedirs(os.path.join(output_dir, f'{population}-allsubj'))

            if exclude_subject == True:
                all_subj.save(os.path.join(output_dir, f'{population}-allsubj', f'{population}-allsubj-{task}-excluded-{exclude_subject_list}.fif'), overwrite=True) 
            elif exclude_subject == False:
                all_subj.save(os.path.join(output_dir, f'{population}-allsubj', f'{population}-allsubj-{task}.fif'), overwrite=True) 