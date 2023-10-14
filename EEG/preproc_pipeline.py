import functions.preproc as pp
import sys
import os

##############################################################################################################
# Parameters to be changed by the user

# Define task
task = 'N2pc'
#task = 'Alpheye'
#task = 'RESTINGSTATECLOSE'
#task = 'RESTINGSTATEOPEN'
##############################################################################################################

# Path to data
script_dir = os.path.dirname(os.path.abspath(__file__))
if 'nicolaspiron/Documents' in script_dir:
    print('Running on Nicolas\'s Laptop')
    print('Paths set automatically')
    input_path = '/Users/nicolaspiron/Documents/Master_thesis/EEG/toBIDS/BIDS_data/sourcedata'
    output_path = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/output_preproc'

elif 'shared_PULSATION' in script_dir:
    print('Running on Lab-Calc server')
    print('Paths set automatically')
    input_path = '/home/nicolasp/local_PULSATION/Master_thesis/EEG/toBIDS/BIDS_data/sourcedata'
    output_path = '/home/nicolasp/shared_PULSATION/derivative'

def preproc_pipeline(subject_id):
    ''' Preprocessing pipeline for EEG data

    Parameters
    ----------
    subject_id : str or int
        Subject ID : 2 digits (e.g. '01' or '21')
    
    Returns
    -------
    None
    '''

    # Load data
    if task == 'RESTINGSTATECLOSE' or task == 'RESTINGSTATEOPEN':
        raw = pp.load_data(subject_id, task, input_path, plot_data=True)
        # Filter and interpolate
        raw = pp.filter_and_interpolate(subject_id=subject_id, task=task, raw=raw, output_path=output_path, plot_data=False)
        # Epoch data
        epochs = pp.epoch_data(subject_id=subject_id, task=task, raw=raw, e_list=None, output_path=output_path)

    elif task == 'N2pc' or task == 'Alpheye':
        raw, e_list = pp.load_data(subject_id, task, input_path, plot_data=True)
        # Filter and interpolate
        raw = pp.filter_and_interpolate(subject_id=subject_id, task=task, raw=raw, output_path=output_path, plot_data=False)
        # Epoch data
        epochs = pp.epoch_data(subject_id=subject_id, task=task, raw=raw, e_list=e_list, output_path=output_path)

    # Apply ICA and reject bad epochs
    epochs_clean = pp.automated_epochs_rejection(subject_id=subject_id, task=task, epochs=epochs, output_path=output_path)
    # Plot cleaned data
    pp.quality_check_plots(subject_id=subject_id, task=task, epochs=epochs, epochs_clean=epochs_clean, output_path=output_path)

    print('Preprocessing pipeline completed for subject ' + subject_id + '! :)')

if __name__ == "__main__":
    # Check if at least one command-line argument is provided
    if len(sys.argv) < 2:
        print("Usage: python preproc.py <subject_id>")
    else:
        # Get the first command-line argument
        subject_id = sys.argv[1]
        preproc_pipeline(subject_id)
