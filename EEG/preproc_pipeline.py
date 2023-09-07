import functions.preproc as pp
import sys

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
    # Define task
    task = 'N2pc'
    # Path to data
    input_path = '/home/nicolasp/local_PULSATION/Master_thesis/EEG/toBIDS/BIDS_data/sourcedata'
    # Where the output files are saved
    output_path = '/home/nicolasp/shared_PULSATION/derivative'
    # Load data
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
