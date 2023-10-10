import sys
import os
import argparse

# Add the path to the functions to the system path
current_dir = os.path.join(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from functions import ERP as erp

# parameters to be changed by the user

input_dir = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/output_preproc'
# Where the output files are saved
output_dir = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/output_preproc'

population = 'control'

# Subject list when analysing single subjects
subject_list = [1]

# List of subjects to be excluded from the grand average
excluded_subjects_list = []

def loop_over_subjects(subject_list, input_dir, output_dir):
   '''Loop over subjects and plot the topography of alpha power

    Parameters
    ----------
    subject_list : list
        List of subjects to be analysed
    input_dir : str
        Path to data
    output_dir : str
        Where the output files are saved
    
    Returns
    -------
    None
   
    '''
   
   for subject in subject_list:
        subject_id = str(subject)
        if len(subject_id) == 1:
            subject_id = '0' + subject_id

        erp.combine_evoked(subject_id, input_dir, output_dir)
        print(f'================ Subject {subject_id} done ================')
       

if __name__ == '__main__':


    # Create an ArgumentParser object to handle command-line arguments
    parser = argparse.ArgumentParser(description="Run specific functions based on user input")
    parser.add_argument("mode", choices=["single", "GA"], help="Choose 'single' for single subjects or 'GA' for Grand Average")
    # Parse the command-line arguments
    args = parser.parse_args()

    if args.mode == "GA":
        # Prompt the user to enter a two-digit number for the single subject
        yes_or_no = input("Do you want to exclude subjects? ('yes' or 'no'): ")
        if yes_or_no == 'yes':
            erp.combine_evoked('GA', input_dir, output_dir, exclude_subjects=True, excluded_subjects_list=excluded_subjects_list, population=population)
        if yes_or_no == 'no':
            erp.combine_evoked('GA', input_dir, output_dir, exclude_subjects=False, excluded_subjects_list=[], population=population)
        # Call the single subject function with the provided input
        
        print()
    elif args.mode == "single":

        loop_over_subjects(subject_list, input_dir, output_dir)
        print('================ All subjects done ================')

    else:
        print("Invalid mode. Please choose 'single' or 'GA'.")