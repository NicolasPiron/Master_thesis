import sys
import os
import argparse

# Add the path to the functions to the system path
current_dir = os.path.join(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import functions.ERP as erp
from set_paths import get_paths

# Path to data
input_dir, output_dir = get_paths()
# Population (control or stroke)
population = 'control'

# Subject list when analysing single subjects
subject_list = [1]

# List of subjects to be excluded from the grand average
excluded_subjects_list = [4, 1, 2, 3]

##############################################################################################################

def loop_over_subjects_topo(subject_list, input_dir, output_dir):

   for subject in subject_list:
        subject_id = str(subject)
        if len(subject_id) == 1:
            subject_id = '0' + subject_id

        erp.plot_erp_topo(subject_id, input_dir, output_dir)
        erp.plot_spectral_topo(subject_id, input_dir, output_dir)
        print(f'================ Subject {subject_id} done ================')

def grand_average_topo(input_dir, output_dir, exclude_subjects=False, excluded_subjects_list=[], population=None):

    if exclude_subjects == True:

        erp.plot_erp_topo(subject_id='GA', input_dir=input_dir, output_dir=output_dir, exclude_subjects=True, excluded_subjects_list=excluded_subjects_list, population=population)
        erp.plot_spectral_topo(subject_id='GA', input_dir=input_dir, output_dir=output_dir, exclude_subjects=True, excluded_subjects_list=excluded_subjects_list, population=population)

    else:

        erp.plot_erp_topo(subject_id='GA', input_dir=input_dir, output_dir=output_dir, exclude_subjects=False, excluded_subjects_list=[], population=population)
        erp.plot_spectral_topo(subject_id='GA', input_dir=input_dir, output_dir=output_dir, exclude_subjects=False, excluded_subjects_list=[], population=population)



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
            grand_average_topo(input_dir, output_dir, exclude_subjects=True, excluded_subjects_list=excluded_subjects_list, population=population)
        if yes_or_no == 'no':
            grand_average_topo(input_dir, output_dir, exclude_subjects=False, excluded_subjects_list=[], population=population)
        # Call the single subject function with the provided input
        
        print()
    elif args.mode == "single":

        loop_over_subjects_topo(subject_list, input_dir, output_dir)
        print('================ All subjects done ================')

    else:
        print("Invalid mode. Please choose 'single' or 'GA'.")