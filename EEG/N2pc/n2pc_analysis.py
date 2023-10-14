import argparse
import n2pc_func.ERP as erp
from set_paths import get_paths
from set_subject_lists import get_subject_list, get_excluded_subjects_list

##############################################################################################################
# This script allows to compute the n2pc waveforms and values for single subjects or for the grand average
# by typing the following command in the terminal : 'python n2pc_analysis.py single' or 'python n2pc_analysis.py GA'
# The user can choose to exclude subjects from the grand average

# Parameters to be changed by the user : 

# Path to data
input_dir, output_dir = get_paths()
# Population (control or stroke)
population = 'control'

# Subject list when analysing single subjects
subject_list = get_subject_list()

# List of subjects to be excluded from the grand average
excluded_subjects_list = get_excluded_subjects_list()
##############################################################################################################

def loop_over_subjects_n2pc(subject_list, input_dir, output_dir):
   '''Loop over subjects and compute n2pc

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
        try:
            # n2pc waveforms
            erp.plot_n2pc(subject_id=subject_id, input_dir=input_dir, output_dir=output_dir)
            erp.plot_n2pc_all_cond(subject_id=subject_id, input_dir=input_dir, output_dir=output_dir)
            # n2pc numerical values
            erp.get_n2pc_values(subject_id, input_dir, output_dir)
            print(f'================ Subject {subject_id} done ================')
        except:
            print(f'Error with subject {subject_id}')
            continue

def grand_average(input_dir, output_dir, exclude_subjects=False, excluded_subjects_list=[], population=None):
    '''Compute grand average n2pc
    
    Parameters
    ----------
    input_dir : str
        Path to data
    output_dir : str    
        Where the output files are saved
    exclude_subjects : bool
        Whether to exclude subjects from the grand average
    excluded_subjects_list : list
        List of subjects to be excluded from the grand average
    population : str
        Population (control or stroke)

    Returns
    -------
    None
    
    '''

    if exclude_subjects == True:

        erp.plot_n2pc('GA', input_dir, output_dir, exclude_subjects=True, excluded_subjects_list=excluded_subjects_list, population=population)
        erp.get_n2pc_values('GA', input_dir, output_dir, exclude_subjects=True, excluded_subjects_list=excluded_subjects_list, population=population)
        erp.plot_n2pc_all_cond('GA', input_dir, output_dir, exclude_subjects=True, excluded_subjects_list=excluded_subjects_list, population=population)
        print(f'================ Grand Average done (subjects {excluded_subjects_list} excluded) ================')
    else:

        erp.plot_n2pc('GA', input_dir, output_dir, exclude_subjects=False, excluded_subjects_list=[], population=population)
        erp.get_n2pc_values('GA', input_dir, output_dir, exclude_subjects=False, excluded_subjects_list=[], population=population)
        erp.plot_n2pc_all_cond('GA', input_dir, output_dir, exclude_subjects=False, excluded_subjects_list=[], population=population)
        print(f'================ Grand Average done (no subject excluded) ================')


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
            grand_average(input_dir, output_dir, exclude_subjects=True, excluded_subjects_list=excluded_subjects_list, population=population)
        if yes_or_no == 'no':
            grand_average(input_dir, output_dir, exclude_subjects=False, excluded_subjects_list=[], population=population)
        # Call the single subject function with the provided input
        
        print()
    elif args.mode == "single":

        loop_over_subjects_n2pc(subject_list, input_dir, output_dir)
        print('================ All subjects done ================')

    else:
        print("Invalid mode. Please choose 'single' or 'GA'.")
