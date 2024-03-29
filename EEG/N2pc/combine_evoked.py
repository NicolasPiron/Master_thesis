import argparse
import n2pc_func.ERP as erp
from n2pc_func.set_paths import get_paths
from n2pc_func.set_subject_lists import get_subject_list, get_excluded_subjects_list

##############################################################################################################
# parameters to be changed by the user

population = 'control'

input_dir, output_dir = get_paths()
# Subject list when analysing single subjects
subject_list = get_subject_list()
# List of subjects to be excluded from the grand average
excluded_subjects_list = get_excluded_subjects_list()
##############################################################################################################

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

        try:
            erp.combine_evoked(subject_id, input_dir, output_dir)
            print(f'================ Subject {subject_id} done ================')
        except:
            print(f'================ No data for subject {subject_id}! ================')
            continue

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