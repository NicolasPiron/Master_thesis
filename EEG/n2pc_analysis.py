import functions.ERP as erp
import sys
import argparse


##############################################################################################################

# Variables to be changed by the user : 

# Path to data
input_dir = '/home/nicolasp/shared_PULSATION/derivative'
# Where the output files are saved
output_dir = '/home/nicolasp/shared_PULSATION/derivative'
# Population (control or stroke)
population = 'control'

# Subject list
subject_list = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 51, 52, 53, 54, 55]

excluded_subjects_list = [9, 10, 15, 20]

##############################################################################################################

def loop_for_evoked(subject_list, input_dir, output_dir):

    for subject in subject_list:
        subject_id = str(subject)
        if len(subject_id) == 1:
            subject_id = '0' + subject_id
        erp.to_evoked(subject_id, input_dir, output_dir)


def loop_over_subjects_n2pc(subject_list, input_dir, output_dir):
    """
    Loop over subjects and compute n2pc
    :param subject_list: list of subjects
    :param input_dir: path to the input directory
    :param output_dir: path to the output directory
    :return:
    """

    for subject in subject_list:
        subject_id = str(subject)
        if len(subject_id) == 1:
            subject_id = '0' + subject_id
        # n2pc waveforms
        erp.plot_n2pc(subject_id=subject_id, input_dir=input_dir, output_dir=output_dir)
        # n2pc numerical values
        erp.get_n2pc_values(subject_id, input_dir, output_dir)

def grand_average(input_dir, output_dir, exclude_subjects=False, excluded_subjects_list=[], population=None):

    if exclude_subjects == True:

        erp.plot_n2pc('GA', input_dir, output_dir, exclude_subjects=True, excluded_subjects_list=excluded_subjects_list, population=population)

        erp.get_n2pc_values('GA', input_dir, output_dir, exclude_subjects=True, excluded_subjects_list=excluded_subjects_list, population=population)
    
    else:

        erp.plot_n2pc('GA', input_dir, output_dir, exclude_subjects=False, excluded_subjects_list=[], population=population)

        erp.get_n2pc_values('GA', input_dir, output_dir, exclude_subjects=False, excluded_subjects_list=[], population=population)


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
            grand_average(input_dir, output_dir, xclude_subjects=False, excluded_subjects_list=[], population=population)
        # Call the single subject function with the provided input
        
        print()
    elif args.mode == "single":

        loop_over_subjects_n2pc(subject_list, input_dir, output_dir)

    else:
        print("Invalid mode. Please choose 'single' or 'GA'.")
