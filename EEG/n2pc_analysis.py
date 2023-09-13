import functions.ERP as erp
import sys
import argparse


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


def grand_average(input_dir, output_dir, task, population, exclude_subjects=False, excluded_subjects_list=[]):
    
    if exclude_subjects:

        files = erp.get_population_files(input_dir, task, population)

        epochs, excluded_subjects_string = erp.get_population_epochs(files, exclude_subjects=exclude_subject, excluded_subjects_list=excluded_subjects_list)

        title = f'{population}-excluded_subjects-{excluded_subjects_string}'

    else:

        files = erp.get_population_files(input_dir, task, population)

        epochs = erp.get_population_epochs(files)

        title = f'{population}-all_subjects'

    erp.plot_n2pc('GA', input_dir, output_dir, epochs_=epochs, title=title)

    erp.get_n2pc_values('GA', input_dir, output_dir, epochs_=epochs, title=title)


if __name__ == '__main__':

    # Path to data
    input_dir = '/home/nicolasp/shared_PULSATION/derivative'
    # Where the output files are saved
    output_dir = '/home/nicolasp/shared_PULSATION/derivative'
    # Task, in this case always N2pc
    task = 'N2pc'
    # Population (control or stroke)
    population = 'control'
    # Subject list
    subject_list = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 51, 52, 53, 54, 55]

    excluded_subjects_list = [9, 10, 15, 20]

    exclude_subject = False

    # Create an ArgumentParser object to handle command-line arguments
    parser = argparse.ArgumentParser(description="Run specific functions based on user input")
    parser.add_argument("mode", choices=["single", "GA"], help="Choose 'single' for single subjects or 'GA' for Grand Average")
    # Parse the command-line arguments
    args = parser.parse_args()

    if args.mode == "GA":
        # Prompt the user to enter a two-digit number for the single subject
        yes_or_no = input("Do you want to exclude subjects? ('yes' or 'no'): ")
        if yes_or_no == 'yes':
            grand_average(input_dir, output_dir, task, population, exclude_subjects=True, excluded_subjects_list=excluded_subjects_list)
        if yes_or_no == 'no':
            grand_average(input_dir, output_dir, task, population)
        # Call the single subject function with the provided input
        
        print()
    elif args.mode == "single":

        loop_over_subjects_n2pc(subject_list, input_dir, output_dir)

    else:
        print("Invalid mode. Please choose 'single' or 'GA'.")
