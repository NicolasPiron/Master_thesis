import sys
import os
import argparse
import pandas as pd

# Add the path to the functions to the system path
current_dir = os.path.join(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from functions.alpha import single_subj_alpha_epoch
from set_paths import get_paths


# Run this file to get the alpha power by side for each epochs. You can chose the participant,
# or run it for all participants. The output is a .csv file with the alpha power for each epoch.
# To get a single subject, write 'single' after the file name in the terminal, and then the subject ID when asked.
# To get all subjects, write 'all' after the file name in the terminal.

input_path, output_path = get_paths()

def get_df_all_subj():

    # WARNING : the paths are hard-coded, so you need to change them if you want to use this script
    subdirectories = []
    # Loop over the subdirectories, find the .csv files and append them to a list
    for root, dirs, files in os.walk(input_path):
        for name in dirs:
            if name.startswith('sub-'):
                subdirectories.append(name)
    
    # Loop over the subdirectories and create the dataframe for each subject
    for subj in subdirectories:
        # Compute alpha power and save it in a dataframe
        subject_id = subj[-2:]
        try:
            single_subj_alpha_epoch(subject_id, input_path, output_path)
            print(f'==================== Dataframe created and saved for subject {subject_id}! :)')
        except:
            print(f"==================== No data (epochs or reject log) for subject {subj}! O_o'")
            continue

    df_list = []
    missing_subj = []
    for subj in subdirectories:
        if os.path.exists(os.path.join(input_path, subj, 'N2pc', 'alpha-power-df', f'{subj}-alpha-power-per-epoch.csv')):
            df_list.append(pd.read_csv((os.path.join(input_path, subj, 'N2pc', 'alpha-power-df', f'{subj}-alpha-power-per-epoch.csv'))))
        else:
            print(f"==================== No alpha power dataframe for subject {subj}! O_o'")
            missing_subj.append(subj)
    # Concatenate all dataframes in the list
    df = pd.concat(df_list)
    # Save dataframe as .csv file
    if not os.path.exists(os.path.join(output_path,'all_subj','N2pc', 'alpha-power-allsubj')):
        os.makedirs(os.path.join(output_path,'all_subj','N2pc', 'alpha-power-allsubj'))
    df.to_csv(os.path.join(output_path,'all_subj','N2pc', 'alpha-power-allsubj', 'alpha-power-per-epoch-allsubj.csv'), index=False)
    print(f'==================== Dataframe created and saved for all subjects! :) (except for {missing_subj})')

if __name__ == "__main__":
    
    # Create an ArgumentParser object to handle command-line arguments
    parser = argparse.ArgumentParser(description="Run specific functions based on user input")
    # Add an argument to specify the mode (single or all)
    parser.add_argument("mode", choices=["single", "all"], help="Choose 'single' for single subject or 'all' for all subjects")
    # Parse the command-line arguments
    args = parser.parse_args()

    # Check the value of the 'mode' argument and run the corresponding function
    if args.mode == "single":
        # Prompt the user to enter a two-digit number for the single subject
        subject_id = input("Enter a two-digit number for the single subject: ")
        # Call the single subject function with the provided input
        single_subj_alpha_epoch(subject_id, input_path, output_path)
        print(f'==================== Dataframe created and saved for subject {subject_id}! :)')
    elif args.mode == "all":
        # Call the all subjects function
        get_df_all_subj()
    else:
        print("Invalid mode. Please choose 'single' or 'all'.")

