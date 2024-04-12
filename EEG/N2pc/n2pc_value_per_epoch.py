import os
import argparse
import pandas as pd
import n2pc_func.ERP as erp
from n2pc_func.set_paths import get_paths

########################################################################################################################
# Parameters to be changed by the user
input_dir, output_dir = get_paths()
########################################################################################################################

def loop_over_all_subj(input_dir, output_dir):
    
    subdirectories = []
    # Loop over the subdirectories, find the .csv files and append them to a list
    for root, dirs, files in os.walk(input_dir):
        for name in dirs:
            if name.startswith('sub-'):
                subdirectories.append(name)
    subdirectories.sort()

    # Loop over the subdirectories and create the dataframe for each subject
    for subj in subdirectories:
        # Compute n2pc values and save thems in a dataframe
        subject_id = subj[-2:]

        try:
            erp.get_df_n2pc_values_epoch(subject_id, input_dir, output_dir)
            print(f'==================== Dataframe created and saved for subject {subject_id}! :)')
        except:
            print(f"==================== No data (epochs or reject log) for subject {subject_id}! O_o'")
            continue


    df_list = []
    missing_subj = []
    for subj in subdirectories:
        if os.path.exists(os.path.join(input_dir, subj,'N2pc', 'n2pc-values', f'{subj}-n2pc_values_per_epoch.csv')):
            df_list.append(pd.read_csv((os.path.join(input_dir, subj,'N2pc', 'n2pc-values', f'{subj}-n2pc_values_per_epoch.csv'))))
        else:
            print(f"==================== No dataframe for subject {subj}! O_o'")
            missing_subj.append(subj)
    # Concatenate all dataframes in the list
    df = pd.concat(df_list)
    # Save dataframe as .csv file
    if not os.path.exists(os.path.join(output_dir, 'all_subj','N2pc', 'n2pc-values', 'n2pc-values-per-epoch')):
        os.makedirs(os.path.join(output_dir, 'all_subj','N2pc', 'n2pc-values', 'n2pc-values-per-epoch'))
    df.to_csv(os.path.join(output_dir, 'all_subj','N2pc', 'n2pc-values', 'n2pc-values-per-epoch', 'n2pc_values_per_epoch-allsubj.csv'), index=False)
    print(f'==================== Dataframe created and saved for all subjects! :) (except for {missing_subj})')


if __name__ == '__main__':

    #erp.get_df_n2pc_values_epoch('01', input_dir, output_dir)
    loop_over_all_subj(input_dir, output_dir)
