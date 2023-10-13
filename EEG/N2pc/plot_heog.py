import sys
import os

current_dir = os.path.join(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import functions.HEOG as heog

##############################################################################################################
# Parameters to be changed by the user

# Path to data
input_dir = ''
# Where the output files are saved
output_dir = ''
# Population (control or stroke)

# Subject list when analysing single subjects
subject_list = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 51, 52, 53, 54, 55, 56, 57, 58, 59]

##############################################################################################################

def loop_over_subjects_topo(subject_list, input_dir, output_dir):

   for subject in subject_list:
        subject_id = str(subject)
        if len(subject_id) == 1:
            subject_id = '0' + subject_id
        try:
            heog.plot_heog_erp(subject_id, input_dir, output_dir)
            print(f'================ Subject {subject_id} done ================')
        except:
            print(f'================ Subject {subject_id} failed ================')
            continue


if __name__ == '__main__':

    loop_over_subjects_topo(subject_list, input_dir, output_dir)