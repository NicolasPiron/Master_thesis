import sys
import os
import argparse

# Add the path to the functions to the system path
current_dir = os.path.join(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import functions.ERP as erp


# Path to data
input_dir = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/output_preproc'
# Task
task = 'N2pc'
# Subject list
subject_list = [1, 2, 3, 4]



def loop_for_evoked(subject_list, task, input_dir):

    for subject in subject_list:
        subject_id = str(subject)
        if len(subject_id) == 1:
            subject_id = '0' + subject_id
        erp.to_evoked(subject_id, task, input_dir)

if __name__ == '__main__':

    loop_for_evoked(subject_list, task, input_dir)