import sys
import os
import argparse

# Add the path to the functions to the system path
current_dir = os.path.join(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import functions.ERP as erp
from set_paths import get_paths
from set_subject_lists import get_subject_list

##############################################################################################################
# Path to data
input_dir, _ = get_paths()
# Task
task = 'N2pc'
# Subject list
subject_list = get_subject_list()
##############################################################################################################



def loop_for_evoked(subject_list, task, input_dir):

    for subject in subject_list:
        subject_id = str(subject)
        if len(subject_id) == 1:
            subject_id = '0' + subject_id
        try:
            erp.to_evoked(subject_id, task, input_dir)
        except:
            print('Error with subject ' + subject_id)
            continue

if __name__ == '__main__':

    loop_for_evoked(subject_list, task, input_dir)