import n2pc_func.ERP as erp
from n2pc_func.set_paths import get_paths
from n2pc_func.set_subject_lists import get_subject_list

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