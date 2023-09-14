import functions.ERP as erp


# Path to data
input_dir = '/home/nicolasp/shared_PULSATION/derivative'
# Task
task = 'N2pc'
# Subject list
subject_list = [1, 2, 3, 4, 6, 7, 8, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 51, 52, 53, 54, 55, 56, 57, 58, 59]



def loop_for_evoked(subject_list, task, input_dir):

    for subject in subject_list:
        subject_id = str(subject)
        if len(subject_id) == 1:
            subject_id = '0' + subject_id
        erp.to_evoked(subject_id, task, input_dir)

if __name__ == '__main__':

    loop_for_evoked(subject_list, task, input_dir)