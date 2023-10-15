import n2pc_func.HEOG as heog
from n2pc_func.set_paths import get_paths
from n2pc_func.set_subject_lists import get_subject_list

##############################################################################################################
# Parameters to be changed by the user

# Path to data
input_dir, output_dir = get_paths()
# Population (control or stroke)

# Subject list when analysing single subjects
subject_list = get_subject_list()
##############################################################################################################

def loop_over_subjects_heog(subject_list, input_dir, output_dir):

   for subject in subject_list:
        subject_id = str(subject)
        if len(subject_id) == 1:
            subject_id = '0' + subject_id
        try:
            heog.plot_heog_erp(subject_id, input_dir, output_dir)
            heog.rejection_report_heog_artifact(subject_id, input_dir, output_dir)
            heog.to_evoked(subject_id, input_dir)
            heog.reject_heog_artifacts(subject_id, input_dir, output_dir)
            heog.plot_n2pc_clean(subject_id, input_dir, output_dir)
            print(f'================ Subject {subject_id} done ================')
        except:
            print(f'================ Subject {subject_id} failed ================')
            continue


if __name__ == '__main__':

    loop_over_subjects_heog(subject_list, input_dir, output_dir)
    heog.report_heog_all_subj(input_dir, output_dir)