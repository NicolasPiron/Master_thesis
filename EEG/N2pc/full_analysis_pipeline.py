from n2pc_func.set_paths import get_paths
from n2pc_func.set_subject_lists import get_subject_list, get_excluded_subjects_list
import n2pc_func.ERP  as erp
import n2pc_func.HEOG as heog
import n2pc_func.subject_rejection as subject_rejection
from n2pc_func.alpha import get_power_df_all_subj
import os
import pandas as pd


input_dir, output_dir = get_paths()
# Subject list when analysing single subjects
subject_list = get_subject_list()
# List of subjects to be excluded from the grand average
excluded_subjects_list = get_excluded_subjects_list()

task = 'N2pc'

non_pulvinar_stroke_subjects = [52, 54, 55, 56, 57, 58]

pulvinar_stroke_subjects = [51, 53, 59, 60]

# 1st step: create evoked files

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

loop_for_evoked(subject_list, task, input_dir)

# 2nd step: combine evoked files

def loop_over_subjects(subject_list, input_dir, output_dir):
   
   for subject in subject_list:
        subject_id = str(subject)
        if len(subject_id) == 1:
            subject_id = '0' + subject_id

        try:
            erp.combine_evoked(subject_id, input_dir, output_dir)
            print(f'================ Subject {subject_id} done ================')
        except:
            print(f'================ No data for subject {subject_id}! ================')
            continue

    # 2.1 combine the evoked files for all individual subjects

loop_over_subjects(subject_list, input_dir, output_dir)
print('================ All subjects done ================')

    # 2.2: combine Grand average evoked files for the old control group

erp.combine_evoked('GA', input_dir, output_dir, exclude_subjects=True, excluded_subjects_list=excluded_subjects_list, population='old_control')

    # 2.3: combine Grand average evoked files for the young control group

erp.combine_evoked('GA', input_dir, output_dir, exclude_subjects=True, excluded_subjects_list=excluded_subjects_list, population='young_control')

    # 2.4: combine Grand average evoked files for the stroke group

        # 2.4.1: combine Grand average evoked files for the pulvinar stroke group (exclude other stroke subjects)

excluded_subjects_list = non_pulvinar_stroke_subjects
erp.combine_evoked('GA', input_dir, output_dir, exclude_subjects=True, excluded_subjects_list=excluded_subjects_list, population='stroke')

        # 2.4.2: combine Grand average evoked files for the non-pulvinar stroke group (exclude pulvinar subjects)

excluded_subjects_list = pulvinar_stroke_subjects
erp.combine_evoked('GA', input_dir, output_dir, exclude_subjects=True, excluded_subjects_list=excluded_subjects_list, population='stroke')

# reset the excluded_subjects_list
excluded_subjects_list = get_excluded_subjects_list()

# 3rd step: plot HEOG artifacts

def loop_over_subjects_heog(subject_list, input_dir, output_dir):

   for subject in subject_list:
        subject_id = str(subject)
        if len(subject_id) == 1:
            subject_id = '0' + subject_id
        try:
            heog.plot_heog_erp(subject_id, input_dir, output_dir)
            print(f'================ Subject {subject_id} done -plot_heog_erp- ================')
        except:
            print(f'================ Subject {subject_id} failed -plot_heog_erp- ================')
            continue
        try:
            heog.rejection_report_heog_artifact(subject_id, input_dir, output_dir)
            print(f'================ Subject {subject_id} done -rejection_report_heog_artifact- ================')
        except:
            print(f'================ Subject {subject_id} failed -rejection_report_heog_artifact- ================')
            continue
        try:
            heog.reject_heog_artifacts(subject_id, input_dir, output_dir)
            print(f'================ Subject {subject_id} done -reject_heog_artifacts- ================')
        except:
            print(f'================ Subject {subject_id} failed -reject_heog_artifacts- ================')
            continue
        try:
            heog.to_evoked(subject_id, input_dir)
            print(f'================ Subject {subject_id} done -to_evoked- ================')
        except:
            print(f'================ Subject {subject_id} failed -to_evoked- ================')
            continue
        try:
            heog.plot_n2pc_clean(subject_id, input_dir, output_dir)
            print(f'================ Subject {subject_id} done -plot_n2pc_clean- ================')
        except:
            print(f'================ Subject {subject_id} failed -plot_n2pc_clean- ================')
            continue

loop_over_subjects_heog(subject_list, input_dir, output_dir)
heog.report_heog_all_subj(input_dir, output_dir)

# 4th step: get the subjects rejection report

for subject_id in subject_list:
        subject_rejection.plot_rejection_proportion(subject_id, input_dir, output_dir)
subject_rejection.get_rejected_trials_proportion_all_subj(input_dir, output_dir)

# 5th step: plot the peak of the N2pc

erp.get_peak_latency_grand_average(input_dir, output_dir, excluded_subjects_list=excluded_subjects_list)

# 6th step: plot the waveform of the N2pc

def loop_over_subjects_n2pc(subject_list, input_dir, output_dir):

   for subject in subject_list:
        subject_id = str(subject)
        if len(subject_id) == 1:
            subject_id = '0' + subject_id
        try:
            # n2pc waveforms
            erp.plot_n2pc(subject_id=subject_id, input_dir=input_dir, output_dir=output_dir)
            erp.plot_n2pc_all_cond(subject_id=subject_id, input_dir=input_dir, output_dir=output_dir)
            # n2pc numerical values
            erp.get_n2pc_values(subject_id, input_dir, output_dir)
            print(f'================ Subject {subject_id} done ================')
        except:
            print(f'Error with subject {subject_id}')
            continue

    # 6.1: plot the waveform of the N2pc for all individual subjects

loop_over_subjects_n2pc(subject_list, input_dir, output_dir)

    # 6.2: plot the waveform of the N2pc for the old control group

def grand_average(input_dir, output_dir, excluded_subjects_list=[], population=None):

    erp.plot_n2pc('GA', input_dir, output_dir, exclude_subjects=True, excluded_subjects_list=excluded_subjects_list, population=population)
    erp.get_n2pc_values('GA', input_dir, output_dir, exclude_subjects=True, excluded_subjects_list=excluded_subjects_list, population=population)
    erp.plot_n2pc_all_cond('GA', input_dir, output_dir, exclude_subjects=True, excluded_subjects_list=excluded_subjects_list, population=population)
    print(f'================ Grand Average done (subjects {excluded_subjects_list} excluded) ================')

grand_average(input_dir, output_dir, excluded_subjects_list=excluded_subjects_list, population='old_control')

    # 6.3: plot the waveform of the N2pc for the young control group

grand_average(input_dir, output_dir, excluded_subjects_list=excluded_subjects_list, population='young_control')

    # 6.4: plot the waveform of the N2pc for the stroke group

        # 6.4.1: plot the waveform of the N2pc for the pulvinar stroke group (exclude other stroke subjects)

excluded_subjects_list = non_pulvinar_stroke_subjects
grand_average(input_dir, output_dir, excluded_subjects_list=excluded_subjects_list, population='stroke')

        # 6.4.2: plot the waveform of the N2pc for the non-pulvinar stroke group (exclude pulvinar subjects)

excluded_subjects_list = pulvinar_stroke_subjects
grand_average(input_dir, output_dir, excluded_subjects_list=excluded_subjects_list, population='stroke')

# reset the excluded_subjects_list
excluded_subjects_list = get_excluded_subjects_list()

# 7th step: plot the topography of the N2pc

def loop_over_subjects_topo(subject_list, input_dir, output_dir):

   for subject in subject_list:
        subject_id = str(subject)
        if len(subject_id) == 1:
            subject_id = '0' + subject_id
        try:
            erp.plot_erp_topo(subject_id, input_dir, output_dir)
            erp.plot_spectral_topo(subject_id, input_dir, output_dir)
            print(f'================ Subject {subject_id} done ================')
        except:
            print(f'================ Subject {subject_id} failed ================')
            continue
def grand_average_topo(input_dir, output_dir, excluded_subjects_list=[], population=None):

        try:
            erp.plot_erp_topo(subject_id='GA', input_dir=input_dir, output_dir=output_dir, exclude_subjects=True, excluded_subjects_list=excluded_subjects_list, population=population)
            erp.plot_spectral_topo(subject_id='GA', input_dir=input_dir, output_dir=output_dir, exclude_subjects=True, excluded_subjects_list=excluded_subjects_list, population=population)
            print('================ GA done! ================')
        except:
            print('================ GA failed! ================')


    # 7.1: plot the topography of the N2pc for all individual subjects

loop_over_subjects_topo(subject_list, input_dir, output_dir)

    # 7.2: plot the topography of the N2pc for the old control group

grand_average_topo(input_dir, output_dir, excluded_subjects_list=excluded_subjects_list, population='old_control')

    # 7.3: plot the topography of the N2pc for the young control group

grand_average_topo(input_dir, output_dir, excluded_subjects_list=excluded_subjects_list, population='young_control')

    # 7.4: plot the topography of the N2pc for the stroke group

        # 7.4.1: plot the topography of the N2pc for the pulvinar stroke group (exclude other stroke subjects)
    
excluded_subjects_list = non_pulvinar_stroke_subjects
grand_average_topo(input_dir, output_dir, excluded_subjects_list=excluded_subjects_list, population='stroke')

        # 7.4.2: plot the topography of the N2pc for the non-pulvinar stroke group (exclude pulvinar subjects)

excluded_subjects_list = pulvinar_stroke_subjects
grand_average_topo(input_dir, output_dir, excluded_subjects_list=excluded_subjects_list, population='stroke')

# reset the excluded_subjects_list
excluded_subjects_list = get_excluded_subjects_list()

# 8th step: create a dataframe with the n2pc values per epoch

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

loop_over_all_subj(input_dir, output_dir)

# 9th step: create a dataframe with the frequencies of each elec per epoch

get_power_df_all_subj(input_dir, output_dir)
print(f'==================== Dataframe created and saved for all subjects! :)')

print('=========================================================================')
print('==================== ALL DONE! :) =======================================')
print('=========================================================================')