import argparse
import n2pc_func.ERP as erp
from n2pc_func.set_paths import get_paths
from n2pc_func.set_subject_lists import get_subject_list

##############################################################################################################
# parameters to be changed by the user


input_dir, output_dir = get_paths()

full_subject_list = ['01', '02', '03', '04', '06', '07', '12', '13', '16', '17', '18', '19', '20', '21', '22', '23','70', '71', '72',
                      '73', '75', '76', '77', '78', '79', '80', '81', '82', '84', '85', '86', '87','52', '54', '55', '56', '58','51', '53', '59', '60']
##############################################################################################################

for subject_id in full_subject_list:
    try:
        erp.combine_evoked_single_subj(subject_id=subject_id, input_dir=input_dir, output_dir=output_dir)
    except:

        print(f'Error with subject {subject_id} during combine_evoked_single_subj')
        continue