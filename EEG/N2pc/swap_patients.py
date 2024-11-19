from n2pc_func.ERP import *
from n2pc_func.swap_epo import *
from n2pc_func.set_paths import get_paths

i, o = get_paths()
left_lesion_patients = [52, 55, 56, 60]

for patient in left_lesion_patients:
    try:
        combine_swapped_evoked_patient(patient, i, o)
    except:
        print('Error with patient: ', patient, ' for evoked')
        continue
    try:
        main_swap_epo('01', i)
    except:
        print('Error with patient: ', patient, ' for epochs')
        continue