import n2pc_func.ERP as erp
from n2pc_func.set_paths import get_paths

i, o = get_paths()
right_lesion_patients = [51, 53, 54, 58, 59]

for patient in right_lesion_patients:
    try:
        erp.combine_swapped_evoked_patient(patient, i, o)
    except:
        print('Error with patient: ', patient)
        continue

