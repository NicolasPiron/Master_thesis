import n2pc_func.ERP as erp
from n2pc_func.set_paths import get_paths

i, o = get_paths()
left_lesion_patients = [52, 55, 56, 60]

for patient in left_lesion_patients:
    try:
        erp.combine_swapped_evoked_patient(patient, i, o)
    except:
        print('Error with patient: ', patient)
        continue

