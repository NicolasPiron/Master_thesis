import functions.ERP as ERP

# Path to data
input_dir = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/output_preproc'
# Where the output files are saved
output_dir = '/Users/nicolaspiron/Documents/PULSATION/Python_MNE/output_preproc'

# Subject list
subject_list = [1, 2, 3, 21]

for subject in subject_list:
    subject_id = str(subject)
    # n2pc waveforms
    ERP.plot_n2pc(subject_id=subject_id, input_dir=input_dir, output_dir=output_dir)
    # n2pc numerical values
    ERP.get_n2pc_values(subject_id, input_dir, output_dir)
    # n2pc topography

# Grand average