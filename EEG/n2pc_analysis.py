import functions.ERP as ERP

if __name__ == '__main__':

    # Path to data
    input_dir = '/home/nicolasp/shared_PULSATION/derivative'
    # Where the output files are saved
    output_dir = '/home/nicolasp/shared_PULSATION/derivative'

    # Subject list
    subject_list = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 51, 52, 53, 54, 55]

    for subject in subject_list:
        subject_id = str(subject)
        if len(subject_id) == 1:
            subject_id = '0' + subject_id
        # n2pc waveforms
        ERP.plot_n2pc(subject_id=subject_id, input_dir=input_dir, output_dir=output_dir)
        # n2pc numerical values
        ERP.get_n2pc_values(subject_id, input_dir, output_dir)
        # n2pc topography

    # Grand average
