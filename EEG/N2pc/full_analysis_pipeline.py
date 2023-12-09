from n2pc_func.set_paths import get_paths
import n2pc_func.ERP  as erp
import n2pc_func.HEOG as heog
import n2pc_func.subject_rejection as subject_rejection
from n2pc_func.alpha import get_power_df_all_subj
import os
import pandas as pd

##############################################################################################################
input_dir, output_dir = get_paths()

population_dict = {'old_control': ['01', '02', '03', '04', '06', '07', '12', '13', '16', '17', '18', '19', '20', '21', '22', '23'],
                    'young_control': ['70', '71', '72', '73', '75', '76', '77', '78', '79', '80', '81', '82', '84', '85', '86', '87'],
                    'thal_control': ['52', '54', '55', '56', '58'],
                    'pulvinar': ['51', '53', '59', '60']

}
full_subject_list = ['01', '02', '03', '04', '06', '07', '12', '13', '16', '17', '18', '19', '20', '21', '22', '23','70', '71', '72',
                      '73', '75', '76', '77', '78', '79', '80', '81', '82', '84', '85', '86', '87','52', '54', '55', '56', '58','51', '53', '59', '60']
##############################################################################################################

# Loop over subjects and compute n2pc -> plot n2pc waveforms, topomaps and get values

#for subject_id in full_subject_list:
#    try:
#        erp.to_evoked(subject_id=subject_id, input_dir=input_dir)
#    except:
#        print(f'Error with subject {subject_id} during to_evoked')
#        continue
#    try:
#        erp.combine_evoked_single_subj(subject_id=subject_id, input_dir=input_dir, output_dir=output_dir)
#    except:
#        print(f'Error with subject {subject_id} during combine_evoked_single_subj')
#        continue
#    try:
#        erp.plot_n2pc_single_subj(subject_id=subject_id, input_dir=input_dir, output_dir=output_dir)
#   except:
#        print(f'Error with subject {subject_id} during plot_n2pc_single_subj')
#        pass
#    try:
#        erp.plot_erp_topo_single_subj(subject_id=subject_id, input_dir=input_dir, output_dir=output_dir)
#    except:
#        print(f'Error with subject {subject_id} during plot_erp_topo_single_subj')
#        pass
#    try:
#        erp.plot_spectral_topo_single_subj(subject_id=subject_id, input_dir=input_dir, output_dir=output_dir)
#    except:
#        print(f'Error with subject {subject_id} during plot_spectral_topo_single_subj')
#        pass
#    try:
#        erp.plot_n2pc_all_cond_single_subj(subject_id=subject_id, input_dir=input_dir, output_dir=output_dir)
#    except:
#        print(f'Error with subject {subject_id} during plot_n2pc_all_cond_single_subj')
#        pass
#    try:
#        erp.get_n2pc_values_single_subj(subject_id=subject_id, input_dir=input_dir, output_dir=output_dir)
#    except:
#        print(f'Error with subject {subject_id} during get_n2pc_values_single_subj')
#        pass
#    try:
#        erp.get_peak_latency_single_subj(subject_id=subject_id, input_dir=input_dir, output_dir=output_dir)
#    except:
#        print(f'Error with subject {subject_id} during get_peak_latency_single_subj')
#        pass

for population, subject_list in population_dict.items():
    try:
        erp.combine_evoked_population(input_dir=input_dir, output_dir=output_dir, subject_list=subject_list, population=population)
    except:
        print(f'Error with population {population} during combine_evoked_pop')
        continue
    try:
        erp.plot_spectral_topo_population(input_dir=input_dir, output_dir=output_dir, population=population)
    except:
        print(f'Error with population {population} during plot_spectral_topo_pop')
        pass
    try:
        erp.plot_erp_topo_population(input_dir=input_dir, output_dir=output_dir, population=population)
    except:
        print(f'Error with population {population} during plot_erp_topo_pop')
        pass
    try:
        erp.plot_n2pc_population(input_dir=input_dir, output_dir=output_dir, subject_list=subject_list, population=population)
    except:
        print(f'Error with population {population} during plot_n2pc_pop')
        pass
    try:
        erp.plot_n2pc_all_cond_population(input_dir=input_dir, output_dir=output_dir, subject_list=subject_list, population=population)
    except:
        print(f'Error with population {population} during plot_n2pc_all_cond_pop')
        pass
    try:
        erp.get_n2pc_values_population(input_dir=input_dir, output_dir=output_dir, subject_list=subject_list, population=population)
    except:
        print(f'Error with population {population} during get_n2pc_values_pop')
        pass
    try:
        erp.get_peak_latency_grand_average(input_dir=input_dir, output_dir=output_dir, population=population)
    except:
        print(f'Error with population {population} during get_peak_latency_grand_average')
        pass
    try:
        erp.get_peak_latency_grand_average(input_dir=input_dir, output_dir=output_dir, population=population)
    except:
        print(f'Error with population {population} during get_peak_latency_population')
        pass

try:
    erp.all_peak_latencies_report(input_dir=input_dir, output_dir=output_dir)
except:
    print(f'Error with all_peak_latencies_report')
    pass

