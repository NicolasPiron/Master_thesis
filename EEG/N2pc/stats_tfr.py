from n2pc_func.time_freq import run_f_test_latdiff, run_f_test_tfr, viz_mean_alpha_power
from n2pc_func.set_paths import get_paths

i, o = get_paths()

population_dict = {
    'Healthy': ['01', '02', '03', '04', '06', '07', '12', '13', '16', '17', '18', '19', '20', '21', '22', '23'],
#    'young_control': ['70', '71', '72', '73', '75', '76', '77', '78', '79', '80', '81', '82', '84', '85', '86', '87'],
    'Thalamus': ['52', '54', '55', '56', '58'],
    'Pulvinar': ['51', '53', '59', '60']
}

comp_list = [
#    ['Healthy', 'young_control'],
    ['Healthy', 'Pulvinar'],
    ['Healthy', 'Thalamus'],
    ['Thalamus', 'Pulvinar'],
    ['Healthy', 'Pulvinar', 'Thalamus'],
]

# population_dict = {
#     'test': ['01', '02'],
#     'test2': ['03', '03'],
# }
# comp_list = [['test', 'test2'],]

# thresh_list = [2, 3, 4, 5, 6, 7, 8]
swp_id = ['52', '55', '56', '60']
ch_names = ['PO7', 'PO8']

if __name__ == '__main__':
    for comp in comp_list:
        viz_mean_alpha_power(
            population_dict=population_dict,
            comp=comp,
            swp_id=swp_id,
            input_dir=i,
        )

    # # contra-ipsi
    # for grpn1, grpn2 in comp_list:
    #     sbj1 = population_dict[grpn1]
    #     sbj2 = population_dict[grpn2]
    #     run_f_test_latdiff(
    #         sbj_list1=sbj1,
    #         grpn1=grpn1,
    #         sbj_list2=sbj2,
    #         grpn2=grpn2,
    #         swp_id=swp_id,
    #         thresh=None,
    #         crop=False,
    #         input_dir=i,
    #     )
    # # just PO7 and PO8
    # for grpn1, grpn2 in comp_list:
    #     sbj1 = population_dict[grpn1]
    #     sbj2 = population_dict[grpn2]
    #     for ch in ch_names:
    #         run_f_test_tfr(
    #             sbj_list1=sbj1,
    #             grpn1=grpn1,
    #             sbj_list2=sbj2,
    #             grpn2=grpn2,
    #             ch_name=ch,
    #             swp_id=swp_id,
    #             thresh=None,
    #             crop=False,
    #             input_dir=i,
    #         )