from n2pc_func.time_freq import *
from n2pc_func.set_paths import get_paths


i, o = get_paths()
population_dict = {'Healthy': ['01', '02', '03', '04', '06', '07', '12', '13', '16', '17', '18', '19', '20', '21', '22', '23'],
                    'young_control': ['70', '71', '72', '73', '75', '76', '77', '78', '79', '80', '81', '82', '84', '85', '86', '87'],
                    'Thalamus': ['52', '54', '55', '56', '58'],
                    'Pulvinar': ['51', '53', '59', '60']
}
comp_dict = {'Healthy': 'Pulvinar',
                'Thalamus': 'Pulvinar',
                'Healthy': 'Thalamus',
}

swp_id = ['52', '55', '56', '60']
ch_names = ['PO7', 'PO8']

if __name__ == '__main__':

    for ch in ch_names:
        for grpn1, grpn2 in comp_dict.items():
            sbj1 = population_dict[grpn1]
            sbj2 = population_dict[grpn2]
            run_f_test_tfr(sbj_list1=sbj1,
                grpn1=grpn1,
                sbj_list2=sbj2,
                grpn2=grpn2,
                ch_name=ch,
                swp_id=swp_id,
                input_dir=i,
            )