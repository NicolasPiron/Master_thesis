from n2pc_func.time_freq import *
from n2pc_func.set_paths import get_paths

i, o = get_paths()
population_dict = {'old_control': ['01', '02', '03', '04', '06', '07', '12', '13', '16', '17', '18', '19', '20', '21', '22', '23'],
                    'young_control': ['70', '71', '72', '73', '75', '76', '77', '78', '79', '80', '81', '82', '84', '85', '86', '87'],
                    'thal_control': ['52', '54', '55', '56', '58'],
                    'pulvinar': ['51', '53', '59', '60']
}
swp_id = ['52', '55', '56', '60']
ch_names = ['PO7', 'PO8']

run_f_test_tfr(population_dict['old_control'],
    population_dict['young_control'],
    ch_name='PO7',
    swp_id=swp_id,
    input_dir=i,
)