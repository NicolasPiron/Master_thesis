from n2pc_func.time_freq import viz_mean_alpha_power
from n2pc_func.set_paths import get_paths
import numpy as np
import matplotlib.pyplot as plt

i, o = get_paths()
population_dict = {
    'test': ['01', '02'],
    'test2': ['03', '03'],
}

viz_mean_alpha_power(
    population_dict=population_dict,
    swp_id=[],
    input_dir=i,
)
