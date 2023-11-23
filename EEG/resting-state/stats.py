from resting_func.conn_stats import get_nbs_inputs, nbs_bct_corr_z, nbs_report
from resting_func.set_paths import get_paths
import numpy as np

input_dir, output_dir = get_paths()

list_1 = [1, 2, 3, 4, 6, 7, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23]
list_2 =[51, 52, 53, 54, 55, 56, 58, 59, 60]
freqs = np.arange(8, 13)
metric='plv'
condition='RESTINGSTATEOPEN'
thresh=0.00001

pop_dict1 = {'subject_list': list_1,
                'freqs': freqs,
                'metric': metric,
                'condition': condition
}

pop_dict2 = {'subject_list': list_2,
                'freqs': freqs,
                'metric': metric,
                'condition': condition
}


# Get inputs
mat_list, y_vec = get_nbs_inputs(input_dir, pop_dict1, pop_dict2)

# Run NBS
pvals, adj, null = nbs_bct_corr_z(mat_list, thresh=thresh, y_vec=y_vec)

# Save results
name1 = 'old-alpha-open'
name2 = 'stroke-alpha-open'
nbs_report(pvals, adj, null, thresh, output_dir, name1, name2)

