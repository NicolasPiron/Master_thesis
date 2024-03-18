import n2pc_func.ERP as erp
from n2pc_func.set_paths import get_paths
import os

i, _ = get_paths()

subject_list = sorted(os.listdir(i))
subject_list = [sub for sub in subject_list if sub.startswith('sub')]
subject_list = [sub[-2:] for sub in subject_list]

erp.create_long_n2pc_df(subject_list, i)