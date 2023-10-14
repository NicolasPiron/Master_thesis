from time_frequency_func import get_resting_power_df
from set_paths import get_paths

# This file computes the average of the epochs for each subject and each condition (eyes closed and eyes open)
# + for each cluster of electrodes (occip, frontal, parietal, total) and each frequency band (theta, alpha)

#####################################################################
# parameters to be changed by user
input_dir, output_dir = get_paths()

#####################################################################

if __name__ == '__main__':
    get_resting_power_df(input_dir, output_dir)