import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from statannotations.Annotator import Annotator
from resting_func.set_paths import get_paths
# The goal of this script is to calculate for each individual how connectivity and power lateralization are related
# two dataframes are used : one for connectivity and one for power. 

def preproc_con(input_dir):
    con_df = pd.read_csv(os.path.join(input_dir, 'all_subj', 'resting-state', 'average_conn.csv'))
    # replace 'old' by 'healthy' for consistency
    con_df['group'] = con_df['group'].replace('old', 'healthy')
    print('---- \n', con_df.head(), '\n----')
    con_df = con_df.query('freq == "alpha" and eyes == "closed" and hemi == "diff"')
    con_df.rename(columns={'ciPLV': 'ciPLV_lateralization'}, inplace=True)
    return con_df

def preproc_pow(input_dir):
    # specific to alpha, eyes closed, sensor level, occipital regions
    pow_df = pd.read_csv(os.path.join(input_dir, 'all_subj', 'resting-state', 'last_rs_all_subj_power.csv'), sep=';')
    pow_df = pow_df.query('level == "sensor" and eyes == "closed"')
    pow_df = pow_df.query('roi == "occipital_l" or roi == "occipital_r"')
    pow_df = pow_df.drop(columns=['level', 'eyes', 'delta', 'theta', 'low_beta', 'high_beta'])
    pow_df['group'] = pow_df['group'].replace('old', 'healthy')
    # Pivot the dataframe
    print('---- \n', pow_df.head(), '\n----')
    pow_df_pivot = pow_df.pivot(index=["ID", "group"], columns="roi", values="alpha").reset_index()
    # Compute the difference
    pow_df_pivot["power_lateralization"] = pow_df_pivot["occipital_l"] - pow_df_pivot.get("occipital_r", 0)
    print('---- \n', pow_df_pivot.head(), '\n----')

    return pow_df_pivot

def get_merged_df(input_dir):
    con_df = preproc_con(input_dir)
    pow_df = preproc_pow(input_dir)
    df_merged = con_df.merge(pow_df, on=["ID", "group"], how="inner")
    return df_merged

def get_corr(df, group):
    if group is not None:
        df = df.query('group == @group')
    corr, p = stats.spearmanr(df["ciPLV_lateralization"], df["power_lateralization"])
    return corr, p


if __name__ == "__main__":

    i, o = get_paths()
    df = get_merged_df(i)

    for group in df["group"].unique():
        corr, p = get_corr(df, group)
        print(f"Group {group}: correlation = {corr}, p = {p}")
    corr, p = get_corr(df, None)
    print(f"Global correlation = {corr}, p = {p}")

    # sns.set_context("talk")
    # fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    # sns.scatterplot(data=df, x="ciPLV_lateralization", y="power_lateralization", hue="group", ax=ax, palette="Set2")
    # sns.despine()
    # plt.show()
    
    scaler = MinMaxScaler()
    df[["ciPLV_lat_norm", "power_lat_norm"]] = scaler.fit_transform(df[["ciPLV_lateralization", "power_lateralization"]])
    

    # Compute Ratio Index
    epsilon = 1e-10  # Small constant to prevent division by zero
    df["Ratio_Index"] = df["ciPLV_lat_norm"] / (df["power_lat_norm"] + epsilon)

    mean_ratio = df["Ratio_Index"].mean()
    std_ratio = df["Ratio_Index"].std()
    upper_thresh = mean_ratio + 3 * std_ratio
    lower_thresh = mean_ratio - 3 * std_ratio
    df = df[(df["Ratio_Index"] >= lower_thresh) & (df["Ratio_Index"] <= upper_thresh)]

    sns.set_context("talk")
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    sns.swarmplot(data=df, x="group", y="Ratio_Index", ax=ax)
    sns.boxplot(data=df, x="group", y="Ratio_Index", showcaps=False, boxprops={'facecolor':'None'}, showfliers=False, width=0.6, ax=ax)
    
    pairs=[("healthy", "pulvinar"), ("pulvinar", "thalamus"), ("healthy", "thalamus")]
    annotator = Annotator(ax, pairs, data=df, x='group', y='Ratio_Index')
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    annotator.apply_and_annotate()
    ax.set_ylabel("Ratio Index")
    ax.set_title("Laterality Ratio Index")
    sns.despine()
    fig.savefig(os.path.join(o, 'all_subj', 'resting-state', 'lateral_imbalance.png'), dpi=300)