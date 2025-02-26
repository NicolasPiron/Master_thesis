from n2pc_func.conn import get_all_subj_hemi_df
from n2pc_func.set_paths import get_paths
from n2pc_func.params import subject_list, swp_id
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statannotations.Annotator import Annotator

def asign_group(df):
    grp_mapping = {
        'healthy':list(range(1, 24)),
        'pulvinar':[51, 53, 59, 60],
        'thalamus':[52, 54, 55, 56, 57, 58],
        'young':list(range(70, 88))
    }
    id_to_group = {id_: group for group, ids in grp_mapping.items() for id_ in ids}
    df['group'] = df['ID'].map(id_to_group)
    return df

def get_relative_lat(hemi, target):
    lat = np.nan
    if target in ['left', 'right'] and hemi in ['left', 'right']:
        if hemi == target:
            lat = 'ipsi'
        else:
            lat = 'contra'    
    return lat

def preproc_df(df):
    df.fillna('mid', inplace=True)
    df = asign_group(df)
    df['lat'] = df.apply(lambda x: get_relative_lat(x['hemi'], x['target']), axis=1)
    return df

def get_mean_vals(df):
    df = df.groupby(['ID', 'target', 'lat', 'group'])['ciPLV'].mean().reset_index()
    return df

def plot_dist(df, pairs, title):
    sns.set_context('talk')
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    sns.boxplot(x='group', y='ciPLV', data=df, ax=ax, width=0.6, showcaps=False, boxprops={'facecolor':'None'}, showfliers=False)
    sns.swarmplot(x='group', y='ciPLV', data=df, size=5, ax=ax)  
    annotator = Annotator(ax, pairs, data=df, x="group", y="ciPLV")
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    annotator.apply_and_annotate()
    ax.set_title(title)
    sns.despine()
    plt.tight_layout()
    plt.show()
    return fig

def run_anova(df):
    model = smf.ols("ciPLV ~ C(group) * C(lat)", data=df).fit()
    table = sm.stats.anova_lm(model, typ=2)
    return table

i, o = get_paths()
df = get_all_subj_hemi_df(subject_list, swp_id, i)
df = preproc_df(df)
df.to_csv(os.path.join(o, 'all_subj', 'N2pc', 'sensor-connectivity', 'lat_hemi_df.csv'))
avg_df = get_mean_vals(df)
avg_df.to_csv(os.path.join(o, 'all_subj', 'N2pc', 'sensor-connectivity', 'avg_lat_hemi_df.csv'))

res_left = run_anova(avg_df[avg_df["target"] == "left"])
pd.DataFrame(res_left).to_csv(os.path.join(o, 'all_subj', 'N2pc', 'sensor-connectivity', 'anova_left.csv'))
res_right = run_anova(avg_df[avg_df["target"] == "right"])
pd.DataFrame(res_right).to_csv(os.path.join(o, 'all_subj', 'N2pc', 'sensor-connectivity', 'anova_right.csv'))

pairs=[("healthy", "pulvinar"), ("pulvinar", "thalamus"), ("healthy", "thalamus")]
fig1 = plot_dist(avg_df[(avg_df['target'] == 'left') & (avg_df['lat'] == 'contra')], pairs, 'target left contra')
fig1.savefig((os.path.join(o, 'all_subj', 'N2pc', 'sensor-connectivity', 'contra_tl.png')), dpi=300)
fig2 = plot_dist(avg_df[(avg_df['target'] == 'right') & (avg_df['lat'] == 'contra')], pairs, 'target right contra')
fig2.savefig((os.path.join(o, 'all_subj', 'N2pc', 'sensor-connectivity', 'contra_tr.png')), dpi=300)
fig3 = plot_dist(avg_df[(avg_df['target'] == 'left') & (avg_df['lat'] == 'ipsi')], pairs, 'target left ipsi')
fig3.savefig((os.path.join(o, 'all_subj', 'N2pc', 'sensor-connectivity', 'ipsi_tl.png')), dpi=300)
fig4 = plot_dist(avg_df[(avg_df['target'] == 'right') & (avg_df['lat'] == 'ipsi')], pairs, 'target right ipsi')
fig4.savefig((os.path.join(o, 'all_subj', 'N2pc', 'sensor-connectivity', 'ipsi_tr.png')), dpi=300)
# fig_avg = plot_dist(avg_df)
# fig_avg.savefig((os.path.join(o, 'all_subj', 'N2pc', 'sensor-connectivity', 'ciPLV_lateralization_avg.png')), dpi=300)

print('Left target')
print(res_left)
print('Right target')
print(res_right)