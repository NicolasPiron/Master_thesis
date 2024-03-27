from __future__ import division
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from bct.algorithms import get_components

######################
# The NBS functinon is stolen from https://github.com/GidLev/NBS-correlation
######################

def nbs_bct_corr_z(corr_arr, thresh, y_vec, k=1000, extent=True, verbose=False):

    '''
    Performs the NBS for matrices [corr_arr] and vector [y_vec]  for a Pearson's r-statistic threshold of
    [thresh].

    Parameters
    ----------
    corr_arr : NxNxP np.ndarray
        matrix representing the correlation matrices population with P subjects. must be
        symmetric.

    y_vec : 1xP vector representing the behavioral/physiological values to correlate against

    thresh : float
        minimum Pearson's r-value used as threshold
    k : int
        number of permutations used to estimate the empirical null
        distribution, recommended - 10000
    verbose : bool
        print some extra information each iteration. defaults value = False

    Returns
    -------
    pval : Cx1 np.ndarray
        A vector of corrected p-values for each component of the networks
        identified. If at least one p-value is less than thres, the omnibus
        null hypothesis can be rejected at alpha significance. The null
        hypothesis is that the value of the connectivity from each edge has
        equal mean across the two populations.
    adj : IxIxC np.ndarray
        an adjacency matrix identifying the edges comprising each component.
        edges are assigned indexed values.
    null : Kx1 np.ndarray
        A vector of K sampled from the null distribution of maximal component
        size.

    Notes
    -----
    ALGORITHM DESCRIPTION
    The NBS is a nonparametric statistical test used to isolate the
    components of an N x N undirected connectivity matrix that differ
    significantly between two distinct populations. Each element of the
    connectivity matrix stores a connectivity value and each member of
    the two populations possesses a distinct connectivity matrix. A
    component of a connectivity matrix is defined as a set of
    interconnected edges.

    The NBS is essentially a procedure to control the family-wise error
    rate, in the weak sense, when the null hypothesis is tested
    independently at each of the N(N-1)/2 edges comprising the undirected
    connectivity matrix. The NBS can provide greater statistical power
    than conventional procedures for controlling the family-wise error
    rate, such as the false discovery rate, if the set of edges at which
    the null hypothesis is rejected constitues a large component or
    components.

    The NBS comprises fours steps:
    1. Perform a Pearson r test at each edge indepedently to test the
       hypothesis that the value of connectivity between each edge and an
       external variable, corelates across all nodes.
    2. Threshold the Pearson r-statistic available at each edge to form a set of
       suprathreshold edges.
    3. Identify any components in the adjacency matrix defined by the set
       of suprathreshold edges. These are referred to as observed
       components. Compute the size of each observed component
       identified; that is, the number of edges it comprises.
    4. Repeat K times steps 1-3, each time randomly permuting the extarnal
       variable vector and storing the size of the largest component
       identified for each permutation. This yields an empirical estimate
       of the null distribution of maximal component size. A corrected
       p-value for each observed component is then calculated using this
       null distribution.

    [1] Zalesky A, Fornito A, Bullmore ET (2010) Network-based statistic:
        Identifying differences in brain networks. NeuroImage.
        10.1016/j.neuroimage.2010.06.041

     Adopted from the python implementation of the BCT - https://sites.google.com/site/bctnet/, https://pypi.org/project/bctpy/
     Credit for implementing the vectorized version of the code to Gideon Rosenthal

    '''

    def corr_with_vars(x, y):
        # check correlation X -> M (Sobel's test)
        r, _ = stats.pearsonr(x, y)
        z = 0.5 * np.log((1 + r)/(1 - r))
        return z.item(0)

    ix, jx, nx = corr_arr.shape
    ny, = y_vec.shape

    if not ix == jx:
        raise ValueError('Matrices are not symmetrical')
    else:
        n = ix

    if nx != ny:
        raise ValueError('The [y_vec dimension must match the [corr_arr] third dimension')

    # only consider upper triangular edges
    ixes = np.where(np.triu(np.ones((n, n)), 1))

    # number of edges
    m = np.size(ixes, axis=1)

    # vectorize connectivity matrices for speed
    xmat = np.zeros((m, nx))

    for i in range(nx):
        xmat[:, i] = corr_arr[:, :, i][ixes].squeeze()
    del corr_arr

    # perform pearson corr test at each edge

    z_stat = np.apply_along_axis(corr_with_vars, 1, xmat, y_vec)
    print('z_stat: ', z_stat)

    # threshold
    ind_r, = np.where(z_stat > thresh)

    if len(ind_r) == 0:
        raise ValueError("Unsuitable threshold")

    # suprathreshold adjacency matrix
    adj = np.zeros((n, n))
    adjT = np.zeros((n, n))

    if extent:
        adj[(ixes[0][ind_r], ixes[1][ind_r])] = 1
        adj = adj + adj.T  # make symmetrical
    else:
        adj[(ixes[0][ind_r], ixes[1][ind_r])] = 1
        adj = adj + adj.T  # make symmetrical
        adjT[(ixes[0], ixes[1])] = z_stat
        adjT = adjT + adjT.T  # make symmetrical
        adjT[adjT <= thresh] = 0

    a, sz = get_components(adj)

    # convert size from nodes to number of edges
    # only consider components comprising more than one node (e.g. a/l 1 edge)
    ind_sz, = np.where(sz > 1)
    ind_sz += 1
    nr_components = np.size(ind_sz)
    sz_links = np.zeros((nr_components,))
    for i in range(nr_components):
        nodes, = np.where(ind_sz[i] == a)
        if extent:
            sz_links[i] = np.sum(adj[np.ix_(nodes, nodes)]) / 2
        else:
            sz_links[i] = np.sum(adjT[np.ix_(nodes, nodes)]) / 2

        adj[np.ix_(nodes, nodes)] *= (i + 2)

    # subtract 1 to delete any edges not comprising a component
    adj[np.where(adj)] -= 1

    if np.size(sz_links):
        max_sz = np.max(sz_links)
    else:
        # max_sz=0
        raise ValueError('True matrix is degenerate')
    print('max component size is %i' % max_sz)

    # estimate empirical null distribution of maximum component size by
    # generating k independent permutations
    print('estimating null distribution with %i permutations' % k)

    null = np.zeros((k,))
    hit = 0

    ind_shuff1 = np.array(range(0, y_vec.__len__()))
    ind_shuff2 = np.array(range(0, y_vec.__len__()))

    for u in range(k):
        # randomize
        np.random.shuffle(ind_shuff1)
        np.random.shuffle(ind_shuff2)
        # perform pearson corr test at each edge
        z_stat_perm = np.apply_along_axis(corr_with_vars, 1, xmat, y_vec[ind_shuff1])

        ind_r, = np.where(z_stat_perm > thresh)

        adj_perm = np.zeros((n, n))

        if extent:
            adj_perm[(ixes[0][ind_r], ixes[1][ind_r])] = 1
            adj_perm = adj_perm + adj_perm.T
        else:
            adj_perm[(ixes[0], ixes[1])] = z_stat_perm
            adj_perm = adj_perm + adj_perm.T
            adj_perm[adj_perm <= thresh] = 0

        a, sz = get_components(adj_perm)

        ind_sz, = np.where(sz > 1)
        ind_sz += 1
        nr_components_perm = np.size(ind_sz)
        sz_links_perm = np.zeros((nr_components_perm))
        for i in range(nr_components_perm):
            nodes, = np.where(ind_sz[i] == a)
            sz_links_perm[i] = np.sum(adj_perm[np.ix_(nodes, nodes)]) / 2

        if np.size(sz_links_perm):
            null[u] = np.max(sz_links_perm)
        else:
            null[u] = 0

        # compare to the true dataset
        if null[u] >= max_sz:
            hit += 1
        if verbose:
            print('permutation %i of %i.  Permutation max is %s.  Observed max'
                  ' is %s.  P-val estimate is %.3f') % (
                u, k, null[u], max_sz, hit / (u + 1))
        elif (u % (k / 10) == 0 or u == k - 1):
            print('permutation %i of %i.  p-value so far is %.3f' % (u, k,
                                                                     hit / (u + 1)))
    pvals = np.zeros((nr_components,))
    # calculate p-vals
    for i in range(nr_components):
        pvals[i] = np.size(np.where(null >= sz_links[i])) / k

    return pvals, adj, null


def get_nbs_inputs(input_dir, *population_dicts):
    ''' Gets you the matrices and the y_vec in the right format for the NBS function.

    Parameters
    ----------
    input_dir : str
        path to the input directory
    population_dicts : list of dicts
        dictionary with the following keys:
        - subject_list : list with the subjects to be included in the analysis
        - metric : str with the metric to be used
        - freqs : list with the frequencies to be included in the analysis
        - condition : str with the condition to be included in the analysis
    
    Returns
    -------
    stacked_matrices : np.ndarray
        3D array with the matrices of each subject stacked
    group_vec : np.ndarray
        vector with the group of each subject (0 = 1st group, 1 = 2nd group, etc)
    '''
    def get_subjects_df_list(subject_list, metric, freqs, condition, input_dir):
        '''
        Create a list with the matrix (df) of each subject. Transform each matrix to fill the upper triangle with the lower triangle values.
        '''
        matrix_list = []
        subject_list = sorted(subject_list)
        for subject in subject_list:
            subject_id = str(subject).zfill(2)
            if condition == 'RESTINGSTATEOPEN':
                cnd_abrv = 'open'
            elif condition == 'RESTINGSTATECLOSE':
                cnd_abrv = 'closed'
            df = pd.read_csv(os.path.join(input_dir, f'sub-{subject_id}', condition, 'connectivity', 'static', 'conn_data', f'sub-{subject_id}-static-{metric}-{freqs[0]}-{freqs[-1]}-{cnd_abrv}.csv'), index_col=0)
            df = np.triu(df.T, 1) + df
            matrix_list.append(df)
        return matrix_list

    def stack_matrices(*mat_lst):
        '''
        Stack the matrices of each subject in a 3D array
        '''
        new_lst=[]
        for lst in mat_lst:
            for mat in lst:
                new_lst.append(mat)
        stacked_mat = np.stack(new_lst, axis=-1)
        return stacked_mat

    def get_group_vec(*lists):
        '''
        Create a vector with the a value for each subject that represent its group
        '''
        group_vec = []
        for i, lst in enumerate(lists):
            group_vec.extend([i]*len(lst))
        return np.array(group_vec)
    

    matrix_lists = []
    for pop_dict in population_dicts:
        subj_list = pop_dict['subject_list']
        metric_val = pop_dict['metric']
        freqs_val = pop_dict['freqs']
        condition_val = pop_dict['condition']
        
        matrix_list = get_subjects_df_list(subj_list, metric_val, freqs_val, condition_val, input_dir)
        matrix_lists.append(matrix_list)


    stacked_matrices = stack_matrices(*matrix_lists)
    group_vec = get_group_vec(*matrix_lists)

    return stacked_matrices, group_vec

def nbs_report(pvals, adj, null, thresh, output_dir, *names):
    ''' Create a report with the results of the NBS analysis. 

    Parameters
    ----------
    pvals : np.ndarray
        p-values of the NBS analysis
    adj : np.ndarray
        adjacency matrix of the NBS analysis
    null : np.ndarray
        null distribution of the NBS analysis
    output_dir : str
        path to the output directory
    names : list
        list with the names of the comparisons

    Returns
    -------
    None   
    '''

    # Create a string with the names of the comparisons
    names_str = ''
    for i, name in enumerate(names):
        if i < len(names)-1:
            names_str += name + '_VS_'
        else:
            names_str += name
    
    # Create a dataframe with only the pvalues
    pvals_df = pd.DataFrame(pvals, columns=['pvals'])
    # Create a column with the names of the comparisons
    pvals_df['comparison'] = names_str
    pvals_df['thresh'] = thresh

    names_str = names_str + '_' +str(thresh).replace('.', '')

    # Create a dataframe with the null distribution 
    null_df = pd.DataFrame(null)

    # Create a dataframe with the adjacency matrix
    # Get the channels names to use them as columns and index
    ref_matrix = pd.read_csv(os.path.join(output_dir, 'all_subj', 'resting-state', 'connectivity', 'static', 'old_control', 'conn_data', 'old_control-static-plv-4-8-open.csv'), index_col=0)
    chan_names = ref_matrix.index.values
    adj_df = pd.DataFrame(adj, columns=chan_names, index=chan_names)

    # There is a bug that makes the adj_df hold values above 1.0
    adj_df = adj_df.applymap(lambda x: 1 if x > 1 else x)

    # Save the dataframes
    if not os.path.exists(os.path.join(output_dir, 'all_subj', 'resting-state', 'connectivity', 'static', 'clean_nbs_results', names_str)):
        os.makedirs(os.path.join(output_dir, 'all_subj', 'resting-state', 'connectivity', 'static', 'clean_nbs_results', names_str))
    pvals_df.to_csv(os.path.join(output_dir, 'all_subj', 'resting-state', 'connectivity', 'static', 'clean_nbs_results', names_str, 'pvals.csv'))
    null_df.to_csv(os.path.join(output_dir, 'all_subj', 'resting-state', 'connectivity', 'static', 'clean_nbs_results', names_str, 'null.csv'))
    adj_df.to_csv(os.path.join(output_dir, 'all_subj', 'resting-state', 'connectivity', 'static', 'clean_nbs_results', names_str, 'adj.csv'))

def global_pval_df(input_dir, output_dir):

    nbs_dir = os.path.join(input_dir, 'all_subj', 'resting-state', 'connectivity', 'static', 'nbs_results')
    nbs_list = sorted(glob.glob(nbs_dir + '/*'))

    df_list = []

    for lst in nbs_list:

        try:
            pval_file = pd.read_csv(os.path.join(lst, 'pvals.csv'))
            df_list.append(pval_file)
            print('===== pvals.csv was added to the list =====')
            print(f'===== {lst} =====')
        except:
            print('===== pvals.csv was not added to the list =====')
            print(f'===== {lst} =====')

    # stack the dfs
    full_df = pd.concat(df_list, axis=0)
    # take only the significant pvals
    sign_df = full_df[full_df['pvals'] < 0.05]

    # save the dfs
    if not os.path.exists(os.path.join(output_dir, 'all_subj', 'resting-state', 'connectivity', 'static', 'nbs_results', 'all_pvals')):
        os.makedirs(os.path.join(output_dir, 'all_subj', 'resting-state', 'connectivity', 'static', 'nbs_results', 'all_pvals'))
        print('===== all_pvals directory was created =====')
    full_df.to_csv(os.path.join(output_dir, 'all_subj', 'resting-state', 'connectivity', 'static', 'nbs_results', 'all_pvals', 'all_pvals.csv'))
    sign_df.to_csv(os.path.join(output_dir, 'all_subj', 'resting-state', 'connectivity', 'static', 'nbs_results', 'all_pvals', 'sign_pvals.csv'))

    return full_df

def plot_bin_mat(input_dir):
    '''
    Plot the binary matrix of the NBS analysis
    '''
    nbs_dir = os.path.join(input_dir, 'all_subj', 'resting-state', 'connectivity', 'static', 'nbs_results')
    nbs_list = sorted(glob.glob(nbs_dir + '/*'))

    for lst in nbs_list:

        try:

            df = pd.read_csv(os.path.join(lst, 'adj.csv'), index_col=0)

            # Select only the upper triangle
            x = np.tril(df)
            ticks = df.index
            df = pd.DataFrame(x, index=ticks, columns=ticks)

            title = 'thresh: ' + lst.split('_')[-1]

            def add_square(ax, xy, size):
                red_square=plt.Rectangle((xy), size[0], size[1], linewidth=2, edgecolor='red', facecolor=(1, 0 ,0, 0.3))
                ax.add_patch(red_square)
            def add_triangle(ax, xy):
                triangle = plt.Polygon(xy, closed=True, edgecolor='red', facecolor=(1, 0, 0, 0.3))
                ax.add_patch(triangle)
            fig=plt.figure(figsize=(10, 10))
            ax=sns.heatmap(df, cbar=False, square=True, xticklabels=True, yticklabels=True)
            ax.set_title(title, fontsize=20)
            xys=[(3, 56), (19, 56), (37, 56), (3, 37), (3,19), (19, 37)]
            sizes=[(8, 7), (7, 7), (10, 7), (8, 10), (8, 7), (7,10)]
            triangle_vertices = [[(3, 3), (11, 11), (3, 11)], [(19, 19), (26, 26), (19, 26)], [(37, 37), (47, 47), (37, 47)], 
                                [(56, 56), (56, 63), (63, 63)]]
            for i, xy in enumerate(xys):
                add_square(ax, xy, sizes[i])
            for i, vert in enumerate(triangle_vertices):
                add_triangle(ax, vert)
            fig.savefig(os.path.join(lst, 'binary_matrix.png'))
            plt.close()
            print('===== binary matrix was saved =====')
            print(f'===== {lst} =====')
        except:
            print('===== binary matrix was not saved =====')
            print(f'===== {lst} =====')

    return None
