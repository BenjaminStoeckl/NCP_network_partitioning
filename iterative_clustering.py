#!interpreter [optional-arg]
# -*- coding: utf-8 -*-

__author__ = ["Benjamin Stöckl"]
__copyright__ = "Copyright 2025, Graz University of Technology"
__credits__ = ["Benjamin Stöckl"]
__license__ = "MIT"
__maintainer__ = "Benjamin Stöckl"
__status__ = "Development"


import pandas as pd
import numpy as np
import os
import gen_load_aggregation as gla
import evaluation as eva
import time
import scipy.spatial.distance as ssd
import warnings
warnings.filterwarnings('ignore')


def get_adjacency_matrix(df_adj_parameter: pd.DataFrame):
    '''

    creates an adjacency matrix from the given adjacency parameters. They have to have dimension (N x ...) because the
    second dimension can be any size.

    returns a pandas DataFrame with the adjacency matrix with size (NxN)
    '''

    adj_mat = ssd.pdist(df_adj_parameter.to_numpy(), 'euclidean')
    adj_mat = ssd.squareform(adj_mat)

    df_adjacency_matrix = pd.DataFrame(adj_mat, index=df_adj_parameter.index, columns=df_adj_parameter.index)

    return df_adjacency_matrix


def iterative_clustering(case_folder, res_path, tec_type='tecAss', cluster_by='itCluster', l_number_of_clusters=[1]):


    def update_bus_info(df_cluster, df_bi):

        '''
        Define "Old Cluster" as new buses and add new buses
        '''

        # only define new buses if clusters defined in df_bus_info
        if 'cluster' in df_bi.columns:
            #TODO: Add weighting of the coordinates (long term)

            # df_cluster.reset_index(drop=False, inplace=True)

            for i, row in df_bi.iterrows():
                row['cluster'] = df_cluster.loc[row['cluster'], 'cluster']


        else:
            # if no defined clusters are in bus info, only add clusters to df
            df_bi = df_bi.join(df_cluster, how="left")

        return df_bi


    def update_cluster_info(df_cluster, df_ci):

        '''
        Define "Old Cluster" as new buses and add new buses
        '''

        # only define new buses if clusters defined in df_bus_info
        if 'cluster' in df_ci.columns:

            df_ci.rename(columns={'cluster': 'OldCluster'}, inplace=True)

            for i, row in df_ci.copy().iterrows():
                df_ci.loc[i, 'cluster'] = str(int(df_cluster.loc[row['OldCluster'], 'cluster']))

            df_ci['cluster'] = df_ci['cluster'].astype(int)

            df_ci.drop(columns='OldCluster', inplace=True)

        else:

            df_ci['cluster'] = df_cluster['cluster']

        return df_ci



# BEGIN MAIN iterative_clustering --------------------------------------------------------------------------------------

    # load input data & data from model run
    duals, lines, df_ptdf, ptdf, df_incidence_matrix = eva.load_data(res_path, case_folder)
    lines.drop(columns=['line', 'Xline', 'Country'], inplace=True)

    # get vectors for duals
    df_phi_upper, phi_upper = eva.get_duals_as_vector(duals, 'eMaxTransport')
    df_phi_lower, phi_lower = eva.get_duals_as_vector(duals, 'eMinTransport')

    if 'LMP' in cluster_by:

        # get difference of lmps to slack bus for initial clustering
        df_lmp_diff = df_ptdf.T @ (df_phi_upper - df_phi_lower)

    elif 'NCI' in cluster_by:

        df_nci = eva.get_node_cluster_information_matrix(df_ptdf, df_phi_upper, df_phi_lower).T
        df_lmp_diff = df_nci

    else:
        exit('No valid cluster_by parameter given. Please choose between "LMP" and "NCI".')


    # define df for sum of differences
    df_lmp_diff_sum = df_lmp_diff
    df_lmp_diff_sum['counter'] = 1

    # INDEX OF df_lmp_diff MUST HAVE NODES
# ===================== POSSIBLE BEGIN FOR FOR LOOP =====================

    orig_case = case_folder

    loop_counter = 0
    number_of_clusters = len(df_lmp_diff)

    # load bus info data and set bus to index
    df_bus_info = pd.read_csv(os.path.join('data', case_folder, 'bus_info.csv'), sep=";", decimal=",")
    df_bus_info.drop(columns={'Country', 'BaseVolt', 'Name', 'cluster'}, inplace=True)
    df_bus_info.set_index('bus', inplace=True)

    df_cluster_info = df_bus_info.copy()
    df_cluster_info.index = df_cluster_info.index.astype(str)

    loop_cluster = True

    while number_of_clusters > min(l_number_of_clusters):

        start = time.time()
        df_adjacency_matrix = get_adjacency_matrix(df_lmp_diff)

        # calculate incidence matrix on nodal basis
        df_node_inc_matrix = df_incidence_matrix.T @ df_incidence_matrix

        # fill diagonal of node incidence matrix with 0
        node_inc_matrix = df_node_inc_matrix.to_numpy()
        np.fill_diagonal(node_inc_matrix, 0)

        df_node_inc_matrix = pd.DataFrame(node_inc_matrix, index=df_lmp_diff.index, columns=df_lmp_diff.index).abs()
        df_node_inc_matrix = df_node_inc_matrix.replace(0, 100)  # replace entries with no connection with 100 to avoid clustering
        df_node_inc_matrix = df_node_inc_matrix.replace(1, 0)  # replace entries with connection with 0

        # add both matrices to get a combined matrix
        df_weighted_adj_matrix = df_adjacency_matrix.abs() + df_node_inc_matrix

        dist_matrix = df_weighted_adj_matrix.to_numpy()
        np.fill_diagonal(dist_matrix, 1000)
        df_dist_matrix = pd.DataFrame(dist_matrix, index=df_lmp_diff.index, columns=df_lmp_diff.index)

        min_index_flat = np.argmin(dist_matrix)  # index of min value for one dimensional array
        min_index = np.unravel_index(min_index_flat, dist_matrix.shape)  # index of min value for two dimensional array with dimensions of dist_matrix

        # get corresponding node indices
        node1 = df_lmp_diff.index[min_index[0]]
        node2 = df_lmp_diff.index[min_index[1]]

        end = time.time()
        print('Time to identify nodes: ' + str(end - start))

        print('Iteration: ' + str(loop_counter) + ', Cluster Nodes: ' + str(node1) + ' and ' + str(node2))

        loop_counter_1 = loop_counter
        num_of_nodes_1 = ptdf.shape[1]
        number_of_clusters = len(df_lmp_diff) - 1

        start = time.time()

        # create df_node_cluster and initialize with individual cluster at each node 1...len(num_of_nodes)
        df_node_cluster = pd.DataFrame([0 for i in range(0, len(df_lmp_diff))], index=df_lmp_diff.index, columns=['cluster'])


        # assign every node an individual cluster except the nodes 'node1' and 'node2'
        cluster_counter = 2
        for i, row in df_node_cluster.iterrows():
            if i == node1:
                df_node_cluster.loc[i, 'cluster'] = 1
            elif i == node2:
                df_node_cluster.loc[i, 'cluster'] = 1
            else:
                df_node_cluster.loc[i, 'cluster'] = cluster_counter
                cluster_counter = cluster_counter + 1

        # define nodes of lines to clusters
        lines = eva.assign_lines_to_clusters(lines, df_node_cluster)

        # if lines have same cluster, drop in line and in PTDF matrix
        if (lines['cluster1'] == lines['cluster2']).any():
            drop_index = lines[lines['cluster1'] == lines['cluster2']][['bus1', 'bus2']]
            df_ptdf.drop((drop_index['bus1'].values[0], drop_index['bus2'].values[0]), axis='index', inplace=True)
            ptdf = df_ptdf.to_numpy()
            lines.drop(drop_index.index, axis='index', inplace=True)

        # need reduced ptdf for df_incidence_matrix
        zone_incidence_matrix, _ = eva.get_node_zone_incidence_matrix(df_node_cluster, df_node_cluster['cluster'].max())
        df_zone_incidence_matrix = pd.DataFrame(zone_incidence_matrix,
                                                columns=df_node_cluster.index,
                                                index=range(1, df_node_cluster['cluster'].max() + 1))

        # get zone lines from lines witch have different clusters
        zone_lines = lines.loc[lines['cluster1'] != lines['cluster2']]

        # order zone lines in ascending order (by number of cluster)
        for i, row in zone_lines.iterrows():
            if row['cluster1'] > row['cluster2']:
                zone_lines.loc[i, 'cluster1'], zone_lines.loc[i, 'cluster2'] = \
                    zone_lines.loc[i, 'cluster2'], zone_lines.loc[i, 'cluster1']
                zone_lines.loc[i, 'bus1'], zone_lines.loc[i, 'bus2'] = \
                    zone_lines.loc[i, 'bus2'], zone_lines.loc[i, 'bus1']

        # get lines between clusters and drop all duplicates
        linked_zones = zone_lines[['cluster1', 'cluster2']].copy().astype(int)
        linked_zones.drop_duplicates(inplace=True)

        t_zone_links = list(zip(linked_zones['cluster1'].values, linked_zones['cluster2'].values))

        lines.set_index(['bus1', 'bus2'], inplace=True, drop=False)

        # inter zone lines are directed (1/-1)
        df_inter_zones_lines = pd.DataFrame(0,
                                            index=pd.MultiIndex.from_tuples(t_zone_links,
                                                                            names=['cluster1', 'cluster2']),
                                            columns=lines.index)
        for i, row in lines.iterrows():
            if (int(row['cluster1']), int(row['cluster2'])) in t_zone_links:
                df_inter_zones_lines.loc[(int(row['cluster1']), int(row['cluster2'])), i] = 1
            elif (int(row['cluster2']), int(row['cluster1'])) in t_zone_links:
                df_inter_zones_lines.loc[(int(row['cluster2']), int(row['cluster1'])), i] = - 1

        inter_zones_lines = df_inter_zones_lines.to_numpy()

        end = time.time()
        print('Time to create matrices: ' + str(end - start))

        # get reduced ptdf matrix and save to input folder
        df_ptdf, ptdf = eva.get_reduced_ptdf_matrix(ptdf, inter_zones_lines, df_inter_zones_lines,
                                                    zone_incidence_matrix, df_T_bz=df_zone_incidence_matrix,
                                                    df_ptdf=df_ptdf, use_df=True)

        lines.drop(columns=['bus1', 'bus2'], inplace=True)
        lines.rename(columns={'cluster1': 'bus1', 'cluster2': 'bus2'}, inplace=True)

        # order lines to ascend order (by number of cluster)
        for i, row in lines.iterrows():
            if row['bus1'] > row['bus2']:
                lines.loc[i, 'bus1'], lines.loc[i, 'bus2'] = \
                    lines.loc[i, 'bus2'], lines.loc[i, 'bus1']

        lines.reset_index(inplace=True, drop=True)
        lines = lines.groupby(['bus1', 'bus2'], as_index=False).sum()
        lines = lines.astype({'bus1': int, 'bus2': int})
        lines.reset_index(inplace=True, drop=True)
        df_cluster_info = update_cluster_info(df_node_cluster, df_cluster_info)

        # only export case if needed
        if number_of_clusters in l_number_of_clusters:

            out_case = orig_case + '_red_' + cluster_by + '_' + str(number_of_clusters) + '_' + tec_type
            gla.check_if_folder_exists_and_create('data', out_case)

            # export again, so existing nodes have same cluster
            df_bus_info = eva.export_assigned_clusters_to_bus_info(case_folder, df_node_cluster)
            df_cluster_info.to_csv(os.path.join('data', out_case, 'cluster_info.csv'), sep=';', decimal=',', index=True)

            # export aggregated case
            gla.assign_clustered_case('data', res_path, case_folder, out_case, 1,
                                      df_bus_info=df_cluster_info)

            # add Xline to lines if not exists
            if 'Xline' not in lines.columns:
                lines['Xline'] = 0.0

            # export reduced ptdf & lines data
            df_ptdf.to_csv(os.path.join('data', out_case, 'ptdf.csv'), index=True, sep=';', decimal=',')
            lines.to_csv(os.path.join('data', out_case, 'lines.csv'), index=False, sep=';', decimal=',')

            # set new case folder
            # case_folder = out_case

        # redefine df_incidence_matrix
        _, df_incidence_matrix = eva.get_incidence_matrix(lines, df_ptdf)

        # # make mean of lmp diff
        # df_lmp_diff['cluster'] = df_node_cluster['cluster']
        #
        # # redefine df_lmp_diff
        # df_lmp_diff = df_lmp_diff.groupby('cluster').mean()
        # df_lmp_diff.index.name = 'OldCluster'

        # make weighted mean of lmp diff
        # define new cluster in df_lmp_diff_sum
        df_lmp_diff_sum['cluster'] = df_node_cluster['cluster']

        # calculate the sum of the values in df_lmp_diff_sum per cluster
        df_lmp_diff_sum = df_lmp_diff_sum.groupby('cluster').sum()

        # calculate new df_lmp_diff by dividing the sum by the number of nodes in the cluster
        df_lmp_diff = (df_lmp_diff_sum.T / df_lmp_diff_sum['counter']).T
        df_lmp_diff.drop(columns="counter", inplace=True)
        df_lmp_diff.index.name = 'OldCluster'

        loop_counter = loop_counter + 1


    return None




if __name__ == '__main__':

    case_folder = 'IEEE_24_p19'
    results_path = os.path.join('.', 'results')
    cluster_methods = [
        'itClusterLMP',  # LMP-ANAC
        'itClusterNCI'   # NCP-ANAC
    ]
    l_num_of_nodes = list(range(23, 0, -1))

    for cluster_by in cluster_methods:
        iterative_clustering(case_folder, results_path, cluster_by=cluster_by, l_number_of_clusters=l_num_of_nodes)


