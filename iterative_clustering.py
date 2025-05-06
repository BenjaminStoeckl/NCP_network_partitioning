#!interpreter [optional-arg]
# -*- coding: utf-8 -*-

__author__ = ["Benjamin Stöckl"]
__copyright__ = "Copyright 2023, Graz University of Technology"
__credits__ = ["Benjamin Stöckl"]
__license__ = "MIT"
__maintainer__ = "Benjamin Stöckl"
__status__ = "Development"

import pandas as pd
import numpy as np
import os
import data
import gen_load_aggregation as gla
import grid_partitioning as gp
import time
import scipy.spatial.distance as ssd
from natsort import natsorted
import warnings

warnings.filterwarnings('ignore')


def get_adjacency_matrix(df_adj_parameter: pd.DataFrame):
    '''

    creates an adjacency matrix from the given adjacency parameters. They have to have dimension (N x ...) because the
    second dimension can be any size.

    returns a pandas DataFrame with the adjacency matrix with size (NxN)
    '''

    # get adjacency matrix of lmp differences

    # lmp_diff = df_lmp_diff.to_numpy()[:, 0]
    # adjacency_matrix = lmp_diff[:, np.newaxis] - lmp_diff[np.newaxis, :]

    adj_mat = ssd.pdist(df_adj_parameter.to_numpy(), 'euclidean')
    adj_mat = ssd.squareform(adj_mat)

    df_adjacency_matrix = pd.DataFrame(adj_mat, index=df_adj_parameter.index, columns=df_adj_parameter.index)

    return df_adjacency_matrix


def iterative_clustering(case_folder: str,
                         res_path: str,
                         data_folder: str,
                         tec_type='tecAss',
                         cluster_by='itCluster',
                         l_number_of_clusters=[-1]
                         ):

    """

    Note:

    :param case_folder: name of case folder
    :param res_path:    path to results folder
    :param data_folder: name of data folder
    :param tec_type:    definition of assigning generators or aggregating
    :param cluster_by:  define clustering algorithm
    :param l_number_of_clusters: list of number of clusters
    :return:
    """

    def update_cluster_info(df_cluster, df_ci):

        '''
        Define "Old Cluster" as new buses and add new buses
        '''

        # only define new buses if clusters defined in df_bus_info
        if 'cluster' in df_ci.columns:

            df_ci.rename(columns={'cluster': 'OldCluster'}, inplace=True)

            if df_ci.index.dtype != object:
                df_ci.index = df_ci.index.astype(str)

            for i, row in df_ci.copy().iterrows():
                try:
                    df_ci.loc[i, 'cluster'] = str(int(df_cluster.loc[row['OldCluster'], 'cluster']))
                except:
                    breakpoint()
            # df_ci['cluster'] = df_ci['cluster'].astype(int)

            df_ci.drop(columns='OldCluster', inplace=True)

        else:

            df_ci['cluster'] = df_cluster['cluster']

        return df_ci


    def define_node_cluster(distance_metric: pd.DataFrame, node1: str, node2: str):

        # create df_node_cluster and initialize with individual cluster at each node 1...len(num_of_nodes)
        df_node_cluster = pd.DataFrame([str(i) for i in range(1, len(distance_metric) + 1)],
                                       index=distance_metric.index, columns=['cluster'])
        df_node_cluster.index = df_node_cluster.index.astype(str)

        # don't change the cluster of the nodes 'node1' and 'node2' if the number of clusters is the same as the number of nodes
        if number_of_clusters == len(distance_metric_sum):
            pass
        else:
            # assign every node an individual cluster except the nodes 'node1' and 'node2'
            cluster_counter = 2
            for idx, row in df_node_cluster.iterrows():
                if idx == node1:
                    df_node_cluster.loc[idx, 'cluster'] = 1
                elif idx == node2:
                    df_node_cluster.loc[idx, 'cluster'] = 1
                else:
                    df_node_cluster.loc[idx, 'cluster'] = cluster_counter
                    cluster_counter = cluster_counter + 1

        return df_node_cluster



# BEGIN MAIN iterative_clustering --------------------------------------------------------------------------------------

    if l_number_of_clusters == [-1]:
        exit('Please define a list of number of clusters for adjacent node agglomerative clustering.')

    # load input data & data from model run
    duals, lines, df_ptdf, df_incidence_matrix_full = gp.load_data(res_path, case_folder, data_folder)

    for column in ['Xline', 'line', 'Country']:
        if column in lines.columns:
            lines.drop(columns=column, inplace=True)

    # get vectors for duals
    df_phi_upper, phi_upper = gp.get_duals_as_vector(duals, 'eMaxTransport')
    df_phi_lower, phi_lower = gp.get_duals_as_vector(duals, 'eMinTransport')

    if 'lmp' in cluster_by:

        distance_metric = df_ptdf.T @ (df_phi_upper - df_phi_lower)  # get difference of lmps to slack bus for initial clustering
        distance_metric = distance_metric.reindex(natsorted(distance_metric.index))

    elif 'ncp' in cluster_by:

        distance_metric = gp.get_network_congestion_price_matrix(df_ptdf, df_phi_upper, df_phi_lower)
        distance_metric.columns = pd.RangeIndex(distance_metric.columns.size)  # rest columns to avoid errors with MultiIndex
        distance_metric.reindex(natsorted(distance_metric.index))

    else:
        exit('No valid cluster_by parameter given. Please choose between "LMP" and "NCI".')

    if distance_metric.index.dtype != object:
        breakpoint()

    # define df for sum of differences
    distance_metric_sum = distance_metric
    distance_metric_sum['counter'] = 1

    # INDEX OF distance_metric MUST HAVE NODES
    number_of_clusters = len(distance_metric)

    df_bus_info = pd.read_csv(os.path.join(data_folder, 'data', case_folder, 'bus_info.csv'), sep=";", decimal=",")
    df_bus_info.drop(columns={'cluster'}, inplace=True)
    df_bus_info.set_index('bus', inplace=True)

    df_cluster_info = df_bus_info.copy()
    df_cluster_info.index = df_cluster_info.index.astype(str)
    df_incidence_matrix = df_incidence_matrix_full.copy()
    df_node_inc_matrix = df_incidence_matrix.T @ df_incidence_matrix

    # load slack bus from original case
    slack_bus_original_grid = data.load_opf_parameter(case_folder, 'SlackBus', ',', data_folder)

    start = time.time()
    while number_of_clusters >= min(l_number_of_clusters):  # and number_of_clusters > 1:

        # get the adjacency matrix of the lmp differences
        df_adjacency_matrix = get_adjacency_matrix(distance_metric)

        # check if df_node_inc_matrix and distance_metric have the same index and columns
        if (df_node_inc_matrix.index != distance_metric.index).all() or (df_node_inc_matrix.columns != distance_metric.index).all():
           exit('df_node_inc_matrix and distance_metric have different indices or columns. Please check the input data.')

        # make a copy of the node incidence matrix to get the unweighted version
        df_node_inc_matrix_unweighted = df_node_inc_matrix.copy()

        # fill diagonal of node incidence matrix with 0
        node_inc_matrix = df_node_inc_matrix.to_numpy()
        np.fill_diagonal(node_inc_matrix, 0)
        df_node_inc_matrix = pd.DataFrame(node_inc_matrix, index=distance_metric.index, columns=distance_metric.index).abs()

        # replace 0 values with high value to avoid clustering and replace connected nodes (1) with 0
        df_node_inc_matrix = df_node_inc_matrix.replace(0, df_adjacency_matrix.max().max()*2)  # replace entries with no connection with 100 to avoid clustering
        df_node_inc_matrix = df_node_inc_matrix.replace(1, 0)  # replace entries with connection with 0

        # add both matrices to get a combined matrix
        df_weighted_adj_matrix = df_adjacency_matrix.abs() + df_node_inc_matrix
        dist_matrix = df_weighted_adj_matrix.to_numpy()
        np.fill_diagonal(dist_matrix, dist_matrix.max().max() * 2)  # set diagonal to value bigger than max to avoid combining same nodes

        # get index of minimum value and corresponding indices in dataframe
        min_index_flat = np.argmin(dist_matrix)  # index of min value for one dimensional array
        min_index = np.unravel_index(min_index_flat, dist_matrix.shape)  # index of min value for two dimensional array with dimensions of dist_matrix

        # get corresponding node indices
        node1 = distance_metric.index[min_index[0]]
        node2 = distance_metric.index[min_index[1]]

        # define df_node_cluster
        df_node_cluster = define_node_cluster(distance_metric, node1, node2)

        df_cluster_info = update_cluster_info(df_node_cluster, df_cluster_info)

        # if only one aggregation is needed, return df_cluster_info
        if len(l_number_of_clusters) == 1 and number_of_clusters in l_number_of_clusters:
            return df_cluster_info

        # define bus zone incidence matrix from df_node_cluster
        df_zone_incidence_matrix = gp.get_node_zone_incidence_matrix(df_node_cluster)

        if number_of_clusters in l_number_of_clusters:

            # export grid partitioning time
            end = time.time()
            gp.export_grid_partitioning_time(res_path, case_folder, cluster_by, number_of_clusters, start, end)

            # export aggregated grid
            agg_case = f'{case_folder}_red_{cluster_by}_{number_of_clusters}_{tec_type}'
            gla.check_if_folder_exists_and_create(data_folder, 'data', agg_case)
            gp.export_aggregated_grid(res_path,
                                      data_folder,
                                      case_folder,
                                      agg_case,
                                      df_cluster_info,
                                      lines,
                                      df_ptdf.copy(),
                                      slack_bus_original_grid
                                      )

            start = time.time()  # start time for next iteration

        # define new zone node incidence matrix for next iteration
        df_node_inc_matrix = (df_zone_incidence_matrix @ df_node_inc_matrix_unweighted @ df_zone_incidence_matrix.T).abs()

        # reorder columns and index by natsorted columns
        df_node_inc_matrix = df_node_inc_matrix[natsorted(df_node_inc_matrix.columns)]
        df_node_inc_matrix = df_node_inc_matrix.reindex(natsorted(df_node_inc_matrix.index))

        if 'cluster' in distance_metric_sum.columns:
            distance_metric_sum.drop(columns='cluster', inplace=True)

        # distance_metric_sum['cluster'] = df_node_cluster['cluster']
        distance_metric_sum = distance_metric_sum.join(df_node_cluster, how='left')

        # calculate the sum of the values in distance_metric_sum per cluster
        distance_metric_sum = distance_metric_sum.groupby('cluster').sum()
        distance_metric_sum = distance_metric_sum.reindex(natsorted(distance_metric_sum.index))
        distance_metric_sum.index = distance_metric_sum.index.astype(str)

        # calculate new distance_metric by dividing the sum by the number of nodes in the cluster
        distance_metric = (distance_metric_sum.T / distance_metric_sum['counter']).T
        distance_metric.drop(columns="counter", inplace=True)
        distance_metric.index.name = 'OldCluster'

        # reduce number of clusters
        number_of_clusters = number_of_clusters - 1

        # end while


    return pd.DataFrame




if __name__ == '__main__':

    periods = 1
    results_folder = os.path.join('..', '..', 'GridAggregation', 'results')
    data_folder = os.path.join('..', '..', 'GridAggregation')
    case_folder = 'IEEE_24_p19'
    cluster_by = 'lmpAnac'
    l_num_of_nodes = list(range(24, 0, -1))

    iterative_clustering(case_folder,
                         results_path,
                         data_folder,
                         cluster_by=cluster_by,
                         l_number_of_clusters=l_num_of_nodes
                         )


