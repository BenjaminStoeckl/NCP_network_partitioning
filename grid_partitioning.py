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
import gen_load_aggregation as gla
import data
from sklearn.cluster import KMeans, SpectralClustering
from sklearn_extra.cluster import KMedoids
import iterative_clustering as ic
from natsort import natsorted
from scipy.sparse.linalg import eigsh
from sklearn.utils._arpack import _init_arpack_v0
from sklearn.utils.extmath import _deterministic_vector_sign_flip
from sklearn.utils import (
    check_random_state,
)

import warnings
warnings.filterwarnings('ignore')


def get_duals_as_vector(df_duals, constraint_name):

    # get vectors for duals
    df = df_duals.loc[:, (i.startswith(constraint_name) for i in df_duals.columns.values)].T
    aux_index = []
    for i in df.index.values:
        df.loc[i, 'bus1'] = i.split('_')[1]
        df.loc[i, 'bus2'] = i.split('_')[2]
        # aux_index.append((i.split('_')[1], i.split('_')[2]))
    # df.index = aux_index

    df.set_index(['bus1', 'bus2'], drop=True, inplace=True)

    # sort ptdf by natsorted indices
    df = df.reindex(natsorted(df.index))

    #reorder columns by natsorted columns
    df = df[natsorted(df.columns)]

    return df, df.to_numpy()


def get_network_congestion_price_matrix(df_ptdf, df_phi_upper, df_phi_lower):
    # make vector for phi and filter for empty entries
    df_phi = df_phi_upper - df_phi_lower
    df_filt_phi = df_phi[df_phi != 0]
    df_filt_phi.dropna(how='all', inplace=True)  # only drop rows if all columns are NaN
    df_filt_phi.fillna(0, inplace=True)

    # set datatype of Multiindex to string
    df_ptdf.index = df_ptdf.index.set_levels([
        df_ptdf.index.levels[0].astype(str),
        df_ptdf.index.levels[1].astype(str)
    ])

    # reduce ptdf to lines with dual variables
    df_ptdf_filt = df_ptdf.loc[df_filt_phi.index, :]



    # determine NCP matrix for single or multi-period cases
    l_aux = []
    df_aux = pd.DataFrame()
    for period in df_filt_phi.columns:

        df_aux = df_ptdf_filt.mul(df_filt_phi[period], axis=0)  # calculate NCP for every period
        df_aux.reset_index(inplace=True)
        df_aux['period'] = period
        df_aux.set_index(['period', 'bus1', 'bus2'], inplace=True)

        l_aux.append(df_aux.T)

    # Use squeeze() to convert the DataFrame to a Series
    # df = df_ptdf_filt.mul(df_filt_phi.squeeze(), axis=0)  # only works for single period cases
    df = pd.concat(l_aux, axis=1)  # used for multi-period cases

    return df


def export_assigned_clusters_to_bus_info(case_folder: str, df_cluster: pd.DataFrame, data_folder=''):

    df = df_cluster['cluster']
    df_bus_info = pd.read_csv(os.path.join(data_folder, 'data', case_folder, 'bus_info.csv'), sep=';', decimal=',')
    df_bus_info = df_bus_info.astype({'bus': str})
    df_bus_info.set_index('bus', inplace=True, drop=True)

    # if column cluster already existing in df_bus_info drop to assign new
    if 'cluster' in df_bus_info.columns:
        df_bus_info.drop(columns='cluster', inplace=True)

    if df.index.name == 'OldCluster':
        df.index.name = 'bus'

    if df.index.dtype == 'int64':
        df.index = df.index.astype(str)

    df_bus_info = df_bus_info.join(df, how='left')

    # df_bus_info['cluster'] = df_cluster['cluster'].tolist()
    if 'bus' in df_bus_info.columns:
        df_bus_info.drop(columns={'bus'}, inplace=True)

    df_bus_info.to_csv(os.path.join(data_folder, 'data', case_folder, 'bus_info.csv'), sep=';', decimal=',', index=True)

    return df_bus_info


def assign_lines_to_clusters(lines, df_cluster):

    # check if df_cluster has index with strings
    if df_cluster.index.dtype != object:
        exit('df_cluster has wrong index type!')

    for i in lines.index:
        try:
            lines.loc[i, 'cluster1'] = str(int(df_cluster.loc[i[0], 'cluster']))  # assign clusters to lines
            lines.loc[i, 'cluster2'] = str(int(df_cluster.loc[i[1], 'cluster']))
        except:
            breakpoint()
    # else:
    #     for i in lines.index:
    #         lines.loc[i, 'cluster1'] = int(df_cluster.loc[lines.loc[i, 'bus1'], 'cluster'])
    #         lines.loc[i, 'cluster2'] = int(df_cluster.loc[lines.loc[i, 'bus2'], 'cluster'])

    # drop lines with the same cluster
    # lines = lines[lines['cluster1'] != lines['cluster2']]

    return lines


def get_line_limits(zone_lines: pd.DataFrame):

    """
    The function defines the line limits for the aggregated grid. The line limits are defined by the sum of the parallel
    lines.

    :param zone_lines:
    :return:
    """

    # get lines with different clusters (zone_lines) -> are defining the max line limit
    df_line_limits = zone_lines[['Pmax', 'cluster1', 'cluster2']]
    df_line_limits = df_line_limits.groupby(['cluster1', 'cluster2']).sum()

    df_line_limits.reset_index(inplace=True, drop=False)
    df_line_limits.rename(columns={'cluster1': 'bus1', 'cluster2': 'bus2'}, inplace=True)
    df_line_limits = df_line_limits.astype({'bus1': 'int', 'bus2': 'int'})
    df_line_limits[['Xline', 'Country', 'line']] = 0

    return df_line_limits


def load_data(res_path, case_folder, data_folder='', slack_bus=""):

    # load duals
    duals = pd.read_csv(os.path.join(res_path, case_folder, 'duals.csv'), sep=';', decimal=',')
    # load line data
    lines = pd.read_csv(os.path.join(data_folder, 'data', case_folder, 'lines.csv'), sep=';', decimal=',')
    lines = lines.astype({'bus1': str, 'bus2': str})
    lines.set_index(['bus1', 'bus2'], inplace=True)

    # load ptdf data
    df_ptdf = data.load_ptdf(case_folder, ',', data_folder)

    # get full incidence matrix without slack bus column! dimension: L x B-1
    df_incidence_matrix_full = get_incidence_matrix(lines, df_ptdf)

    return duals, lines, df_ptdf, df_incidence_matrix_full


def get_incidence_matrix(lines, df_ptdf, slack_bus="", directed=True):

    """
    The function defines the incidence matrix for a given network. The matrix either has dimension L x B or L x B-1 .
    (slack-bus adjusted). The incidence matrix maps the lines of the network to the buses of the network. The matrix
    entries are 1 if the bus is the start or end of the line and -1 if the bus is the end of the line.

    :param lines:
    :param df_ptdf:
    :param slack_bus:
    :param directed:
    :return:
    """

    cell_value = -1 if directed else 1

    # if lines are already indexed by bus1 and bus2 only copy, else set index to bus1 and bus2 first
    if lines.index.names == ['bus1', 'bus2']:
        temp_lines = lines.copy()
    else:
        temp_lines = lines.astype({'bus1': str, 'bus2': str})
        temp_lines.set_index(['bus1', 'bus2'], inplace=True)

    df_C = pd.DataFrame(0, index=temp_lines.index, columns=df_ptdf.columns.astype(str), dtype=int)
    for idx, row in temp_lines.iterrows():
        df_C.loc[idx, str(idx[0])] = 1
        df_C.loc[idx, str(idx[1])] = cell_value

    # drop slack bus if slack bus is defined
    if slack_bus != "":
        df_C.drop(columns=slack_bus, inplace=True)  # drop slack bus -> C matrix has dimension L x B-1

    # sort lines by natsorted indices
    df_C = df_C.reindex(natsorted(df_C.index))
    # reorder columns by natsorted columns
    df_C = df_C[natsorted(df_C.columns)]

    return df_C


def get_node_zone_incidence_matrix(df_cluster: pd.DataFrame()):

    """
    The function defines the node-zone incidence matrix for the clustering. The node-zone incidence matrix is a matrix
    which maps the buses of the full grid to the aggregated grid. The matrix has dimension (B_r -1 x B-1) where B_r
    is the number of buses in the reduced grid and B is the number of buses in the full grid.

    :param df_cluster: Dataframe with assigned clusters to buses of full grid
    :param num_of_clus: number of clusters in the aggregated grid
    :return:
    """

    # print(df_cluster['cluster'].astype(int).astype(str).unique())


    # define node-zone incidence matrix with index clusters and columns buses of full grid
    df_nzi = pd.DataFrame(0,
                          index=df_cluster['cluster'].astype(int).astype(str).unique(),
                          columns=df_cluster.index.astype(int).astype(str))

    # set entries to 1 if bus has cluster assigned
    for idx, row in df_cluster.iterrows():
        df_nzi.loc[str(int(row['cluster'])), str(idx)] = 1

    return df_nzi


def get_line_incidence_matrix(lines: pd.DataFrame):

    # get lines which connects the clustered zones
    zone_lines = lines.loc[lines['cluster1'] != lines['cluster2']]
    if 'line' in zone_lines.columns:
        zone_lines = zone_lines.drop(columns=['line'])

    # order zone lines to ascending order (by number of cluster)
    zone_lines.reset_index(drop=False, inplace=True)
    for i, row in zone_lines.iterrows():
        if int(row['cluster1']) > int(row['cluster2']):
            zone_lines.loc[i, 'cluster1'], zone_lines.loc[i, 'cluster2'] = zone_lines.loc[i, 'cluster2'], zone_lines.loc[i, 'cluster1']
            zone_lines.loc[i, 'bus1'], zone_lines.loc[i, 'bus2'] = zone_lines.loc[i, 'bus2'], zone_lines.loc[i, 'bus1']
    zone_lines.set_index(['bus1', 'bus2'], inplace=True)

    # get unique links between zones
    linked_zones = zone_lines[['cluster1', 'cluster2']].copy()
    linked_zones.drop_duplicates(inplace=True)
    # get list of zone links as tuples
    t_zone_links = list(zip(linked_zones['cluster1'].values, linked_zones['cluster2'].values))
    # t_zone_links = [(str(b1), str(b2)) for (b1, b2) in t_zone_links]  # change from int to str

    # lines.set_index(['bus1', 'bus2'], inplace=True, drop=False)  --> index already int

    # inter zone lines are directed (1/-1) (matrix with information which original lines connect the zones)
    df_inter_zones_lines = pd.DataFrame(0,
                                        index=pd.MultiIndex.from_tuples(t_zone_links, names=['cluster1', 'cluster2']),
                                        columns=lines.index)

    # fill inter zone lines matrix (line incidence)
    for i, row in lines.iterrows():
        if (row['cluster1'], row['cluster2']) in t_zone_links:
            df_inter_zones_lines.loc[(row['cluster1'], row['cluster2']), i] = 1
        elif (row['cluster2'], row['cluster1']) in t_zone_links:
            df_inter_zones_lines.loc[(row['cluster2'], row['cluster1']), i] = - 1

    return df_inter_zones_lines, zone_lines


def get_reduced_ptdf_matrix(T_f, T_bz, ptdf=pd.DataFrame(), nodal_injection=pd.DataFrame(), slack_bus_original_grid='',
                            slack_bus_reduced_grid=''):
    # try to drop slack bus from clusters

    if slack_bus_original_grid != "":
        T_bz.drop(slack_bus_original_grid, axis=1, inplace=True)
        ptdf.drop(slack_bus_original_grid, axis=1, inplace=True)

    if slack_bus_reduced_grid != "":
        T_bz.drop(slack_bus_reduced_grid, axis=0, inplace=True)


    if nodal_injection.empty:
        try:
            # T_f PTDF_f T_bz^T(T_bzT_bz^T)^-1
            df_aux = T_bz.T @ np.linalg.inv(T_bz @ T_bz.T)
            df_aux.columns = T_bz.index
            red_ptdf = T_f @ ptdf @ df_aux
        except:
            breakpoint()
    # injection weighted PTDF
    else:

        # delete slack bus from T_bz, nodal_injection and ptdf
        T_bz.drop(int(slack_bus_reduced_grid), axis=0, inplace=True)
        T_bz.drop(slack_bus_original_grid, axis=1, inplace=True)
        nodal_injection.drop(slack_bus_original_grid, axis=0, inplace=True)
        ptdf.drop(slack_bus_original_grid, axis=1, inplace=True)

        # ToDo: check if red-PTDF is correctly calculated!!!

        # T_f PTDF_f diag(p_inj) T_bz^T(diag(T_bz p_inj))^-1
        term_to_inv = T_bz @ nodal_injection
        # check for zero entries to make matrix invertible
        zero_series = term_to_inv[term_to_inv == 0].dropna()
        zero_indices = zero_series.index  # zero indices BUT without slack bus
        term_to_inv[term_to_inv['1'] == 0] = 1
        inv_term = pd.DataFrame(np.linalg.inv(np.diag((term_to_inv)['1'].values)), index=term_to_inv.index, columns=term_to_inv.index)
        # inv_term.loc[zero_indices, :] = 0

        inj_diag = np.diag(nodal_injection.iloc[:, 0])
        inj_diag = pd.DataFrame(data=inj_diag, index=nodal_injection.index, columns=nodal_injection.index)

        red_ptdf = T_f @ ptdf @ inj_diag @ T_bz.T @ inv_term

        # add slack bus to reduced ptdf
        red_ptdf[slack_bus_reduced_grid] = 0

    if red_ptdf.index.names == ['cluster1', 'cluster2']:
        red_ptdf.index.names = ['bus1', 'bus2']


    return red_ptdf


def determine_reduced_ptdf(res_path, data_folder, case_folder, agg_case, df_clus, lines, df_ptdf, slack_bus_original_grid='', injection_weighted_ptdf=False):


    df_zone_incidence_matrix = get_node_zone_incidence_matrix(df_clus)
    data.export_incidence_matrix(df_zone_incidence_matrix, agg_case, ',', data_folder)

    slack_bus_reduced_grid = str(df_clus.loc[slack_bus_original_grid, 'cluster'])

    # get line incidence matrix and export
    line_incidence_matrix, zone_lines = get_line_incidence_matrix(lines)
    data.export_line_incidence(line_incidence_matrix, agg_case, ',', data_folder)

    if injection_weighted_ptdf:
        nodal_injection = pd.read_csv(os.path.join(res_path, case_folder, 'node_inj.csv'), sep=';', decimal=',',
                                      index_col=0, dtype={'bus': str})
    else:
        nodal_injection = pd.DataFrame()

    # get reduced ptdf matrix and save to input folder
    df_red_ptdf = get_reduced_ptdf_matrix(line_incidence_matrix,
                                          df_zone_incidence_matrix,
                                          df_ptdf,
                                          nodal_injection,
                                          slack_bus_original_grid,
                                          slack_bus_reduced_grid
                                          )

    df_red_ptdf[slack_bus_reduced_grid] = 0  # add slack bus column with 0 entries
    data.export_ptdf(df_red_ptdf, agg_case, ',', data_folder, slack_bus=slack_bus_reduced_grid)

    return df_red_ptdf, zone_lines, slack_bus_reduced_grid


def export_aggregated_grid(res_path, data_folder, case_folder, agg_case, df_cluster, lines, df_ptdf, slack_bus_original_grid):

    # export to bus_info.csv of input folder (map plotting)
    df_bus_info = export_assigned_clusters_to_bus_info(case_folder, df_cluster, data_folder)
    df_bus_info.to_csv(os.path.join(data_folder, 'data', agg_case, 'cluster_info.csv'), sep=';', decimal=',',
                       index=True)
    df_bus_info.to_csv(os.path.join(data_folder, 'data', agg_case, 'bus_info.csv'), sep=';', decimal=',',
                       index=True)

    # set data type of cluster to string
    df_cluster['cluster'] = df_cluster['cluster'].astype(int).astype(str)

    # define nodes of lines to clusters
    lines = assign_lines_to_clusters(lines, df_cluster)

    df_red_ptdf, zone_lines, slack_bus_reduced_grid = determine_reduced_ptdf(res_path,
                                                                             data_folder,
                                                                             case_folder,
                                                                             agg_case,
                                                                             df_cluster,
                                                                             lines,
                                                                             df_ptdf,
                                                                             slack_bus_original_grid)

    # define line capacities for lines between zones and export to input folder
    df_line_limits = get_line_limits(zone_lines)
    data.export_lines(df_line_limits, agg_case, ',', data_folder)

    # save slack bus of reduced grid to opf_parameters.csv
    opf_parameters = pd.read_csv(os.path.join(data_folder, 'data', case_folder, 'opf_parameters.csv'), sep=';', decimal=',')
    opf_parameters.loc[opf_parameters['parameter_name'] == 'SlackBus', 'parameter_value'] = slack_bus_reduced_grid
    opf_parameters.to_csv(os.path.join(data_folder, 'data', agg_case, 'opf_parameters.csv'), sep=';', decimal=',')

    return df_cluster


def export_grid_partitioning_time(res_path: str, case_folder: str, cluster_by: str, num_of_clus: int,
                                  start_clustering: float, end_clustering: float):
    """
    Export the time taken for grid partitioning to a CSV file.
    :param res_path: Path to the results folder.
    :param case_folder: Name of the case folder.
    :param cluster_by: Clustering method used.
    :param num_of_clus: Number of clusters.
    :param start_clustering: Start time of the clustering process.
    :param end_clustering: End time of the clustering process.
    """

    if os.path.exists(os.path.join(res_path, case_folder, 'grid_partitioning_time.csv')):
        df_calc_time = pd.read_csv(os.path.join(res_path, case_folder, 'grid_partitioning_time.csv'), sep=';',
                                   decimal=',')
        df_calc_time.set_index('number_of_clusters', inplace=True)
    else:
        df_calc_time = pd.DataFrame()
        df_calc_time.index_name = 'number_of_clusters'
    df_calc_time.loc[num_of_clus, cluster_by] = end_clustering - start_clustering
    df_calc_time.to_csv(os.path.join(res_path, case_folder, 'grid_partitioning_time.csv'), sep=';', decimal=',',
                        index=True, index_label='number_of_clusters')


def grid_aggregation(res_path: str,
                     case_folder: str,
                     agg_case: str,
                     num_of_clus: int,
                     cluster_by='lmp',
                     data_folder='',
                     num_of_dec=-1,
                     list_of_num_of_clust=[-1],
                     periods=1,
                     random_state=None,
                     ):


    def cluster_nodes(res_path, data_folder, case_folder, num_of_clus, num_of_dec, filter_nodes=[], cluster_by='lmps',
                      list_of_num_of_clust=[-1], random_state=None):


        def kmeans_clustering(number_of_clusters, node_distance_metric, random_state=None, filter_nodes=[]):

            clustering = KMeans(n_clusters=number_of_clusters,
                                random_state=random_state
                                ).fit(node_distance_metric)
            clusters = pd.DataFrame(clustering.labels_ + 1, index=distance_metric.index, columns=['cluster'])

            return clusters


        def kmedoids_clustering(distance_metric, num_of_clus, random_state=None):

            # calculate kmedoids with manhattan distance (l1 norm)
            clustering_algorithm = KMedoids(n_clusters=num_of_clus,
                                            metric='euclidean',
                                            max_iter=1000,
                                            method='pam',
                                            random_state=random_state,
                                            ).fit(distance_metric)

            clusters = pd.DataFrame(clustering_algorithm.labels_ + 1, index=distance_metric.index, columns=['cluster'])

            return clusters


        def lmps_by_spectral_clustering(LMP_nodal_diff, df_incidence_matrix, num_of_clus, random_state=None, filter_nodes=[],
                                        diy_spectral_clustering=False):

            """
            Algorithm to cluster nodes by Locational Marginal Prices by Spectral Clustering
            Mathematical definition in "Binding Constraints Grid Aggregation" Overleaf document
            Source: Cao et al. (2018)

            :param LMP_nodal_diff: array with LMP difference to slack bus for every node (B x P)
            :param df_incidence_matrix_undirected: incidence matrix of the network (L x B)
            :param num_of_clus: number of clusters which should get determined
            :param filter_nodes: list with nodes that should be filtered out
            :return: DataFrame with assigned cluster to nodes

            """

            # make sure df_incidence matrix is undirected
            df_incidence_matrix_undirected = abs(df_incidence_matrix)

            # reduce LMP_nodal_diff to dimensions (N x 1)
            period_reduction_method = 'Euclidean'  # 'mean' or 'Euclidean'

            # OPTION 1: mean of lmps of all periods
            if period_reduction_method == 'mean':
                LMP_nodal_diff = LMP_nodal_diff.mean(axis=1).to_frame()

            # OPTION 2: Euclidean norm of all periods
            elif period_reduction_method == 'Euclidean':
                LMP_nodal_diff = np.sqrt(np.square(LMP_nodal_diff).sum(axis=1)).to_frame()
            else:
                exit('No valid period reduction method selected!')

            # calculate inverse LMP difference per line
            inverse_LMP_line_diff = 1/abs(df_incidence_matrix @ LMP_nodal_diff)  # (LxP)

            inverse_LMP_line_diff.replace(np.inf, np.nan, inplace=True)
            inverse_LMP_line_diff.fillna(inverse_LMP_line_diff.max()*10, inplace=True)


            # define auxiliary matrix for spectral clustering
            aux = df_incidence_matrix_undirected.T @ np.diag(inverse_LMP_line_diff.squeeze())  # (B x L) * (L x L) = (B x L)
            aux.columns = inverse_LMP_line_diff.index

            # calculate Laplacian matrix
            laplacian_matrix = aux @ df_incidence_matrix_undirected  # (B x L) * (L x B) = (B x B)

            spectral_clustering = SpectralClustering(n_clusters=num_of_clus,
                                                     affinity='precomputed',
                                                     random_state=random_state,
                                                     ).fit(laplacian_matrix)

            # add assigned cluster to nodes
            LMP_nodal_diff['cluster'] = spectral_clustering.labels_ + 1
            LMP_nodal_diff['cluster'] = LMP_nodal_diff['cluster'].astype(int)

            return LMP_nodal_diff

        # directly return df_cluster in case of single node
        if num_of_clus == 1:
            # define same cluster for all nodes
            df_cluster = pd.DataFrame(str(1), index=df_incidence_matrix_full.columns, columns=['cluster'])
            # export to bus_info.csv of input folder (map plotting)
            df_bus_info = export_assigned_clusters_to_bus_info(case_folder, df_cluster, data_folder)
            df_bus_info.to_csv(os.path.join(data_folder, 'data', agg_case, 'cluster_info.csv'), sep=';', decimal=',',
                               index=True)
            return df_cluster

        # get vectors for duals
        df_phi_upper, _ = get_duals_as_vector(duals, 'eMaxTransport')
        df_phi_lower, _ = get_duals_as_vector(duals, 'eMinTransport')

        # define distance metric for clustering
        if 'lmp' in cluster_by:
            # get difference of lmps to slack bus for initial clustering
            LMP_difference = df_ptdf.T @ (df_phi_upper - df_phi_lower)
            distance_metric = LMP_difference
        elif 'ncp' in cluster_by:
            distance_metric = get_network_congestion_price_matrix(df_ptdf, df_phi_upper, df_phi_lower)
        elif 'ptdf' in cluster_by:
            distance_metric = df_ptdf.T
        else:
            exit('Can\'t identify distance metric!')

        # cluster nodes according to the chosen method
        if 'KMeans' in cluster_by:
            df_cluster = kmeans_clustering(num_of_clus, distance_metric, random_state, filter_nodes)

        elif 'KMedoids' in cluster_by:
            df_cluster = kmedoids_clustering(distance_metric, num_of_clus, random_state)

        elif 'SpectralClustering' in cluster_by:
            df_cluster = lmps_by_spectral_clustering(distance_metric, df_incidence_matrix_full, num_of_clus, random_state, filter_nodes)

        elif 'Anac' in cluster_by:
            if list_of_num_of_clust == [-1]:
                list_of_num_of_clust = [num_of_clus]

            df_cluster = ic.iterative_clustering(case_folder,
                                                 res_path,
                                                 data_folder,
                                                 cluster_by=cluster_by,
                                                 l_number_of_clusters=list_of_num_of_clust,
                                                 )

        else:
            raise ValueError('Cluster by not correctly defined!')

        return df_cluster

    # ================================== BEGIN GRID AGGREGATION ========================================================

    gla.check_if_folder_exists_and_create(os.path.join(data_folder, 'data'), agg_case, delete_existing=True)

    # load input data & data from model run
    duals, lines, df_ptdf, df_incidence_matrix_full = load_data(res_path, case_folder, data_folder)

    # load slack bus from original case
    slack_bus_original_grid = data.load_opf_parameter(case_folder, 'SlackBus', ',', data_folder)

    import time
    start_clustering = time.time()

    # cluster nodes
    df_cluster = cluster_nodes(res_path,
                               data_folder,
                               case_folder,
                               num_of_clus,
                               num_of_dec,
                               cluster_by=cluster_by,
                               list_of_num_of_clust=list_of_num_of_clust,
                               random_state=random_state,
                               )

    end_clustering = time.time()

    # return, if cluster is empty (ANAC) or not the defined number of clusters was found
    if df_cluster.empty or df_cluster.cluster.unique().size != num_of_clus:
        return df_cluster
    else:
        # export calculation time
        export_grid_partitioning_time(res_path, case_folder, cluster_by, num_of_clus, start_clustering, end_clustering)
        # export aggregated grid
        df_cluster = export_aggregated_grid(res_path, data_folder, case_folder, agg_case, df_cluster, lines, df_ptdf, slack_bus_original_grid)

        return df_cluster


if __name__ == '__main__':

    l_number_of_cluster = list(range(24, 0, -1))
    periods = 1
    number_of_cluster = 1
    cluster_method = 'lmpKMeans'
    tec_type = 'tecAss'
    results_folder = os.path.join('..', '..', 'GridAggregation', 'results')
    data_folder = os.path.join('..', '..', 'GridAggregation')
    case = 'IEEE_24_p19'


    agg_case = f'{case}_red_{cluster_method}_{number_of_cluster}_{tec_type}'

    grid_aggregation(results_folder, case, agg_case, number_of_cluster, cluster_method, data_folder,
                     periods=periods,
                     )

    gla.assign_clustered_case(data_folder, results_folder, case, agg_case, number_of_cluster, periods)


