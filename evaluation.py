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
import ptdf_dc_opf as pdc
import gen_load_aggregation as gla
from matplotlib import pyplot as plt, colors
from sklearn.cluster import KMeans, SpectralClustering
from sklearn_extra.cluster import KMedoids
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

    return df, df.to_numpy()


def export_assigned_clusters_to_bus_info(case_folder: str, df_cluster: pd.DataFrame):

    df = df_cluster['cluster']
    df_bus_info = pd.read_csv(os.path.join('data', case_folder, 'bus_info.csv'), sep=';', decimal=',')
    df_bus_info = df_bus_info.astype({'bus': str})
    df_bus_info.set_index('bus', inplace=True, drop=True)

    # if column cluster already existing in df_bus_info drop to assign new
    if 'cluster' in df_bus_info.columns:
        df_bus_info.drop(columns='cluster', inplace=True)
    df_bus_info = df_bus_info.join(df, how='left')

    # df_bus_info['cluster'] = df_cluster['cluster'].tolist()
    if 'bus' in df_bus_info.columns:
        df_bus_info.drop(columns={'bus'}, inplace=True)

    df_bus_info.to_csv(os.path.join('data', case_folder, 'bus_info.csv'), sep=';', decimal=',', index=True)

    return df_bus_info


def get_incidence_matrix(lines, df_ptdf, slack_bus="", undirected=False):

    cell_value = 1 if undirected else -1

    # get incidence matrix C (L_f x B_f-1) of full network
    C = np.zeros((len(lines), len(df_ptdf.columns)), dtype=int)  # create array with zeros and size LxN
    for i in range(0, len(lines)):  # set entry to 1 if bus is in index of line
        # print(i)

        C[i, df_ptdf.columns.get_loc(lines.loc[i, 'bus1'])] = 1
        C[i, df_ptdf.columns.get_loc(lines.loc[i, 'bus2'])] = cell_value

    df_C = pd.DataFrame(data=C, columns=df_ptdf.columns, index=[lines['bus1'], lines['bus2']])

    # drop slack bus if slack bus is defined
    if slack_bus != "":
        df_C.drop(columns=slack_bus, inplace=True)  # drop slack bus -> C matrix has dimension L_f x B_f-1

    C = df_C.to_numpy()

    return C, df_C


def get_line_limits(zone_lines, num_of_cluster):

    # get lines with different clusters (zone_lines) -> are defining the max line limit
    df_line_limits = zone_lines[['Pmax', 'cluster1', 'cluster2']]
    df_line_limits = df_line_limits.groupby(['cluster1', 'cluster2']).sum()

    df_line_limits.reset_index(inplace=True, drop=False)
    df_line_limits.rename(columns={'cluster1': 'bus1', 'cluster2': 'bus2'}, inplace=True)
    df_line_limits = df_line_limits.astype({'bus1': 'int', 'bus2': 'int'})
    df_line_limits[['Xline', 'Country', 'line']] = 0

    return df_line_limits


def assign_lines_to_clusters(lines, df_cluster):
    for i in range(0, len(lines)):
        lines.loc[i, 'cluster1'] = int(df_cluster.loc[lines.loc[i, 'bus1'], 'cluster'])  # assign clusters to lines
        lines.loc[i, 'cluster2'] = int(df_cluster.loc[lines.loc[i, 'bus2'], 'cluster'])

    # drop lines with the same cluster
    # lines = lines[lines['cluster1'] != lines['cluster2']]

    return lines


def load_data(res_path, case_folder, slack_bus=""):

    duals = pd.read_csv(os.path.join(res_path, case_folder, 'duals.csv'), sep=';', decimal=',')
    lines = pd.read_csv(os.path.join('data', case_folder, 'lines.csv'), sep=';', decimal=',')

    lines = lines.astype({'bus1': 'str', 'bus2': 'str'})

    df_ptdf = pd.read_csv(os.path.join('data', case_folder, 'PTDF.csv'), sep=';', decimal=',')
    df_ptdf.rename(columns={'Unnamed: 0': 'bus1', 'Unnamed: 1': 'bus2'}, inplace=True)
    df_ptdf = df_ptdf.astype({'bus1': 'str', 'bus2': 'str'})
    df_ptdf.set_index(['bus1', 'bus2'], inplace=True)

    # TODO: get slack bus and delete it from ptdf
    df_ptdf.sort_index(axis=1, inplace=True)  # sort columns by alphabetical order

    # get full incidence matrix without slack bus column! dimension: L_f x B_f-1
    incidence_matrix_full, df_incidence_matrix_full = get_incidence_matrix(lines, df_ptdf, undirected=False)
    df_ptdf_with_slack = df_ptdf.copy()
    # df_ptdf.drop(columns=slack_bus, inplace=True)
    ptdf = df_ptdf.to_numpy()


    return duals, lines, df_ptdf, ptdf, df_incidence_matrix_full


def get_node_zone_incidence_matrix(df_cluster, num_of_clus):

    df_nzi = pd.DataFrame(0, columns=df_cluster.index, index=range(1, num_of_clus + 1))

    for i in range(0, len(df_cluster)):
        df_nzi.loc[df_cluster.loc[df_cluster.index[i], 'cluster'], df_cluster.index[i]] = 1


    # get zone incidence matrix
    array = np.zeros((num_of_clus, len(df_cluster)), dtype=int)
    for i in range(0, len(df_cluster)):
        array[int(df_cluster.loc[df_cluster.index[i], 'cluster'] - 1), i] = 1

    return array, df_nzi


def get_reduced_ptdf_matrix(ptdf, T_f, df_T_f, T_bz, df_T_bz=None, df_ptdf=pd.DataFrame(), slack_bus_cluster='', use_df=False):
    # try to drop slack bus from clusters

    if use_df:

        df_aux = df_T_bz.T @ np.linalg.inv(df_T_bz @ df_T_bz.T)
        df_aux.columns = df_T_bz.index

        df_red_ptdf = df_T_f @ df_ptdf @ df_aux
        red_ptdf = df_red_ptdf.to_numpy()

        if df_red_ptdf.index.names == ['cluster1', 'cluster2']:
            df_red_ptdf.index.names = ['bus1', 'bus2']

    # T_bz = np.delete(T_bz, slack_bus_cluster - 1, axis=0)
    else:
        # H_red = T_f H_f T_bz^T (T_bz T_bz^T)^-1
        red_ptdf = T_f @ ptdf @ T_bz.T @ np.linalg.inv(T_bz @ T_bz.T)
        # red_ptdf = np.delete(red_ptdf, slack_bus_cluster, axis=1)

        df_red_ptdf = pd.DataFrame(data=red_ptdf, index=df_T_f.index)
        df_red_ptdf.index.names = ['bus1', 'bus2']
        df_red_ptdf.columns = range(1, len(df_red_ptdf.columns) + 1)

    return df_red_ptdf, red_ptdf


def get_node_cluster_information_matrix(df_ptdf, df_phi_upper, df_phi_lower):
    # make vector for phi and filter for empty entries
    df_phi = df_phi_upper - df_phi_lower
    df_filt_phi = df_phi[df_phi != 0]
    df_filt_phi.dropna(inplace=True)

    # set datatype of Multiindex to string
    df_ptdf.index = df_ptdf.index.set_levels([
        df_ptdf.index.levels[0].astype(str),
        df_ptdf.index.levels[1].astype(str)
    ])

    # reduce ptdf to lines with dual variables
    df_ptdf_filt = df_ptdf.loc[df_filt_phi.index, :]

    # Use squeeze() to convert the DataFrame to a Series
    df = df_ptdf_filt.mul(df_filt_phi.squeeze(), axis=0)

    return df


def grid_aggregation(res_path, case_folder, num_of_clus, filter_nodes=[], cluster_by='lmps',
                     addition_out_folder='_red', tec_type='', num_of_dec=-1, case_comment=''):


    def cluster_nodes(res_path, case_folder, num_of_clus, num_of_dec, filter_nodes=[], cluster_by='lmps'):


        def cluster_by_lmps(duals, num_of_clus, filter_nodes=[]):

            # get lmps
            df = duals.loc[:, (i.startswith('eBalance') for i in duals.columns.values)].T
            df['bus'] = [i.split('_')[2] for i in df.index.values]
            df.set_index('bus', inplace=True)

            if len(filter_nodes) != 0:
                df.drop(index=filter_nodes, inplace=True)

            df.rename(columns={0: 'lmp'}, inplace=True)
            df['lmp'] = df['lmp'].astype(float)
            df = df.round(8)

            kmeans = KMeans(n_clusters=num_of_clus).fit(df)
            df['cluster'] = kmeans.labels_ + 1

            return df


        def cluster_by_spectral_clustering(duals, df_incidence_matrix, num_of_clus, filter_nodes=[]):

            # get lmps
            df = duals.loc[:, (i.startswith('eBalance') for i in duals.columns.values)].T
            df['bus'] = [i.split('_')[2] for i in df.index.values]
            df.set_index('bus', inplace=True)

            if len(filter_nodes) != 0:
                df.drop(index=filter_nodes, inplace=True)

            df.rename(columns={0: 'lmp'}, inplace=True)
            df['lmp'] = df['lmp'].astype(float)

            # make sure df_incidence matrix is undirected
            df_incidence_matrix = abs(df_incidence_matrix)

            LMP_diff = 1/abs(df_incidence_matrix @ df)

            aux = df_incidence_matrix.T @ np.diag(LMP_diff.squeeze())  # (B x L) * (L x L) = (B x L)
            aux.columns = LMP_diff.index

            df_laplacia = aux @ df_incidence_matrix  # (B x L) * (L x B) = (B x B)

            spectral_clustering = SpectralClustering(n_clusters=num_of_clus,
                                                     ).fit(df_laplacia)

            df['cluster'] = spectral_clustering.labels_ + 1
            df['cluster'] = df['cluster'].astype(int)

            return df


        def nci_by_kmedoids(df_ptdf, df_phi_upper, df_phi_lower, num_of_clus):

            df_node_cluster_information = get_node_cluster_information_matrix(df_ptdf, df_phi_upper, df_phi_lower).T

            # calculate kmedoids with manhattan distance (l1 norm)
            if num_of_clus > 1:
                kmedoids = KMedoids(n_clusters=num_of_clus, metric='manhattan', max_iter=1000,
                                    method='pam').fit(df_node_cluster_information)
            else:
                return pd.DataFrame()

            df_node_cluster_information['cluster'] = kmedoids.labels_ + 1

            return df_node_cluster_information


        # get vectors for duals
        df_phi_upper, phi_upper = get_duals_as_vector(duals, 'eMaxTransport')
        df_phi_lower, phi_lower = get_duals_as_vector(duals, 'eMinTransport')

        # cluster nodes according to the chosen method
        if 'lmps' in cluster_by:
            df_cluster = cluster_by_lmps(duals, num_of_clus, filter_nodes)
        elif 'SpectralClustering' in cluster_by:
            df_cluster = cluster_by_spectral_clustering(duals, df_incidence_matrix_full, num_of_clus, filter_nodes)
        elif 'KMedoids' in cluster_by:
            df_cluster = nci_by_kmedoids(df_ptdf, df_phi_upper, df_phi_lower, num_of_clus)
        else:
            raise ValueError('Cluster by not correctly defined!')

        if df_cluster.empty:
            return df_cluster, -1
        else:
            # export to bus_info.csv of input folder (map plotting)
            df_bus_info = export_assigned_clusters_to_bus_info(case_folder, df_cluster)
            df_bus_info.to_csv(os.path.join('data', case_folder + '_red_' + cluster_by + '_' + str(num_of_clus)
                                            + '_' + tec_type, 'cluster_info.csv'),
                                sep=';', decimal=',', index=True)

            return df_cluster, num_of_clus


    def determine_reduced_ptdf(df_clus, lines, df_ptdf, addition_out_folder='_ptdf_red', tec_type=''):

        # slack_bus_cluster = df_cluster.loc[slack_bus, 'cluster']
        # df_cluster.drop(index=slack_bus, inplace=True)

        zone_incidence_matrix, df_zone_incidence_matrix = get_node_zone_incidence_matrix(df_clus, num_of_clus)

        zone_lines = lines.loc[lines['cluster1'] != lines['cluster2']]
        zone_lines = zone_lines.drop(columns=['line'])

        # order zone lines to ascending order (by number of cluster)
        for i, row in zone_lines.iterrows():
            if row['cluster1'] > row['cluster2']:
                zone_lines.loc[i, 'cluster1'], zone_lines.loc[i, 'cluster2'] = \
                    zone_lines.loc[i, 'cluster2'], zone_lines.loc[i, 'cluster1']
                zone_lines.loc[i, 'bus1'], zone_lines.loc[i, 'bus2'] = \
                    zone_lines.loc[i, 'bus2'], zone_lines.loc[i, 'bus1']

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
        df_inter_zones_lines.to_csv(os.path.join('data', case_folder + addition_out_folder + '_' + str(num_of_clus)
                                                 + '_' + tec_type,
                                                 'line_incidence.csv'), sep=';', decimal=',', index=True)


        # get reduced ptdf matrix and save to input folder
        df_red_ptdf, red_ptdf = get_reduced_ptdf_matrix(ptdf, inter_zones_lines, df_inter_zones_lines,
                                                        zone_incidence_matrix,
                                                        df_ptdf=df_ptdf,
                                                        df_T_bz=df_zone_incidence_matrix,
                                                        use_df=True,
                                                        )

        gla.check_if_folder_exists_and_create('data', case_folder + addition_out_folder + '_' + str(num_of_clus)
                                              + '_' + tec_type)
        df_red_ptdf.to_csv(os.path.join('data', case_folder + addition_out_folder + '_' + str(num_of_clus)
                                        + '_' + tec_type, 'PTDF.csv'),
                           sep=';', decimal=',')


        return df_red_ptdf, zone_lines


    # ================================== BEGIN GRID AGGREGATION ========================================================


    addition_out_folder = '_red_' + cluster_by
    input_case = '_'.join(case_folder.split('_')[:3])

    # check if folder exists, if so -> delete and create new one
    gla.check_if_folder_exists_and_create('data', input_case + addition_out_folder + '_' + str(num_of_clus)
                                          + '_' + tec_type
                                          # + ('_' + case_comment if case_comment != '' else '')
                                          , delete_existing=True)

    # load input data & data from model run
    duals, lines, df_ptdf, ptdf, df_incidence_matrix_full = load_data(res_path, case_folder)

    # determine network partitioning
    df_cluster, num_of_clus = cluster_nodes(res_path, case_folder, num_of_clus, num_of_dec, cluster_by=cluster_by)

    if df_cluster.empty:
        return num_of_clus
    else:
        # define nodes of lines to clusters
        lines = assign_lines_to_clusters(lines, df_cluster)

        # determine reduced ptdf matrix
        df_red_ptdf, zone_lines = determine_reduced_ptdf(df_cluster, lines, df_ptdf, addition_out_folder, tec_type=tec_type)

        # define line capacities for lines between zones
        df_line_limits = get_line_limits(zone_lines, num_of_clus)
        df_line_limits.to_csv(os.path.join('data', case_folder + addition_out_folder + '_' + str(num_of_clus) + '_' + tec_type, 'lines.csv'),
                              sep=';',
                              decimal=',',
                              index=False)

        return num_of_clus


def evaluate_grid_aggregation(res_path, case_folder, ways_of_clustering, cluster_by='lmps', tec_type=''):

    '''
    Generate Plots for the grid aggregation dependend in to the number of clusters



    '''

    plt.rcParams['font.family'] = 'serif'  # Use LaTeX's default "Computer Modern" font

    iee_colors = ['#022A34',  # dark blue
                  '#78BE73',  # green
                  '#206173',  # # blue
                  '#D58E00',  # orange
                  '#F70146',  # red
                  ]

    d_results = {}

    # get the ofv from the original case
    df_model_stats = pd.read_csv(os.path.join(res_path, case_folder, 'model_stats.csv'), sep=';', decimal=',')
    max_ov_value = df_model_stats.loc[df_model_stats['model_parameter'] == 'ofv', 'parameter_value'].values[0]

    df_mean_errors = pd.DataFrame(index=ways_of_clustering)

    # load results
    for i in ways_of_clustering:
        # load file with model results
        folder_addition = '_red_' + i
        df = pd.read_csv(os.path.join(res_path, case_folder + folder_addition, 'aggregation_stats.csv'), sep=';', decimal=',')

        # get the rel error of ov to the case with most number of clusters
        # max_num_clusters = max(df['number_of_agg_nodes'])
        # max_ov_value = (df.loc[df['number_of_agg_nodes'] == max_num_clusters, 'obj_func_value']).values[
        #     0]

        df['rel_ov_error'] = ((df['obj_func_value'] - max_ov_value) / max_ov_value)

        df.sort_values(by='number_of_agg_nodes', inplace=True)

        # calculate mean of the errors for all aggregations
        df_mean_errors.loc[i, 'mean_ofvre'] = df['rel_ov_error'].mean()
        df_mean_errors.loc[i, 'mean_flow_error'] = df['flow_rel_error'].mean()

        d_results[i] = df


    d_labels = {
                'lmps': 'LMP-KMeans',
                'itClusterLMP': 'LMP-ANAC',
                'lmpSpectralClustering': 'LMP-SC',
                'KMedoids': 'NCP-KMedoids',
                'itClusterNCI': 'NCP-ANAC',
                }

    l_labels = [
                'LMP-KMeans',
                'LMP-SC',
                'LMP-ANAC',
                'NCP-KMedoids',
                'NCP-ANAC',
                ]

    # plot the objective function value error
    for index, clus_type in enumerate(ways_of_clustering):
        plt.plot(d_results[clus_type]['number_of_agg_nodes'], d_results[clus_type]['rel_ov_error']*100,
                 '-', color=iee_colors[index], label=clus_type, )

    plt.gca().invert_xaxis()
    plt.xlabel('Number of Nodes in AM')
    plt.ylabel(' Relative OFV error in %')
    plt.legend(labels=l_labels, loc='upper right')
    plt.grid()

    plt.savefig(os.path.join(res_path, case_folder, 'OFV_error.svg'))
    print('Save OFV Error to:' + os.path.join(res_path, case_folder, 'OFV_error.svg'))

    plt.show()

    # plot the mean power flow error
    for index, clus_type in enumerate(ways_of_clustering):
        plt.plot(d_results[clus_type]['number_of_agg_nodes'], d_results[clus_type]['ll_violation_error_max']*100,
                 '-', color=iee_colors[index], label=d_labels[clus_type], )

    plt.gca().invert_xaxis()
    plt.xlabel('Number of Nodes in AM')
    plt.ylabel('Relative Power Flow Error in %')
    plt.legend(labels=l_labels, loc='upper right')
    plt.grid()

    plt.savefig(os.path.join(res_path, case_folder, 'PowerFlow_error.svg'))
    print('Save OFV Error to:' + os.path.join(res_path, case_folder, 'PowerFlow_error.svg'))

    plt.show()

    return None


def run_grid_aggregation():

    # IEEE 24 bus case
    # l_num_of_nodes = [5]  # determine AM for specific number of nodes
    l_num_of_nodes = list(range(23, 0, -1))   # determine AM for all number of nodes

    recalculate_cluster = False  # NOTE: does not recalculate clusters for ANAC!
    recalculate_aggregated_models = False

    recalculate_cluster = True
    recalculate_aggregated_models = True


    clustering_methods = [
                          'lmps',                   # LMPs - k-means
                          'lmpSpectralClustering',  # LMPs - Spectral Clustering
                          'itClusterLMP',           # LMPs - Adjacent Hierarchical Clustering
                          'KMedoids',            # NCP - k-medoids
                          'itClusterNCI',           # NCP - Adjacent Hierarchical Clustering
                         ]

    # clustering_methods = ['itClusterNCI', 'itClusterLMP']

    results_path = os.path.join('.', 'results')
    tec_type = 'tecAss'  # tecAss

    input_case = 'IEEE_24_p19'
    case_name = input_case
    decimal_places = -1

    for cluster_type in clustering_methods:
        for number_of_clusters in l_num_of_nodes:

            agg_case = case_name + '_red_' + cluster_type + '_' + str(number_of_clusters) + '_' + tec_type

            if recalculate_cluster and not 'itCluster' in cluster_type:

                number_of_clusters = grid_aggregation(results_path, input_case, number_of_clusters,
                                                      cluster_by=cluster_type,
                                                      tec_type=tec_type,
                                                      num_of_dec=decimal_places
                                                      )

                if number_of_clusters == -1:
                    break
                else:
                    # create generation and load
                    gla.assign_clustered_case('data', results_path, input_case, agg_case, number_of_clusters)

            if recalculate_aggregated_models and number_of_clusters != -1:

                print('run agg model: ', agg_case, results_path, input_case, str(number_of_clusters))
                # run agg model
                pdc.run_model(agg_case,
                              1,
                              model_type='ptdfDcOpf',
                              res_folder=results_path,
                              original_case=input_case,
                              number_of_clusters=number_of_clusters,
                              )


    evaluate_grid_aggregation(results_path,
                              input_case,
                              ways_of_clustering=clustering_methods,
                              tec_type=tec_type)



if __name__ == '__main__':

    run_grid_aggregation()








