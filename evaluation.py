#!interpreter [optional-arg]
# -*- coding: utf-8 -*-

__author__ = ["Benjamin Stöckl"]
__copyright__ = "Copyright 2023, Graz University of Technology"
__credits__ = ["Benjamin Stöckl"]
__license__ = "MIT"
__maintainer__ = "Benjamin Stöckl"
__status__ = "Development"

import pandas as pd
import os
import ptdf_dc_opf as pdc
import gen_load_aggregation as gla
import matplotlib
import grid_partitioning as gp
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')


def evaluate_grid_aggregation(res_path, case_folder, ways_of_clustering, cluster_by='lmps', tec_type='',
                              plots_for_export=False):

    '''
    Generate Plots for the grid aggregation dependend in to the number of clusters



    '''

    plt.rcParams['font.family'] = 'serif'  # Use LaTeX's default "Computer Modern" font

    iee_colors = ['#022A34',  # dark blue
                  '#D58E00',  # orange
                  '#78BE73',  # green
                  '#F70146',  # red
                  '#206173',  # # blue
                  '#D58E00',  # ocher
                  '#60A2B3',  # light blue
                  ]

    d_results = {}

    # get teh ofv from the original case
    df_model_stats = pd.read_csv(os.path.join(res_path, case_folder, 'model_stats.csv'), sep=';', decimal=',')
    max_ov_value = df_model_stats.loc[df_model_stats['model_parameter'] == 'ofv', 'parameter_value'].values[0]

    df_mean_errors = pd.DataFrame(index=ways_of_clustering)

    for i in ways_of_clustering:
        # load file with model results
        folder_addition = '_red_' + i
        df = pd.read_csv(os.path.join(res_path, case_folder + folder_addition, 'aggregation_stats.csv'), sep=';', decimal=',')

        df['rel_ov_error'] = ((df['obj_func_value'] - max_ov_value) / max_ov_value)

        df.sort_values(by='number_of_agg_nodes', inplace=True)

        # calculate mean of the errors for all aggregations
        df_mean_errors.loc[i, 'mean_ofvre'] = df['rel_ov_error'].mean()
        df_mean_errors.loc[i, 'mean_flow_error'] = df['flow_rel_error'].mean()

        d_results[i] = df

    # export mean errors to csv & print latex to console
    df_mean_errors.to_csv(os.path.join(res_path, case_folder, 'mean_errors.csv'), sep=';', decimal=',')

    # create plots for the rel objective value error, max rel flow errors, max rel generation error
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))

    d_labels = {'lmpKMeans': 'LMP-KMeans',
                'lmpAnac': 'LMP-ANAC',
                'lmpSpectralClustering': 'LMP-SC',
                'ncpKMeans': 'NCP-KMeans',
                'ncpAnac': 'NCP-ANAC',
                }

    for index, clus_type in enumerate(ways_of_clustering):
        axs[0, 0].plot(d_results[clus_type]['number_of_agg_nodes'], d_results[clus_type]['rel_ov_error'], color=iee_colors[index], label=clus_type)
        axs[0, 1].plot(d_results[clus_type]['number_of_agg_nodes'], d_results[clus_type]['flow_rel_error'], color=iee_colors[index], label=clus_type)
        axs[1, 0].plot(d_results[clus_type]['number_of_agg_nodes'], d_results[clus_type]['flow_rel_error_max_dev'], color=iee_colors[index], label=clus_type)
        axs[1, 1].plot(d_results[clus_type]['number_of_agg_nodes'], d_results[clus_type]['ll_violation_error_max'], color=iee_colors[index], label=clus_type)

    fig.legend(labels=ways_of_clustering, loc='upper left')

    axs[0, 0].set_title('Objective Function Value of FM/AM')
    axs[0, 0].invert_xaxis()
    axs[0, 0].set_ylabel('Relative Error in p.u.')
    axs[0, 0].axhline(0, color='grey', linestyle='--', linewidth=1, label="_nolegend_")

    axs[0, 1].set_title('Mean Power Flow Error (Mapped agg. UD)')
    axs[0, 1].invert_xaxis()
    axs[0, 1].set_ylabel('Relative Error in p.u.')

    axs[1, 0].set_title('Max. Flow Error (Mapped agg. UD)')
    axs[1, 0].invert_xaxis()
    axs[1, 0].set_xlabel('Number of Aggregated Nodes')
    axs[1, 0].set_ylabel('Max Relative Error in p.u.')

    axs[1, 1].set_title('Max. Line Limit Violation (Mapped agg. UD)')
    axs[1, 1].invert_xaxis()
    axs[1, 1].set_xlabel('Number of Aggregated Nodes')
    axs[1, 1].set_ylabel('Max. Relative Error in p.u.')

    fig.show()
    # save fig as svg
    gla.check_if_folder_exists_and_create(res_path, case_folder)
    print('Save Figure to: ' + os.path.join(res_path, case_folder, 'aggregation_evaluation.svg'))
    fig.savefig(os.path.join(res_path, case_folder, 'aggregation_evaluation.svg'))

    if plots_for_export:
        matplotlib.rcParams.update({'font.size': 13})

        plt.axhline(0, color='grey', linestyle='--', linewidth=1, label="_nolegend_")
        for index, clus_type in enumerate(ways_of_clustering):
            plt.plot(d_results[clus_type]['number_of_agg_nodes'], d_results[clus_type]['rel_ov_error']*100,
                     '-', color=iee_colors[index], label=d_labels[clus_type], )

        plt.gca().invert_xaxis()
        plt.xlabel('Number of Nodes in AM')
        plt.ylabel(' Relative OFV error in %')
        plt.legend(loc='upper right')

        plt.savefig(os.path.join(res_path, case_folder, 'OFV_error.svg'))
        print('Save OFV Error to:' + os.path.join(res_path, case_folder, 'OFV_error.svg'))

        plt.show()
        # POWER FLOW ----------------------------------------------------------------
        # only plot the mean power flow error
        for index, clus_type in enumerate(ways_of_clustering):
            plt.plot(d_results[clus_type]['number_of_agg_nodes'], d_results[clus_type]['ll_violation_error_max']*100,
                     '-', color=iee_colors[index], label=d_labels[clus_type], )
            print('Mean max. relative line violation error for ' + clus_type + ': ' + str(d_results[clus_type]['ll_violation_error_max'].mean()))

        plt.gca().invert_xaxis()
        plt.xlabel('Number of Nodes in AM')
        plt.ylabel('Max. relative Line Limit Violation in %')
        plt.legend(loc='upper right')
        plt.ylim(0, 145)

        plt.savefig(os.path.join(res_path, case_folder, 'PowerFlow_error.svg'))
        print('Save OFV Error to:' + os.path.join(res_path, case_folder, 'PowerFlow_error.svg'))

        plt.show()

    return None


def run_grid_aggregation():

    # l_num_of_nodes = [5]
    l_num_of_nodes = list(range(24, 0, -1))
    tec_type = 'tecAss'  # tecAgg or tecAss
    input_case = 'IEEE_24_p19'

    # create folder outside the repository
    results_folder = os.path.join('..', '..', 'GridAggregation', 'results')
    data_folder = os.path.join('..', '..', 'GridAggregation')

    # Calculation Flags
    recalculate_cluster = False
    recalculate_aggregated_models = False

    recalculate_cluster = True
    recalculate_aggregated_models = True

    clustering_methods = [  # Power Tech
                            'lmpSpectralClustering',
                            'lmpKMeans',
                            'ncpKMeans',
                            'lmpAnac',
                            'ncpAnac',
                        ]
    case_name = input_case

    decimal_places = -1
    periods = 1

    for cluster_type in clustering_methods:                         # determine grid aggregations for all clustering methods
        for i in range(1, 2):                                       # check stability of clustering methods
            for number_of_clusters in l_num_of_nodes:               # determine grid aggregations for all levels of reduction
                # define aggregated case
                agg_case = f'{case_name}_red_{cluster_type}_{number_of_clusters}_{tec_type}'
                # if recalculate_cluster determine grid partitioning ang define aggregated models
                if recalculate_cluster:
                    df_cluster = gp.grid_aggregation(results_folder,
                                                     input_case,
                                                     agg_case,
                                                     number_of_clusters,
                                                     cluster_by=cluster_type,
                                                     num_of_dec=decimal_places,
                                                     data_folder=data_folder,
                                                     periods=periods,
                                                     )

                    if df_cluster.empty:  # in case of ANAC
                        for num_of_clus in l_num_of_nodes:
                            agg_case = f'{case_name}_red_{cluster_type}_{num_of_clus}_{tec_type}'
                            # create generation and loads for the aggregated case
                            gla.assign_clustered_case(data_folder,
                                                      results_folder,
                                                      input_case,
                                                      agg_case,
                                                      num_of_clus,
                                                      periods,
                                                      )
                        break
                    else:
                        # create generation and load
                        gla.assign_clustered_case(data_folder, results_folder, input_case, agg_case, number_of_clusters,
                                                  periods)

            for number_of_clusters in l_num_of_nodes:
                agg_case = f'{case_name}_red_{cluster_type}_{number_of_clusters}_{tec_type}'

                if recalculate_aggregated_models:

                    print('run agg model: ', agg_case, results_folder, input_case, str(number_of_clusters))
                    # run agg model
                    pdc.run_model(agg_case,
                                  periods,
                                  res_folder=results_folder,
                                  data_folder=data_folder,
                                  original_case=input_case,
                                  number_of_clusters=number_of_clusters,
                                  activate_NSP=False,
                                  )


    evaluate_grid_aggregation(results_folder, input_case,
                              ways_of_clustering=clustering_methods,
                              tec_type=tec_type)


if __name__ == '__main__':

    run_grid_aggregation()







