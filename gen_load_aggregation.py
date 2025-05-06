#!interpreter [optional-arg]
# -*- coding: utf-8 -*-

__author__ = ["Benjamin Stöckl"]
__copyright__ = "Copyright 2023, Graz University of Technology"
__credits__ = ["Benjamin Stöckl"]
__license__ = "MIT"
__maintainer__ = "Benjamin Stöckl"
__status__ = "Development"

import os
import pandas as pd
import data
import shutil
from datetime import datetime


# ----------------------------------------------- HELPER FUNCTIONS -----------------------------------------------------
def drop_column_if_exists(df: pd.DataFrame, list_of_column_names: list):

    """
    drops columns from df if they exist

    :param df:
    :param list_of_column_names:
    :return:
    """

    for column_name in list_of_column_names:
        if column_name in df.columns:
            df.drop(columns=[column_name], inplace=True)

    return df


def check_if_folder_exists_and_create(path: str, folder='', sub_folder='', delete_existing=False):

    """
    Function checks if given folder exists or creates a new one.
    If the delete_existing flag is set to True, an existing folder will be deleted (moved to archive) first!

    :param path: of the given folder, must be string
    :param folder: name of the given folder, must be string
    :param sub_folder: opt. sub folder in folder
    :param delete_existing: boolean flag if existing folder should be archived first
    :return:
    """

    # check if delete_existing ist string (too many arguments in function call)
    if isinstance(delete_existing, str):
        exit('Warning: delete_existing flag is of type str! - might delete all directories!')

    if delete_existing:
        current_time = datetime.now()
        folder_time = str(current_time.year) + '-' + str(current_time.month) + '-' + str(current_time.day) + '_' + str(current_time.hour) + '-' + str(current_time.minute)
        if os.path.exists(os.path.join(path, folder)):
            shutil.move(os.path.join(path, folder), os.path.join(path, 'archive', folder_time, folder))
        if sub_folder != '' and os.path.exists(os.path.join(path, folder, sub_folder)):
            shutil.move(os.path.join(path, folder, sub_folder), os.path.join(path, 'archive', folder_time, folder, sub_folder))


    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))
    if not os.path.exists(os.path.join(path, folder)):
        os.makedirs(os.path.join(path, folder))
    if sub_folder != '' and not os.path.exists(os.path.join(path, folder, sub_folder)):
        os.makedirs(os.path.join(path, folder, sub_folder))


def load_file_or_gen_df(folder, file_name, columns, folder_name=''):
    '''
    if file exist load data in df, otherwise create empty df
    '''

    if not os.path.exists(os.path.join(folder, folder_name, file_name)):
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.read_csv(os.path.join(folder, folder_name, file_name), sep=';', decimal=',')

    return df


def __copy_csv_files__(input_folder, res_folder, input_scenario_name, output_scenario_name, file_name):

    df = pd.read_csv(os.path.join(input_folder, input_scenario_name, file_name), sep=';', decimal=',')
    df.to_csv(os.path.join(res_folder, output_scenario_name, file_name), index=False, sep=';', decimal=',')

# ----------------------------------------------------------------------------------------------------------------------


def export_import(df_import, out_folder, scenario_name):

    check_if_folder_exists_and_create(out_folder, scenario_name, 'imports')

    df_import.reset_index(inplace=True, drop=True)

    for node in df_import.columns:
        df_temp = pd.DataFrame(columns=['hour', 'imports'])
        df_temp['hour'] = range(1, len(df_import.index) + 1)
        df_temp['imports'] = df_import[node].astype(float).round(2)
        df_temp.to_csv(os.path.join(out_folder, scenario_name, 'imports', str(node) + '.csv'),
                       index=False, sep=';', decimal=',')


def demand_pyomo_export(df_demand: pd.DataFrame(), folder='res_agg_demand', folder_name='temp_demand'):
    """
    exports aggregated demand

    param: df_demand: column names are periods (integers); indices are bus IDs;

    """

    check_if_folder_exists_and_create(os.path.join(folder, 'data'), folder_name, 'loads')

    for i, row in df_demand.iterrows():
        df_temp = pd.DataFrame(columns=['hour', 'load'])
        df_temp['hour'] = row.index
        df_temp['load'] = row.values.astype(float)
        df_temp.to_csv(os.path.join(folder, 'data', folder_name, 'loads', str(i) + '.csv'),
                       index=False, sep=';', decimal=',')


def res_cf_pyomo_export(df_res_cf, output_folder='res_agg_gen', output_case='temp_gen', type_already_defined=False):

    check_if_folder_exists_and_create(os.path.join(output_folder, 'data'), output_case, 'cf')

    if type_already_defined:
        for i, row in df_res_cf.iterrows():
            df_temp = pd.DataFrame(columns=['hour', 'cf'])
            df_temp['hour'] = range(1, len(row)-1)
            df_temp['cf'] = row[2:].values.astype(float)
            df_temp.to_csv(os.path.join(output_folder, 'data', output_case, 'cf', str(row['aggNode']) + '.csv'),
                           index=False, sep=';', decimal=',')

    else:
        for i, row in df_res_cf.iterrows():
            df_temp = pd.DataFrame(columns=['hour', 'cf'])
            df_temp['hour'] = range(1, len(row)+1)
            df_temp['cf'] = row.values.astype(float)
            df_temp.to_csv(os.path.join(output_folder, 'data', output_case, 'cf', str(i) + '.csv'),
                           index=False, sep=';', decimal=',')


def get_clustered_buses(inp_fold, out_case, df_bus_info=pd.DataFrame()):
    """

    load 'bus_info.csv' and make mean from coordinates of cluster. Also make "incidence matrix" of

    return: incidence matrix of original and new buses

    """
    # load bus information and create new coordinates for new buses
    # Todo: bus info loaded from base case!!!
    if df_bus_info.empty:
        df_bus_info = pd.read_csv(os.path.join(inp_fold, 'data', out_case, 'bus_info.csv'), sep=';', decimal=',')
    # df_bus_info.set_index('bus', inplace=True)

    if 'BaseVolt' in df_bus_info.columns:
        df_bus_incidence = drop_column_if_exists(df_bus_info, ['BaseVolt', 'Country', 'Name', 'lat', 'long'])
        df_bus_info = drop_column_if_exists(df_bus_info, ['BaseVolt', 'Country', 'Name'])
    else:
        df_bus_incidence = df_bus_info.drop(columns={'lat', 'long'})

    if df_bus_info.empty:
        df_bus_info.drop(columns={'bus'}, inplace=True)
        df_bus_info = df_bus_info.groupby('cluster').mean()
        df_bus_info.reset_index(inplace=True, drop=False)
        df_bus_info.rename(columns={'cluster': 'bus'}, inplace=True)

    df_bus_info.to_csv(os.path.join(inp_fold, 'data', out_case, 'bus_info.csv'), sep=';', decimal=',', index=False)

    if df_bus_incidence.index.dtype == 'int64':
        df_bus_incidence['bus'] = df_bus_incidence['bus'].astype(str)

    return df_bus_incidence


def aggregate_demand(inp_fold, inp_case, res_fold, out_case, df_bus_incidence, num_of_clus, num_timesteps, single_node=False):

    # load demand data -> list with demand per bus
    l_demand = data.load_demand(inp_case, ',', inp_fold)
    d_dem = {}
    l_dataframes = [pd.DataFrame(data={'period': list(range(1, num_timesteps + 1))})]

    # get demand data as dataframe
    for i in l_demand:
        aux_b = i['bus']
        df = pd.DataFrame(data={aux_b: i['demand']['load'].values})
        l_dataframes.append(df)
    df_demand = pd.concat(l_dataframes, axis=1)  # concat demand to one DataFrame
    df_demand.set_index('period', inplace=True)
    df_demand = df_demand.transpose()
    df_demand.index.name = 'bus'

    if df_bus_incidence.index.name == 'bus':
        df_bus_incidence.reset_index(inplace=True, drop=False)

    # prepare df_bus_incidence: set datatype for buses to string and set index to 'bus'
    df_bus_incidence['bus'] = df_bus_incidence['bus'].astype(str)
    df_bus_incidence.set_index('bus', inplace=True)

    # assign clusters to buses and sum up demand for each cluster
    df_demand = df_demand.join(df_bus_incidence['cluster'], how='left')
    df_demand = df_demand.groupby('cluster').sum()
    df_demand.index.name = 'bus'
    df_demand.columns.name = 'period'

    demand_pyomo_export(df_demand, folder=inp_fold, folder_name=out_case)

    return True


def aggregate_import(inp_fold, inp_case, res_fold, out_case, df_bus_incidence, num_of_clus, num_of_ts):

    # check if imports exist else break
    if not os.path.exists(os.path.join(inp_fold, inp_case, 'imports')):
        return

    l_import = data.load_import(inp_case, decimal=',')

    # imports data
    aux_imp_data = pd.DataFrame(data={'period': list(range(1, num_of_ts + 1))})
    d_imp = {}

    df_temp_imp = [pd.DataFrame(data={'period': list(range(1, num_of_ts + 1))})]

    for i in l_import:
        aux_b = i['bus']
        df = pd.DataFrame(data={aux_b: i['imports']['imports'].values})
        df_temp_imp.append(df)

    aux_imp_data = pd.concat(df_temp_imp, axis=1)

    l_imp_nodes = list(aux_imp_data.keys())
    l_imp_nodes = l_imp_nodes[1:]

    for i, row in aux_imp_data.iterrows():
        for c in row.index[1:]:
            d_imp[c, row.loc['period']] = row[c]

    df_import = pd.DataFrame.from_dict(d_imp, orient='index')
    df_import.reset_index(inplace=True, drop=False)
    df_import[['bus', 'hour']] = pd.DataFrame(df_import['index'].tolist(), index=df_import.index)
    df_import.drop(columns={'index'}, inplace=True)
    df_import.set_index('bus', inplace=True)

    df_import = df_import.join(df_bus_incidence, how='left')
    df_import.dropna(inplace=True, how='any')
    df_import['cluster'] = df_import['cluster'].astype(int)
    df_import.drop(columns={'hour'}, inplace=True)
    df_import = df_import.groupby('cluster').sum()

    df_import = df_import.transpose()

    export_import(df_import, inp_fold, out_case)

    return True


def aggregate_nsp_cost(inp_fold, inp_case, res_fold, out_case, df_bus_incidence):

    if 'bus' in df_bus_incidence.columns:
        df_bus_incidence.set_index('bus', inplace=True)

    df_nsp_cost = pd.read_csv(os.path.join(inp_fold, 'data', inp_case, 'nsp_costs.csv'), sep=';', decimal=',')
    df_nsp_cost.set_index('zone', inplace=True)
    df_nsp_cost.index = df_nsp_cost.index.astype(str)
    if df_nsp_cost.index.name != 'bus':
        df_nsp_cost.index.name = 'bus'

    df_nsp_cost = df_nsp_cost.join(df_bus_incidence, how='left')
    df_nsp_cost = df_nsp_cost.groupby('cluster').mean()
    df_nsp_cost.reset_index(inplace=True, drop=False)
    df_nsp_cost['zone'] = df_nsp_cost['cluster'].astype(str)
    df_nsp_cost.drop(columns={'cluster'}, inplace=True)

    df_nsp_cost.to_csv(os.path.join(inp_fold, 'data', out_case, 'nsp_costs.csv'), sep=';', decimal=',', index=False)

    return True


def assign_clustered_case(inp_fold,
                          res_fold,
                          inp_case,
                          out_case,
                          num_of_clus,
                          periods,
                          df_bus_info=pd.DataFrame()
                          ):

    '''
    Assign generation units to aggregated nodes. If df_bus_info is provided us this, else load data from

    '''

    def assign_thermal_production(inp_fold, inp_case, res_fold, out_case, df_bus_incidence):

        if 'bus' in df_bus_incidence.columns:
            df_bus_incidence.set_index('bus', inplace=True)

        df_thermal = data.load_thermals(inp_case, decimal=',', folder=inp_fold)
        df_thermal.set_index('bus', inplace=True)

        df_thermal = df_thermal.join(df_bus_incidence, how='left')

        df_thermal_incidence = df_thermal[['cluster', 'unit']]

        df_thermal.rename(columns={'cluster': 'bus'}, inplace=True)
        df_thermal.to_csv(os.path.join(inp_fold, 'data', out_case, 'thermals.csv'), sep=';', decimal=',', index=False)

        return df_thermal_incidence


    def assign_res_production(inp_fold, inp_case, res_fold, out_case, df_bus_incidence):

        df_res = data.load_renewables(inp_case, decimal=',', folder=inp_fold)

        if df_res.empty:
            df_res_unit_cluster_bus = pd.DataFrame(columns=['unit', 'cluster', 'type'])
        else:
            if 'bus' in df_bus_incidence.columns:
                df_bus_incidence.set_index('bus', inplace=True)
            # df_bus_incidence.index = df_bus_incidence.index.astype(str)

            df_res.set_index('bus', inplace=True)
            df_res = df_res.join(df_bus_incidence, how='left')

            df_res_unit_cluster_bus = df_res[['unit', 'cluster', 'type']]
            df_res_unit_cluster_bus.reset_index(inplace=True, drop=False)

            df_res.rename(columns={'cluster': 'bus'}, inplace=True)

        df_res.to_csv(os.path.join(inp_fold, 'data', out_case, 'renewables.csv'), sep=';', decimal=',', index=False)

        return df_res_unit_cluster_bus

    # BEGIN assign_clustered_case() -----------------------------------------------------------------

    check_if_folder_exists_and_create(os.path.join(inp_fold, 'data'), out_case)
    check_if_folder_exists_and_create(os.path.join(inp_fold, 'data'), out_case, 'cf')

    if df_bus_info.empty:
        # get bus information and create new bus_info (new coordinates)
        df_bus_incidence = get_clustered_buses(inp_fold, out_case)
    else:
        # get bus incidence from bus info

        if 'BaseVolt' in df_bus_info.columns:

            if df_bus_info.index.name == 'bus':
                df_bus_info.reset_index(inplace=True, drop=False)

            df_bus_incidence = df_bus_info.drop(columns={'BaseVolt', 'Country', 'Name', 'lat', 'long'})
            df_bus_info.drop(columns={'BaseVolt', 'Country', 'Name', 'bus'}, inplace=True)
        else:
            df_bus_incidence = df_bus_info.drop(columns={'lat', 'long'})

            df_bus_info = df_bus_info.groupby('cluster').mean()
            df_bus_info.reset_index(inplace=True, drop=False)
            df_bus_info.rename(columns={'cluster': 'bus'}, inplace=True)
            df_bus_info.to_csv(os.path.join(inp_fold, 'data', out_case, 'bus_info.csv'), sep=';', decimal=',', index=False)


    # map demand to new buses and sum up
    aggregate_demand(inp_fold, inp_case, res_fold, out_case, df_bus_incidence.copy(), num_of_clus, periods)

    # map imports to new buses and sum up
    aggregate_import(inp_fold, inp_case, res_fold, out_case, df_bus_incidence.copy(), num_of_clus, periods)

    # assign thermal production to clustered nodes
    assign_thermal_production(inp_fold, inp_case, res_fold, out_case, df_bus_incidence.copy())

    assign_res_production(inp_fold, inp_case, res_fold, out_case, df_bus_incidence)

    aggregate_nsp_cost(inp_fold, inp_case, res_fold, out_case, df_bus_incidence)

    __copy_csv_files__(os.path.join(inp_fold, 'data'), os.path.join(inp_fold, 'data'), inp_case, out_case,
                       'gen_costs.csv')

    # copy all csv files in 'cf' folder
    file = os.listdir(os.path.join(inp_fold, 'data', inp_case, 'cf'))
    for f in file:
        __copy_csv_files__(os.path.join(inp_fold, 'data'), os.path.join(inp_fold, 'data'),
                           os.path.join(inp_case, 'cf'), os.path.join(out_case, 'cf'), f)


    return None


if __name__ == '__main__':
    single_node = False
    number_of_clusters = 4
    scenario_name = 'IEEE_24_p19'
    output_case = 'IEEE_24_p19_ncpKMeans_' + str(number_of_clusters) + '_tecAss'
    number_of_timesteps = 1
    begin_timestep = 1
    results_folder = os.path.join('..', '..', 'GridAggregation', 'results')
    data_folder = os.path.join('..', '..', 'GridAggregation')

    variable_cost_manipulation = 'sameDifference'  # original, sameDifference or increasing difference

    assign_clustered_case(data_folder, results_folder, scenario_name, output_case, number_of_clusters,
                          number_of_timesteps)