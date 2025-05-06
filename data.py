#!interpreter [optional-arg]
# -*- coding: utf-8 -*-

__author__ = ["David Cardona-Vasquez", "Benjamin Stöckl"]
__copyright__ = "Copyright 2023, Graz University of Technology"
__credits__ = ["David Cardona-Vasquez", "Benjamin Stöckl"]
__license__ = "MIT"
__maintainer__ = "Benjamin Stöckl"
__status__ = "Development"

import glob
import numpy as np
import os
import pandas as pd
import shutil
from natsort import natsorted



def get_vGen_warmstarting_data(results_folder: str, warmstarting_case: str):
    """
    Load generation results from aggregated model results for warmstarting the full model

    :param res_folder:
    :param warmstarting_case:
    :return: aggregated_unit_generation_dict: dict in format {(unit, period): generation}
    """

    if warmstarting_case == '':
        return None
    else:
        # load generation results from aggregated model
        aggregated_unit_generation = pd.read_csv(os.path.join(results_folder, warmstarting_case, 'vGen.csv'), sep=';',
                                                 decimal=',', index_col='unit')
        aggregated_unit_generation.drop(columns={'bus', 'type'}, inplace=True)

        # create dict with generation results from dataframe in format {(unit, period): generation}
        aggregated_unit_generation_dict = {}
        for i, row in aggregated_unit_generation.iterrows():
            aggregated_unit_generation_dict[(i, row['period'])] = row['value']

        return {'vGen': aggregated_unit_generation_dict}


def load_cf(case: str, decimal='.', folder=""):
    """

    :param case:
    :return:
    """

    l_renewable = []
    gen_files = glob.glob(os.path.join(folder, 'data', case, 'cf', '*.csv'))

    for f in gen_files:
        gen_data = pd.read_csv(f, sep=';', decimal=decimal)
        gen_data = gen_data.astype({'hour': 'int64', 'cf': 'float64'})
        gen_name = str(os.path.basename(f).split('.')[0])
        l_renewable.append({'generator': gen_name, 'cf': gen_data})

    return l_renewable


def load_demand(case: str, decimal='.', folder=""):
    """

    :param case:
    :return:
    """

    l_demand = []
    demand_files = glob.glob(os.path.join(folder, 'data', case, 'loads', '*.csv'))

    for f in demand_files:
        load_data = pd.read_csv(f, sep=';', decimal=decimal)
        load_data.load = load_data.load
        load_data = load_data.astype({'hour': 'int64', 'load': 'float64'})
        bus_name = str(os.path.basename(f).split('.')[0])
        l_demand.append({'bus': bus_name, 'demand': load_data})

    return l_demand


def load_import(case: str, decimal='.', folder=""):

    l_import = []
    import_files = glob.glob(os.path.join(folder, 'data', case, 'imports', '*.csv'))

    for f in import_files:
        import_data = pd.read_csv(f, sep=';', decimal=decimal)
        import_data.imports = import_data.imports
        bus_name = str(os.path.basename(f).split('.')[0])
        l_import.append({'bus': bus_name, 'imports': import_data})

    return l_import


def load_lines(case: str, decimal='.', folder=""):
    """

    :param case:
    :return:
    """

    line_data = pd.read_csv(os.path.join(folder, 'data', case, 'lines.csv'), sep=';', decimal=decimal)

    if 'line' not in line_data.columns:
        line_data['line'] = str(line_data['bus1']) + '-' + str(line_data['bus2'])

    if 'Xline' in line_data.columns:
        line_data = line_data.astype({'bus1': 'str', 'bus2': 'str', 'Pmax': 'float64', 'Xline': 'float64',
                          'line': 'str'})
    else:
        line_data = line_data.astype({'bus1': 'str', 'bus2': 'str', 'Pmax': 'float64', 'line': 'str'})

    return line_data


def load_opf_parameters(case: str, decimal='.', folder=""):
    """

    :param case:
    :return:
    """

    #l_parameters = []
    df_parameters = pd.read_csv(os.path.join(folder, 'data', case, 'opf_parameters.csv'), index_col='parameter_name',
                                sep=';', decimal=decimal)
    #l_parameters = df_parameters.to_dict(orient='list')

    return df_parameters


def load_renewables(case: str, decimal='.', folder=""):
    """

    :param case:
    :return:
    """
    renewables_data = pd.read_csv(os.path.join(folder, 'data', case, 'renewables.csv'), sep=';', decimal=decimal)

    if renewables_data.dtypes['bus'] == 'float64':
        renewables_data['bus'] = renewables_data['bus'].astype('int64')

    renewables_data = renewables_data.astype({'min': 'float64', 'max': 'float64', 'bus': 'str', 'type': 'str',
                            'unit': 'str'})

    return renewables_data


def load_thermals(case: str, decimal='.', folder=""):
    """

    :param case:
    :return:
    """

    thermals_data = pd.read_csv(os.path.join(folder, 'data', case, 'thermals.csv'), sep=';', decimal=decimal)

    if thermals_data.dtypes['bus'] == 'float64':
        thermals_data['bus'] = thermals_data['bus'].astype('int64')

    thermals_data = thermals_data.astype({'min': 'float64', 'max': 'float64', 'rmp_up': 'float64', 'rmp_dn': 'float64',
                                'bus': 'str'})

    return thermals_data


def load_ptdf(case: str, decimal='.', folder=''):

    ptdf = pd.read_csv(os.path.join(folder, 'data', case, 'ptdf.csv'), sep=';', decimal=decimal)
    ptdf = ptdf.astype({'bus1': 'str', 'bus2': 'str'})
    ptdf.set_index(['bus1', 'bus2'], inplace=True)

    # sort ptdf by natsorted indices
    ptdf = ptdf.reindex(natsorted(ptdf.index))

    #reorder columns by natsorted columns
    ptdf = ptdf[natsorted(ptdf.columns)]

    return ptdf


def load_opf_parameter(case: str, parameter: str, decimal='.', folder=""):

    df_opf_parameters = pd.read_csv(os.path.join(folder, 'data', case, 'opf_parameters.csv'), sep=';', decimal=',')

    if parameter == 'SlackBus':
        value = str(df_opf_parameters.loc[df_opf_parameters['parameter_name'] == parameter, 'parameter_value'].values[0])
    else:
        value = df_opf_parameters.loc[df_opf_parameters['parameter_name'] == parameter, 'parameter_value'].values[0]

    return value


def __lines_to_dict__(df: pd.DataFrame):
    d = {}
    x = {}

    for i, row in df.iterrows():
        #for c in row.index[1:]:
        d[str(row['bus1']), str(row['bus2'])] = row['Pmax']
        x[str(row['bus1']), str(row['bus2'])] = row['Xline']

    return d, x


def __buses_from_lines__(df: pd.DataFrame):
    d = []

    for i, row in df.iterrows():
        if str(row['bus1']) not in d:
            d.append(row['bus1'])
        if str(row['bus2']) not in d:
            d.append(row['bus2'])

    return d


def export_cf(generator: str, cf: pd.DataFrame, case: str, decimal='.', folder=""):
    """

    :param generator:
    :param cf:
    :param case:
    :return:
    """

    cf.to_csv(os.path.join(folder, 'data', case, 'cf', f'{generator}.csv'), sep=';', decimal=decimal, index=False)

    return None


def export_demand(bus: str, demand: pd.DataFrame, case: str, decimal='.', folder=""):
    """

    :param bus:
    :param demand:
    :param case:
    :return:
    """

    demand.to_csv(os.path.join(folder, 'data', case, 'loads', f'{bus}.csv'), sep=';', decimal=decimal, index=False)

    return None


def export_additional_demand(df_vDemAdd, case, folder=""):

    """
    Export additional demand data to csv file

    :param df_vDemAdd:
    :param case:
    :param res_folder:
    :return:
    """

    # export additional demand data
    df_vDemAdd.to_csv(os.path.join(folder, case, 'vDemAdd.csv'), sep=';', decimal=',', index=False)

    return None


def export_lines(lines: pd.DataFrame, case: str, decimal='.', folder=""):
    """

    :param lines:
    :param case:
    :return:
    """
    if list(lines.index.names) != ['bus1', 'bus2']:
        lines.set_index(['bus1', 'bus2'], inplace=True)

    # sort lines by natsorted indices
    lines = lines.reindex(natsorted(lines.index))

    #reorder columns by natsorted columns
    lines = lines[natsorted(lines.columns)]

    lines.to_csv(os.path.join(folder, 'data', case, 'lines.csv'), sep=';', decimal=decimal, index=True)

    return None


def export_network(network: pd.DataFrame, case: str, decimal='.'):
    """

    :param network:
    :param case:
    :return:
    """

    network.to_csv(os.path.join('data', case, 'network.csv'), sep=';', decimal=decimal, index=False)

    return None


def export_renewables(renewables: pd.DataFrame, case: str, decimal='.'):
    """

    :param renewables:
    :param case:
    :return:
    """

    renewables.to_csv(os.path.join('data', case, 'renewables.csv'), sep=';', decimal=decimal, index=False)

    return None


def export_thermals(thermals: pd.DataFrame, case: str, decimal='.'):
    """

    :param thermals:
    :param case:
    :return:
    """

    thermals.to_csv(os.path.join('data', case, 'thermals.csv'), sep=';', decimal=decimal, index=False)

    return None


def export_incidence_matrix(C, case, decimal='.', folder=''):

    # sort C by natsorted indices
    C = C.reindex(natsorted(C.index))

    # reorder columns by natsorted columns
    C = C[natsorted(C.columns)]

    # export C_red
    C.to_csv(os.path.join(folder, 'data', case, 'C_sb_adj.csv'), decimal=decimal, index=False, header=False, sep=';')

    return


def export_line_incidence(line_incidence: pd.DataFrame, case: str, decimal='.', folder=""):
    """

    :param line_incidence:
    :param case:
    :return:
    """

    # sort lines by natsorted indices
    line_incidence = line_incidence.reindex(natsorted(line_incidence.index))

    #reorder columns by natsorted columns
    line_incidence = line_incidence[natsorted(line_incidence.columns)]

    line_incidence.to_csv(os.path.join(folder, 'data', case, 'line_incidence.csv'), sep=';', decimal=decimal, index=True)

    return None


def export_ptdf(ptdf: pd.DataFrame(), case: str, decimal='.', folder='', ptdf_file_name='ptdf', slack_bus=''):

    """

    :param ptdf:
    :param case:
    :param decimal:
    :param folder:
    :param ptdf_file_name: file name without extension
    :return:
    """

    # ptdf.index = ptdf.index.astype(str)
    ptdf.columns = ptdf.columns.astype(str)

    # sort ptdf by natsorted indices
    ptdf = ptdf.reindex(natsorted(ptdf.index))

    #reorder columns by natsorted columns
    ptdf = ptdf[natsorted(ptdf.columns)]

    # export ptdf
    ptdf.to_csv(os.path.join(folder, 'data', case, ptdf_file_name + '.csv'), decimal=decimal, sep=';', index=True)
    ptdf.to_csv(os.path.join(folder, 'data', case, ptdf_file_name + '_wo_indices.csv'), header=False, decimal=decimal, sep=';', index=False)

    if slack_bus != '':
        ptdf.drop([slack_bus], axis=1, inplace=True)
        ptdf.to_csv(os.path.join(folder, 'data', case, ptdf_file_name + '_sb_adj.csv'), header=False, decimal=decimal, sep=';', index=False)

    return


def demand_dict_to_df(demand_data: dict):

    dataframes = []
    for i in demand_data:
        aux_b = i['bus']
        df = pd.DataFrame(data={aux_b: i['demand']['load'].values})
        dataframes.append(df)
    df = pd.concat(dataframes, axis=1).T
    df.columns = demand_data[0]['demand']['hour'].values

    return df


def import_data_to_df(imports, periods):

    # aux_imp_data = pd.DataFrame(data={'period': list(range(1, periods + 1))})

    # if no import data is available return empty DataFrame
    if len(imports) == 0:
        return pd.DataFrame()

    # list with dfs of bus imports
    df_temp_imp = []

    for i in imports:
        aux_b = i['bus']
        df = pd.DataFrame(i['imports']['imports'].values, index=[aux_b], columns=range(1, periods+1))
        df_temp_imp.append(df)

    aux_imp_data = pd.concat(df_temp_imp, axis=0)

    aux_imp_data.index.name = 'bus'
    aux_imp_data.columns.name = 'period'

    return aux_imp_data


def perturb_data(df: pd.DataFrame, column:str, n_std: float = 1, trunc=True, min_val=0, max_val= np.Inf) -> pd.DataFrame:

    df_aux = df.copy()
    df_aux[column] = df_aux[column].astype('float64')
    std = df_aux[column].std()
    idx_perturb = (df_aux[column] != 0) & (df_aux[column] != 1)
    rnd = np.random.uniform(-std*n_std, std*n_std, len(df_aux.loc[idx_perturb, column]))
    df_aux.loc[idx_perturb, column] = (df_aux.loc[idx_perturb, column] + rnd)

    if df_aux[column].min() < 0:
        print('Negative values for variable', column)

    idx_perturb_min = (df_aux[column] < min_val) & idx_perturb
    df_aux.loc[idx_perturb_min, column] = min_val
    idx_perturb_max = (df_aux[column] > max_val) & idx_perturb
    df_aux.loc[idx_perturb_max, column] = max_val

    df_aux.loc[idx_perturb, column] = df_aux.loc[idx_perturb, column].round(2)
    if trunc:
        df_aux.loc[:, column] = df_aux.loc[:, column].round(0)


    return df_aux


def perturb_demand(df: pd.DataFrame, n_std: float = 0.2, trunc=True, min_val=0, max_val= np.Inf):

    # group by demand by hour (index)
    df_aux = df.copy()
    df_aux['load'] = df_aux['load'].astype('float64')
    df_aux['day'] = (df_aux['hour']-1) // 24 + 1

    df_day_dem = df_aux.groupby(by='day').agg({'load': 'sum'}).reset_index()
    df_day_dem.rename(columns={'load': 'day_load'}, inplace=True)
    df_day_dem['day_load_pert'] = perturb_data(df_day_dem, 'day_load', n_std, trunc, min_val, max_val)['day_load']


    df_aux_2 = pd.merge(df_aux, df_day_dem, on='day', how='left')
    df_aux_2['load_pert'] = (df_aux_2['load'] / df_aux_2['day_load']) * df_aux_2['day_load_pert']

    df_aux_2['load_pert'] = df_aux_2['load_pert'].round(0)
    df_aux['load'] = df_aux_2['load_pert']
    df_aux.drop(columns=['day'], inplace=True)

    if df_aux.load.min() < 0:
        print('Negative demand values')

    return df_aux


def create_scenario(case: str, base_case: str, n_std_cf: float = 1, n_std_dem: float = 0.2, decimal='.'):
    """

    :param case:
    :param base_case:
    :param n_std:
    :param trunc:
    :param min_val:
    :param max_val:
    :return:
    """

    # check if the case does not exist
    if not os.path.exists(os.path.join('data', case)):
        os.mkdir(os.path.join('data', case))
        os.mkdir(os.path.join('data', case, 'cf'))
        os.mkdir(os.path.join('data', case, 'loads'))
    else:
        # print a warning
        print(f'Case {case} already exists. Overwriting data...')


    files = ['renewables.csv', 'thermals.csv', 'network.csv', 'lines.csv', 'gen_costs.xlsx', 'nsp_costs.xlsx']
    for f in files:
        shutil.copyfile(os.path.join('data', base_case, f), os.path.join('data', case, f))


    cf_data = load_cf(base_case, decimal)
    demand_data = load_demand(base_case, decimal)


    for i in cf_data:
        i['cf'] = perturb_data(i['cf'], 'cf', n_std_cf, False, 0, 1)

    for i in demand_data:
        i['demand'] = perturb_demand(i['demand'], n_std_dem, True, 0, np.Inf)

    for i in cf_data:
        export_cf(i['generator'], i['cf'], case)
    for i in demand_data:
        export_demand(i['bus'], i['demand'], case)

    return None


def fill_model_data_opf(renewables, thermals, opf, lines, cf, demand, imports, periods, bus_info=pd.DataFrame()):
    l_renewable_gen = renewables.loc[:, 'unit'].to_list()
    l_thermal_gen = thermals.loc[:, 'unit'].to_list()
    l_gen = l_renewable_gen.copy()
    l_gen.extend(l_thermal_gen)

    # l_bus = __buses_from_lines__(lines)

    # alternative: get buses from bus_info:
    if bus_info.empty:
        l_bus = __buses_from_lines__(lines)
    else:
        l_bus = bus_info.loc[:, 'bus'].to_list()


    d_line_power, d_line_react = __lines_to_dict__(lines)

    # create df for cf
    print(periods)
    d_cf = {}

    dataframes = [pd.DataFrame(data={'period': list(range(1, periods + 1))})]

    for i in cf:
        aux_g = i['generator']
        df = pd.DataFrame(data={aux_g: i['cf']['cf'].values})
        dataframes.append(df)

    aux_cf_data = pd.concat(dataframes, axis=1)


    for i, row in aux_cf_data.iterrows():
        for c in row.index[1:]:
            # only create cf if there is a generator unit
            if c in l_renewable_gen:
                d_cf[c, int(row.loc['period'])] = row[c]

    # alt -----------------------------------------------------------------
    aux_dem_data = pd.DataFrame(data={'period': list(range(1, periods+1))})
    d_dem = {}
    dataframes = [pd.DataFrame(data={'period': list(range(1, periods + 1))})]

    for i in demand:
        aux_b = i['bus']
        df = pd.DataFrame(data={aux_b: i['demand']['load'].values})
        dataframes.append(df)

    aux_dem_data = pd.concat(dataframes, axis=1)

    for i, row in aux_dem_data.iterrows():
        for c in row.index[1:]:
            d_dem[c, row.loc['period']] = row[c]

    #--------------

    # imports data
    aux_imp_data = pd.DataFrame(data={'period': list(range(1, periods + 1))})
    d_imp = {}

    df_temp_imp = [pd.DataFrame(data={'period': list(range(1, periods + 1))})]

    for i in imports:
        aux_b = i['bus']
        df = pd.DataFrame(data={aux_b: i['imports']['imports'].values})
        df_temp_imp.append(df)

    aux_imp_data = pd.concat(df_temp_imp, axis=1)

    l_imp_nodes = list(aux_imp_data.keys())
    l_imp_nodes = l_imp_nodes[1:]

    for i, row in aux_imp_data.iterrows():
        for c in row.index[1:]:
            d_imp[c, row.loc['period']] = row[c]



    d_ru = {}
    d_rd = {}
    d_gmax = {}
    d_gmin = {}
    d_gb = {}
    for i, row in thermals.iterrows():
        d_ru[row['unit']] = row['rmp_up']
        d_rd[row['unit']] = row['rmp_dn']

        d_gmax[row['unit']] = row['max']
        d_gmin[row['unit']] = row['min']
        d_gb[row['unit'], row['bus']] = 1

    for i, row in renewables.iterrows():
        d_gmax[row['unit']] = row['max']
        d_gmin[row['unit']] = row['min']
        d_gb[row['unit'], row['bus']] = 1

    i_basepower = int(opf.loc['BasePower', 'parameter_value'])
    # i_basepower = 0.0 #float(opf.loc['BasePower', 'parameter_value'])
    str_slackbus = str(opf.loc['SlackBus', 'parameter_value'])
    i_maxanglediff = int(opf.loc['MaxAngleDiff', 'parameter_value'])*np.pi/180

    data = {
        'bpower': i_basepower,
        'slckbus': str_slackbus,
        'maxangdiff': i_maxanglediff,
        'renewables': l_renewable_gen,
        'thermals': l_thermal_gen,
        'generators': l_gen,
        'bus': l_bus,
        #'network': d_nw,
        'line_lim': d_line_power,
        'Xline': d_line_react,
        'cf': d_cf,
        'demand': d_dem,
        'imports': d_imp,
        'imp_nodes': l_imp_nodes,
        'rmpup': d_ru,
        'rmpdn': d_rd,
        'gmax': d_gmax,
        'gmin': d_gmin,
        'gb': d_gb,
    }

    return data





