#!interpreter [optional-arg]
# -*- coding: utf-8 -*-

__author__ = ["David Cardona-Vasquez"]
__copyright__ = "Copyright 2023, Graz University of Technology"
__credits__ = ["David Cardona-Vasquez"]
__license__ = "MIT"
__maintainer__ = "David Cardona-Vasquez"
__status__ = "Development"

import glob
import numpy as np
import os
import pandas as pd


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


def load_network(case: str, decimal='.', folder=""):
    """

    :param case:
    :return:
    """

    network_data = pd.read_csv(os.path.join(folder, 'data', case, 'network.csv'), sep=';', decimal=decimal)
    network_data.fillna(value=0, inplace=True)

    return network_data


def load_opf_parameters(case: str,decimal='.', folder=""):
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


def __matrix_to_dict__(df: pd.DataFrame):
    d = {}

    for i, row in df.iterrows():
        for c in row.index[1:]:
            d[row.iloc[0], c] = row[c]

    return d


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


def demand_dict_to_df(demand_data: dict):

    dataframes = []
    for i in demand_data:
        aux_b = i['bus']
        df = pd.DataFrame(data={aux_b: i['demand']['load'].values})
        dataframes.append(df)
    df = pd.concat(dataframes, axis=1).T

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


def fill_model_data_opf(renewables, thermals, opf, lines, cf, demand, imports, periods):
    l_renewable_gen = renewables.loc[:, 'unit'].to_list()
    l_thermal_gen = thermals.loc[:, 'unit'].to_list()
    l_gen = l_renewable_gen.copy()
    l_gen.extend(l_thermal_gen)

    l_bus = __buses_from_lines__(lines)
    d_line_power, d_line_react = __lines_to_dict__(lines)

    # create df for cf
    #aux_cf_data = pd.DataFrame(data={'period': list(range(1, periods+1))})
    print(periods)
    d_cf = {}

    # alt -----------------------------------------------------------------
    # for i in cf:
    #     aux_g = i['generator']
    #     aux_cf_data[aux_g] = i['cf']['cf'].values

    # neu -----------------------------------------------------------------
    dataframes = [pd.DataFrame(data={'period': list(range(1, periods + 1))})]

    for i in cf:
        aux_g = i['generator']
        df = pd.DataFrame(data={aux_g: i['cf']['cf'].values})
        dataframes.append(df)

    aux_cf_data = pd.concat(dataframes, axis=1)

    # ENDE neu -----------------------------------------------------------------

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



