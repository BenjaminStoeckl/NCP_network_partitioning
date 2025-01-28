#!interpreter [optional-arg]
# -*- coding: utf-8 -*-

__author__ = ["Benjamin Stöckl"]
__copyright__ = "Copyright 2024, Graz University of Technology"
__credits__ = ["Benjamin Stöckl"]
__license__ = "MIT"
__maintainer__ = "Benjamin Stöckl"
__status__ = "Development"

import os
import pandas as pd
import data
import shutil


# ------------------------------------ HELPER FUNCTIONS ------------------------------------
def check_if_folder_exists_and_create(folder, scenario_folder='', sub_folder='', delete_existing=False):

    if delete_existing:
        if os.path.exists(os.path.join(folder, scenario_folder)):
            shutil.rmtree(os.path.join(folder, scenario_folder))
        if sub_folder != '' and os.path.exists(os.path.join(folder, scenario_folder, sub_folder)):
            shutil.rmtree(os.path.join(folder, scenario_folder, sub_folder))


    if not os.path.exists(os.path.join(folder)):
        os.makedirs(os.path.join(folder))
    if not os.path.exists(os.path.join(folder, scenario_folder)):
        os.makedirs(os.path.join(folder, scenario_folder))
    if sub_folder != '' and not os.path.exists(os.path.join(folder, scenario_folder, sub_folder)):
        os.makedirs(os.path.join(folder, scenario_folder, sub_folder))


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


def get_res_nodes(input_folder, file_name):

    # load res gen data
    df = load_res_gen_data(input_folder, file_name)

    # only use original ID of unit and the bus information
    df = df[['orig_id', 'Bus', 'Type']]

    return df.reset_index(drop=True)


# ------------------------------------ LOAD DATA FUNCTIONS ------------------------------------
def load_demand_data(folder, file_name):

    df_demand_original = pd.read_excel(os.path.join(folder, file_name), sheet_name='Demand', skiprows=3, header=1)
    df_demand_original.drop(columns=['Unnamed: 0', 'Unnamed: 1'], inplace=True)

    df_demand_original.rename(columns={'Unnamed: 2': 'MILES_Node'}, inplace=True)
    df_demand_original['MILES_Node'] = df_demand_original['MILES_Node'].apply(lambda x: x.split('.')[1])

    return df_demand_original


def load_thermal_data(folder, file_name):

    df_thermal_original = pd.read_excel(os.path.join(folder, file_name), sheet_name='ThermalGen', skiprows=3, header=1)
    df_thermal_original.drop(columns=['Unnamed: 0', 'Unnamed: 1'], inplace=True)

    df_thermal_original.rename(columns={'Unnamed: 2': 'LEGO_id'}, inplace=True)
    df_thermal_original.drop([0], inplace=True)

    df_thermal_original = df_thermal_original.astype({'MaxProd': 'float64', 'MinProd': 'float64', 'RampUp': 'float64', 'RampDw': 'float64',
                                'StartupCost': 'int32'})

    return df_thermal_original


def load_res_gen_data(folder, file_name):

    df = pd.read_excel(os.path.join(folder, file_name), sheet_name='VRES', skiprows=3, header=1)
    df.drop([0], inplace=True)
    df.drop(columns=['Unnamed: 0', 'Unnamed: 1'], inplace=True)
    df.rename(columns={'Unnamed: 2': 'orig_id'}, inplace=True)
    df = df[df['ExisUnits'] > 0]
    df = df[['orig_id', 'ExisUnits', 'MaxProd', 'OMVarCost', 'Bus', 'Name', 'Type']]

    return df


def load_res_cf_data(folder, file_name, sheet_name):

    df = pd.read_excel(os.path.join(folder, file_name),
                              sheet_name=sheet_name, skiprows=4, header=1)
    df.drop(columns={'Unnamed: 0', 'Unnamed: 1'}, inplace=True)
    df.rename(columns={'Unnamed: 2': 'orig_node'}, inplace=True)
    df.dropna(axis=0, how='all', inplace=True)
    #df['orig_node'] = df['orig_node'].astype(str)
    df.insert(1, 'type', '')
    df['type'] = df['orig_node'].apply(lambda x: x.split('.')[2])
    df['orig_node'] = df['orig_node'].astype(str).apply(lambda x: x.split('.')[1])

    return df


def load_hydro_gen_data(folder, file_name):

        df = pd.read_excel(io=os.path.join(folder, file_name), sheet_name='RunOfRiver', header=1,
                           usecols={'Unnamed: 1', 'Unnamed: 2', 'ExisUnits', 'MaxProd', 'MaxCons', 'DisEffic',
                                    'OMVarCost', 'Bus', 'Name'},
                           skiprows=3, )
        df.rename(columns={'Unnamed: 2': 'LEGO_id'}, inplace=True)
        df = df[df['Unnamed: 1'].apply(lambda x: not str(x).startswith('*'))]
        df.drop(columns=['Unnamed: 1'], inplace=True)
        df.dropna(axis=0, how='all', inplace=True)
        #df.drop([0], inplace=True)
        df = df[df['ExisUnits'] > 0]
        df.reset_index(drop=True, inplace=True)

        df_aux = pd.read_excel(os.path.join(folder, file_name), sheet_name='StorageUnits', header=1,
                               usecols={'Unnamed: 1', 'Unnamed: 2', 'ExisUnits', 'MaxProd', 'MaxCons', 'DisEffic',
                                        'OMVarCost', 'Bus', 'Name'},
                               skiprows=3)
        df_aux = df_aux[df_aux['Unnamed: 1'].apply(lambda x: not str(x).startswith('*'))]
        df_aux = df_aux[df_aux['ExisUnits'] > 0]
        df_aux.drop(columns=['Unnamed: 1'], inplace=True)
        df_aux.dropna(axis=0, how='all', inplace=True)
        df_aux.rename(columns={'Unnamed: 2': 'LEGO_id'}, inplace=True)
        df_aux.reset_index(drop=True, inplace=True)

        df = pd.concat([df, df_aux], ignore_index=True)

        return df


def load_hydro_cf_data(folder, file_name, sheet_name):

    df = pd.read_excel(os.path.join(folder, file_name), sheet_name=sheet_name, skiprows=3, header=1)
    df = df[df['Unnamed: 1'].apply(lambda x: not str(x).startswith('*'))]
    df.drop(columns=['Unnamed: 0', 'Unnamed: 1'], inplace=True)
    df.rename(columns={'Unnamed: 2': 'orig_hydro_pp'}, inplace=True)
    df.dropna(axis=0, how='all', inplace=True)
    df['orig_hydro_pp'] = df['orig_hydro_pp'].astype(str).apply(lambda x: x.split('.')[1])
    df.reset_index(drop=True, inplace=True)

    list_keys = df.keys().to_series()[df.keys().to_series().astype(str).apply(lambda x: x.startswith('k'))].values
    number_of_hours = int(max([x.split('k')[1] for x in list_keys]))

    # check if df has more columns than number_of_hours + 1
    if len(df.columns) > number_of_hours + 1:
        excess_columns = len(df.columns) - number_of_hours - 1
        for i in range(excess_columns):
            df.drop(columns=df.columns[-1], inplace=True)

    return df


# ------------------------------------ EXPORT FUNCTIONS ------------------------------------
def demand_pyomo_export(df_demand, folder='res_agg_demand', folder_name='temp_demand'):

    check_if_folder_exists_and_create(folder, folder_name, 'loads')

    for i, row in df_demand.iterrows():
        df_temp = pd.DataFrame(columns=['hour', 'load'])
        df_temp['hour'] = range(1, len(row))
        df_temp['load'] = row[1:].values.astype(float)
        df_temp.to_csv(os.path.join(folder, folder_name, 'loads', str(row.iloc[0]) + '.csv'),
                       index=False, sep=';', decimal=',')


def thermal_gen_pyomo_export(df_thermal_in, thermal_gen_aggregated=False, folder='res_agg_gen',
                             folder_name='temp_thermal', without_ramping=False, min_prod_to_zero=False,
                             single_node=False):

    df_thermal = df_thermal_in.copy()
    # create folder if not exists
    if not os.path.exists(os.path.join(folder, folder_name)):
        os.makedirs(os.path.join(folder, folder_name))

    if thermal_gen_aggregated:
        #df_thermal_pyomo = df_thermal[['aggNode', 'MaxProd', 'MinProd', 'RampUp', 'RampDw', 'StartupCost']]
        df_thermal_pyomo = df_thermal[['unit', 'MaxProd', 'MinProd', 'RampUp', 'RampDw', 'StartupCost', 'aggNode']]

        # if ramping constraints should not be active set RampUp and RampDw to MaxProd
        if without_ramping:
            df_thermal_pyomo.loc[:, 'RampUp'] = df_thermal_pyomo.loc[:, 'MaxProd']
            df_thermal_pyomo.loc[:, 'RampDw'] = df_thermal_pyomo.loc[:, 'MaxProd']

        if min_prod_to_zero:
            df_thermal_pyomo.loc[:, 'MinProd'] = 0

        if single_node:
            df_thermal_pyomo.loc[:, 'aggNode'] = 'SingleNode'

        df_thermal_pyomo = df_thermal_pyomo.rename(columns={'MaxProd': 'max', 'MinProd': 'min',
                                                            'RampUp': 'rmp_up', 'RampDw': 'rmp_dn',
                                                            'StartupCost': 'commit', 'aggNode': 'bus'})

        df_thermal_pyomo.to_csv(os.path.join(folder, folder_name, 'thermals.csv'),
                                index=False, sep=';', decimal=',')
    else:
        if not 'aggNode' in df_thermal.columns:
            df_thermal.rename(columns={'Bus': 'aggNode'}, inplace=True)
            df_thermal.drop(columns={'Name'}, inplace=True)
            df_thermal.rename(columns={'unit': 'Name'}, inplace=True)

        df_thermal_pyomo = df_thermal[['Name', 'MinProd', 'MaxProd', 'RampUp', 'RampDw', 'StartupCost', 'aggNode']]
        df_thermal_pyomo = df_thermal_pyomo.rename(columns={'Name': 'unit', 'MaxProd': 'max', 'MinProd': 'min',
                                                            'RampUp': 'rmp_up', 'RampDw': 'rmp_dn',
                                                            'StartupCost': 'commit', 'aggNode': 'bus'})

        df_thermal_pyomo.to_csv(os.path.join(folder, folder_name, 'thermals.csv'),
                                index=False, sep=';', decimal=',')


def res_gen_pyomo_export(df_res_gen, folder='res_agg', folder_name='temp_scenario', single_node=False, res_agg=True):
    # create folder if not exists
    check_if_folder_exists_and_create(folder, folder_name)

    df_res_pyomo = load_file_or_gen_df(folder, 'renewables.csv',
                                            ['unit', 'max', 'min', 'type', 'bus'], folder_name)

    df_res_gen.insert(0, 'unit', '')

    if res_agg:
        df_res_gen.loc[df_res_gen['Type'] == 'Wind', 'unit'] += 'w_' + df_res_gen['aggNode']
        df_res_gen.loc[df_res_gen['Type'] == 'Photovoltaik', 'unit'] += 's_' + df_res_gen['aggNode']
    else:
        df_res_gen.loc[df_res_gen['Type'] == 'Wind', 'unit'] += 'w_' + df_res_gen['orig_id']
        df_res_gen.loc[df_res_gen['Type'] == 'Photovoltaik', 'unit'] += 's_' + df_res_gen['orig_id']

    df_res_gen.loc[df_res_gen['Type'] == 'Wind', 'Type'] = 'wind'
    df_res_gen.loc[df_res_gen['Type'] == 'Photovoltaik', 'Type'] = 'solar'

    df_res_gen.insert(6, 'min', 0)
    df_res_gen = df_res_gen[['unit', 'MaxProd', 'min', 'Type', 'aggNode']]
    df_res_gen = df_res_gen.rename(columns={'MaxProd': 'max', 'Type': 'type', 'aggNode': 'bus'})

    for i, row in df_res_gen.iterrows():
        if row['unit'] in df_res_pyomo['unit'].values:
            df_res_pyomo.loc[df_res_pyomo['unit'] == row['unit']] = \
                [row['unit'], row['max'], row['min'], row['type'], row['bus']]
        else:
            df_res_pyomo.loc[len(df_res_pyomo)] = \
                [row['unit'], row['max'], row['min'], row['type'], row['bus']]

    if single_node:
        df_res_pyomo.loc[:, 'bus'] = 'SingleNode'

    df_res_pyomo.to_csv(os.path.join(folder, folder_name, 'renewables.csv'), index=False, sep=';', decimal=',')


def res_cf_pyomo_export(df_res_cf, folder='res_agg_gen', folder_name='temp_gen', type_already_defined=False):

    check_if_folder_exists_and_create(folder, folder_name, 'cf')

    if type_already_defined:
        for i, row in df_res_cf.iterrows():
            df_temp = pd.DataFrame(columns=['hour', 'cf'])
            df_temp['hour'] = range(1, len(row)-1)
            df_temp['cf'] = row[2:].values.astype(float)
            df_temp.to_csv(os.path.join(folder, folder_name, 'cf', str(row['aggNode']) + '.csv'),
                           index=False, sep=';', decimal=',')

    else:
        for i, row in df_res_cf.iterrows():
            df_temp = pd.DataFrame(columns=['hour', 'cf'])
            df_temp['hour'] = range(1, len(row)-1)
            df_temp['cf'] = row[2:].values.astype(float)
            df_temp.to_csv(os.path.join(folder, folder_name, 'cf', row['type'][0].lower() + '_' + str(row['aggNode']) + '.csv'),
                           index=False, sep=';', decimal=',')


def hydro_gen_pyomo_export(df_hydro_gen, hydro_gen_aggregated=False, folder='res_agg_gen', folder_name='temp_hydro',
                           hydro_as_res=False, single_node=False):

    # create folder if not exists
    check_if_folder_exists_and_create(folder, folder_name)

    df_hydro_gen_pyomo = df_hydro_gen
    df_hydro_gen_pyomo.insert(0, 'unit', '')
    df_hydro_gen_pyomo.insert(2, 'type', 'hydro')
    df_hydro_gen_pyomo.insert(6, 'min', 0)

    if hydro_gen_aggregated:

        df_hydro_gen_pyomo.loc[:, 'unit'] += 'hy_' + df_hydro_gen_pyomo['aggNode']

        df_hydro_gen_pyomo = df_hydro_gen_pyomo[['unit', 'MaxProd', 'min', 'type', 'aggNode']]
        df_hydro_gen_pyomo = df_hydro_gen_pyomo.rename(columns={'MaxProd': 'max', 'aggNode': 'bus'})
        file_name = 'agg_hydro.csv'

    else:

        df_hydro_gen_pyomo.loc[:, 'unit'] += 'hy_' + df_hydro_gen_pyomo['LEGO_id']
        df_hydro_gen_pyomo = df_hydro_gen_pyomo[['unit', 'MaxProd', 'min', 'type', 'aggNode']]
        df_hydro_gen_pyomo = df_hydro_gen_pyomo.rename(columns={'MaxProd': 'max', 'aggNode': 'bus'})
        file_name = 'ass_hydro.csv'

    if single_node:
        df_hydro_gen_pyomo.loc[:, 'bus'] = 'SingleNode'

    # if hydro generation has same definition as renewable generation
    if hydro_as_res:
        df_renewables = load_file_or_gen_df(folder, 'renewables.csv',
                                                    ['unit', 'max', 'min', 'type', 'bus'], folder_name)

        for i, row in df_hydro_gen_pyomo.iterrows():
            if row['unit'] in df_renewables['unit'].values:
                df_renewables.loc[df_renewables['unit'] == row['unit']] = \
                    [row['unit'], row['max'], row['min'], 'hydro', row['bus']]
            else:
                df_renewables.loc[len(df_renewables)] = \
                    [row['unit'], row['max'], row['min'], 'hydro', row['bus']]

        df_renewables.to_csv(os.path.join(folder, folder_name, 'renewables.csv'), index=False, sep=';', decimal=',')

    else:

        df_hydro_gen_pyomo.to_csv(os.path.join(folder, folder_name, file_name),
                                  index=False, sep=';', decimal=',')


def hydro_cf_pyomo_export(df_hydro_cf, hydro_cf_aggregated=False, folder='res_agg_gen', folder_name='temp_hydro_cf'):

    # create folder if not exists
    check_if_folder_exists_and_create(folder, folder_name, 'cf')

    df_hydro_cf_pyomo = df_hydro_cf
    df_hydro_cf_pyomo.loc[:, 'aggNode'] = 'hy_' + df_hydro_cf_pyomo['aggNode']

    for i, row in df_hydro_cf_pyomo.iterrows():
        df_temp = pd.DataFrame(columns=['hour', 'cf'])
        df_temp['hour'] = range(1, len(row))
        df_temp['cf'] = row[1:].values
        df_temp['cf'] = df_temp['cf'].astype(float)
        df_temp.to_csv(os.path.join(folder, folder_name, 'cf', row['aggNode'] + '.csv'),
                       index=False, sep=';', decimal=',')


def res_gen_cost_export(df_res, folder, folder_name, variable_cost='original', vc_difference=1):

    # create file if not exists
    if not os.path.exists(os.path.join(folder, folder_name)):
        os.makedirs(os.path.join(folder, folder_name))

    df_gen_cost = load_file_or_gen_df(folder, 'gen_costs.csv', ['resource', 'CG'], folder_name)

    l_gc_counter = {}
    for i, row in df_res.iterrows():

        # create variable cost with original values
        if variable_cost == 'original':
            if row['unit'] in df_gen_cost['resource'].values:
                df_gen_cost.loc[df_gen_cost['resource'] == row['unit'], 'CG'] = row['OMVarCost']
            else:
                df_gen_cost.loc[len(df_gen_cost)] = [row['unit'], row['OMVarCost']]

        elif variable_cost == 'sameDifference' or variable_cost == 'increasingDifference':
            # check if RES have same OMVarCost and if so change accordingly
            if df_res['OMVarCost'].eq(row['OMVarCost']).any():
                if row['OMVarCost'] not in l_gc_counter.keys():
                    l_gc_counter[row['OMVarCost']] = 1
                else:
                    l_gc_counter[row['OMVarCost']] += 1

                # create variable cost with same difference
                if variable_cost == 'sameDifference':
                    if row['unit'] in df_gen_cost['resource'].values:
                        df_gen_cost.loc[df_gen_cost['resource'] == row['unit'], 'CG'] = \
                                             row['OMVarCost'] + (vc_difference * l_gc_counter[row['OMVarCost']])

                    else:
                        df_gen_cost.loc[len(df_gen_cost)] = [row['unit'],
                                             row['OMVarCost'] + (vc_difference * l_gc_counter[row['OMVarCost']])]

                # create variable cost with increasing difference
                elif variable_cost == 'increasingDifference':
                    if row['unit'] in df_gen_cost['resource'].values:
                        df_gen_cost.loc[df_gen_cost['resource'] == row['unit'], 'CG'] = \
                            row['OMVarCost'] * (vc_difference + (l_gc_counter[row['OMVarCost']] * 0.001))
                    else:
                        df_gen_cost.loc[len(df_gen_cost)] = [row['unit'],
                                        row['OMVarCost'] * (vc_difference + (l_gc_counter[row['OMVarCost']] * 0.001))]
            else:
                if row['unit'] in df_gen_cost['resource'].values:
                    df_gen_cost.loc[df_gen_cost['resource'] == row['unit'], 'CG'] = row['OMVarCost']
                else:
                    df_gen_cost.loc[len(df_gen_cost)] = [row['unit'], row['OMVarCost']]
        else:
            exit('Wrong variable cost type! Choose between original, sameDifference or increasingDifference')

    #df_gen_cost.reset_index(drop=True, inplace=True)

    df_gen_cost.to_csv(os.path.join(folder, folder_name, 'gen_costs.csv'), index=False, sep=';', decimal=',')


def gen_cost_export(df_thermal, folder, folder_name, variable_cost='original', vc_difference=1):

    # create file if not exists
    if not os.path.exists(os.path.join(folder, folder_name)):
        os.makedirs(os.path.join(folder, folder_name))

    if not os.path.exists(os.path.join(folder, folder_name, 'gen_costs.csv')):
        df_gen_cost = pd.DataFrame(columns=['resource', 'CG'])
    else:
        df_gen_cost = pd.read_csv(os.path.join(folder, folder_name, 'gen_costs.csv'), sep=';', decimal=',')

    # define datatype otherwise value comparison will not work
    df_thermal['OMVarCost'] = df_thermal['OMVarCost'].astype(float)
    df_gen_cost['CG'] = df_gen_cost['CG'].astype(float)

    # create variable cost with original values
    if variable_cost == 'original':
        for i, row in df_thermal.iterrows():

            # cost per MWh = OMVarCost + FuelCost * SlopeVarCost
            th_cost = round(row['OMVarCost'] + row['FuelCost'] * row['SlopeVarCost'], 2)

            # if unit already exists change value otherwise add row
            if row['unit'] in df_gen_cost['resource'].values:
                df_gen_cost.loc[df_gen_cost['resource'] == row['unit'], 'CG'] = th_cost
            else:
                df_gen_cost.loc[len(df_gen_cost)] = th_cost

    # create variable cost with same difference
    elif variable_cost == 'sameDifference':
        l_gc_counter = {}

        # loop over all thermal units
        for i, row in df_thermal.iterrows():

            # check if multiple units have the same OMVarCost
            # if so, increase the cost by vc_difference
            # else, set the cost to the original cost
            if df_thermal['OMVarCost'].eq(row['OMVarCost']).any():

                # if multiple OMVarCost increase counter or set to 1
                if row['OMVarCost'] in l_gc_counter.keys():
                    l_gc_counter[row['OMVarCost']] += 1
                else:
                    l_gc_counter[row['OMVarCost']] = 1

                # define new variable cost
                new_sd_cost = row['OMVarCost'] + row['FuelCost'] * row['SlopeVarCost'] \
                                               + (vc_difference * l_gc_counter[row['OMVarCost']])

                # check if unit already exists in the gen_costs file
                # if exists, update the cost with the new cost
                # else, add a new row with the new cost
                if row['unit'] in df_gen_cost['resource'].values:
                    df_gen_cost.loc[df_gen_cost['resource'] == row['unit'], 'CG'] = new_sd_cost
                else:
                    df_gen_cost.loc[len(df_gen_cost)] = [row['unit'], new_sd_cost]
            else:
                th_cost = row['OMVarCost'] + row['FuelCost'] * row['SlopeVarCost']
                if row['unit'] in df_gen_cost['resource'].values:
                    df_gen_cost.loc[df_gen_cost['resource'] == row['unit'], 'CG'] = th_cost
                else:
                    df_gen_cost.loc[len(df_gen_cost)] = [row['unit'], th_cost]

    # create variable cost with increasing difference
    elif variable_cost == 'increasingDifference':
        l_gc_counter = {}
        for i, row in df_thermal.iterrows():

            # check if multiple units have the same OMVarCost
            # if so, increase the cost by vc_difference
            # else, set the cost to the original cost
            if df_gen_cost.eq(row['OMVarCost']).sum().sum() != 0:

                # increase counter or set to 1
                if row['OMVarCost'] not in l_gc_counter.keys():
                    l_gc_counter[row['OMVarCost']] = 1
                else:
                    l_gc_counter[row['OMVarCost']] += 1

                # check if unit already exists in the gen_costs file
                # if exists, update the cost with the new cost
                # else, add a new row with the new cost
                if row['unit'] in df_gen_cost['resource'].values:
                    df_gen_cost.loc[df_gen_cost['resource'] == row['unit'], 'CG'] = \
                        row['OMVarCost'] * (vc_difference + (l_gc_counter[row['OMVarCost']] * 0.01))
                else:
                    df_gen_cost.loc[len(df_gen_cost)] = [row['unit'],
                                row['OMVarCost'] * (vc_difference + (l_gc_counter[row['OMVarCost']] * 0.01))]

            else:
                if row['unit'] in df_gen_cost['resource'].values:
                    df_gen_cost.loc[df_gen_cost['resource'] == row['unit'], 'CG'] = row['OMVarCost']
                else:
                    df_gen_cost.loc[len(df_gen_cost)] = [row['unit'], row['OMVarCost']]

    else:
        exit('Wrong variable cost type! Choose between original, sameDifference or increasingDifference')

    df_gen_cost.reset_index(drop=True, inplace=True)

    df_gen_cost.to_csv(os.path.join(folder, folder_name, 'gen_costs.csv'), index=False, sep=';', decimal=',')


def export_import(df_import, out_folder, scenario_name):

    check_if_folder_exists_and_create(out_folder, scenario_name, 'imports')

    df_import.reset_index(inplace=True, drop=True)

    for node in df_import.columns:
        df_temp = pd.DataFrame(columns=['hour', 'imports'])
        df_temp['hour'] = range(1, len(df_import.index) + 1)
        df_temp['imports'] = df_import[node].astype(float).round(2)
        df_temp.to_csv(os.path.join(out_folder, scenario_name, 'imports', str(node) + '.csv'),
                       index=False, sep=';', decimal=',')

# ------------------------------------ ASSIGN FUNCTIONS ------------------------------------

def create_nsp_cost_for_nodes(df_nodes, NSP_COST, folder, case_name, single_node=False, variable_cost='original'):

    if single_node:
        df_nodes = pd.DataFrame(columns=['zone', 'CNSP'])
        df_nodes.loc[0, 'zone'] = 'SingleNode'
        df_nodes.loc[0, 'CNSP'] = NSP_COST

    else:
        df_nodes.drop(columns=['originalNode'], inplace=True)
        df_nodes = df_nodes.groupby('aggNode').mean()

        if variable_cost == 'original':
            df_nodes.insert(len(df_nodes.columns), 'CNSP', NSP_COST)

        elif variable_cost == 'sameDifference':
            #df_nodes.insert(len(df_nodes.columns), 'CNSP', NSP_COST + len(df_nodes))
            df_nodes.loc[:, 'CNSP'] = [NSP_COST + x for x in range(0, len(df_nodes))]

        elif variable_cost == 'increasingDifference':
            print(len(df_nodes))
            df_nodes.loc[:, 'CNSP'] = [NSP_COST + x ** 1.1 for x in range(0, len(df_nodes))]

        df_nodes.reset_index(inplace=True)
        df_nodes = df_nodes.rename(columns={'aggNode': 'zone'})

    df_nodes.to_csv(os.path.join(folder, case_name, 'nsp_costs.csv'), sep=';', decimal=',', index=False)


def get_clustered_buses(inp_fold, inp_case, out_case, df_bus_info=pd.DataFrame()):
    '''

    load 'bus_info.csv' and make mean from coordinates of cluster. Also make "incidence matrix" of

    return: incidence matrix of original and new buses

    '''
    # load bus information and create new coordinates for new buses
    if df_bus_info.empty:
        df_bus_info = pd.read_csv(os.path.join(inp_fold, inp_case, 'bus_info.csv'), sep=';', decimal=',')
    # df_bus_info.set_index('bus', inplace=True)

    if 'BaseVolt' in df_bus_info.columns:

        df_bus_incidence = df_bus_info.drop(columns={'BaseVolt', 'Country', 'Name', 'lat', 'long'})

        df_bus_info.drop(columns={'BaseVolt', 'Country', 'Name'}, inplace=True)
    else:
        df_bus_incidence = df_bus_info.drop(columns={'lat', 'long'})

    if df_bus_info.empty:
        df_bus_info.drop(columns={'bus'}, inplace=True)
        df_bus_info = df_bus_info.groupby('cluster').mean()
        df_bus_info.reset_index(inplace=True, drop=False)
        df_bus_info.rename(columns={'cluster': 'bus'}, inplace=True)

    df_bus_info.to_csv(os.path.join(inp_fold, out_case, 'bus_info.csv'), sep=';', decimal=',', index=False)

    return df_bus_incidence


def aggregate_demand(inp_fold, inp_case, res_fold, out_case, df_bus_incidence, num_timesteps, single_node=False):

    l_demand = data.load_demand(inp_case, decimal=',')
    d_dem = {}
    dataframes = [pd.DataFrame(data={'period': list(range(1, num_timesteps + 1))})]

    for i in l_demand:
        aux_b = i['bus']
        df = pd.DataFrame(data={aux_b: i['demand']['load'].values})
        dataframes.append(df)

    aux_dem_data = pd.concat(dataframes, axis=1)

    for i, row in aux_dem_data.iterrows():
        for c in row.index[1:]:
            d_dem[c, row.loc['period']] = row[c]

    df_demand = pd.DataFrame.from_dict(d_dem, orient='index')
    df_demand.reset_index(inplace=True, drop=False)
    df_demand[['bus', 'hour']] = pd.DataFrame(df_demand['index'].tolist(), index=df_demand.index)
    df_demand.drop(columns={'index'}, inplace=True)
    df_demand.set_index('bus', inplace=True)

    if 'bus' not in df_bus_incidence.columns and df_bus_incidence.index.name == 'bus':
        df_bus_incidence.reset_index(inplace=True, drop=False)

    if df_bus_incidence['bus'].dtype == 'float64':
        df_bus_incidence['bus'] = df_bus_incidence['bus'].astype(int)

    df_bus_incidence['bus'] = df_bus_incidence['bus'].astype(str)
    df_bus_incidence.set_index('bus', inplace=True)

    df_demand = df_demand.join(df_bus_incidence, how='left')
    df_demand.drop(columns={'hour'}, inplace=True)
    df_demand = df_demand.groupby('cluster').sum()
    df_demand.reset_index(inplace=True, drop=False)
    df_demand['cluster'] = df_demand['cluster'].astype(int).astype(str)

    if 'load' not in df_demand.columns:
        df_demand.rename(columns={0: 'load'}, inplace=True)

    demand_pyomo_export(df_demand, folder=inp_fold, folder_name=out_case)

    return True


def aggregate_import(inp_fold, inp_case, res_fold, out_case, df_bus_incidence, num_of_ts):

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

    lego_to_pyomo.export_import(df_import, inp_fold, out_case)

    return True


def aggregate_nsp_cost(inp_fold, inp_case, res_fold, out_case, df_bus_incidence):

    df_nsp_cost = pd.read_csv(os.path.join(inp_fold, inp_case, 'nsp_costs.csv'), sep=';', decimal=',')
    df_nsp_cost.set_index('zone', inplace=True)

    df_nsp_cost = df_nsp_cost.join(df_bus_incidence, how='left')
    df_nsp_cost = df_nsp_cost.groupby('cluster').mean()
    df_nsp_cost.reset_index(inplace=True, drop=False)
    df_nsp_cost['zone'] = df_nsp_cost['cluster'].astype(str)
    df_nsp_cost.drop(columns={'cluster'}, inplace=True)

    df_nsp_cost.to_csv(os.path.join(inp_fold, out_case, 'nsp_costs.csv'), sep=';', decimal=',', index=False)

    return True


def define_opf_parameters(inp_fold, input_case, out_case, slack_bus='1'):

    df_opf = pd.read_csv(os.path.join(inp_fold, input_case, 'opf_parameters.csv'), sep=';', decimal=',')
    df_opf.loc[df_opf['parameter_name'] == 'SlackBus', 'parameter_value'] = slack_bus
    df_opf.to_csv(os.path.join(inp_fold, out_case, 'opf_parameters.csv'), sep=';', decimal=',', index=False)

    return None


def assign_clustered_case(inp_fold, res_fold, inp_case, out_case, num_of_ts, opf='PTDF', df_bus_info=pd.DataFrame()):

    '''
    Assign generation units to aggregated nodes. If df_bus_info is provided us this, else load data from

    '''

    def assign_thermal_production(inp_fold, inp_case, res_fold, out_case, df_bus_incidence):

        df_thermal = data.load_thermals(inp_case, decimal=',')
        df_thermal.set_index('bus', inplace=True)

        df_thermal = df_thermal.join(df_bus_incidence, how='left')

        df_thermal_incidence = df_thermal[['cluster', 'unit']]

        df_thermal.rename(columns={'cluster': 'bus'}, inplace=True)
        df_thermal.to_csv(os.path.join(inp_fold, out_case, 'thermals.csv'), sep=';', decimal=',', index=False)

        return df_thermal_incidence


    def assign_res_production(inp_fold, inp_case, res_fold, out_case, df_bus_incidence):
        df_res = data.load_renewables(inp_case, decimal=',')

        df_res.set_index('bus', inplace=True)
        df_res = df_res.join(df_bus_incidence, how='left')

        df_res_unit_cluster_bus = df_res[['unit', 'cluster', 'type']]
        df_res_unit_cluster_bus.reset_index(inplace=True, drop=False)

        df_res.rename(columns={'cluster': 'bus'}, inplace=True)
        df_res.to_csv(os.path.join(inp_fold, out_case, 'renewables.csv'), sep=';', decimal=',', index=False)

        return df_res_unit_cluster_bus


    check_if_folder_exists_and_create(inp_fold, out_case)
    check_if_folder_exists_and_create(inp_fold, out_case, 'cf')

    if df_bus_info.empty:
        # get bus information and create new bus_info (new coordinates)
        df_bus_incidence = get_clustered_buses(inp_fold, inp_case, out_case)
    else:
        # get bus incidence from bus info

        if 'BaseVolt' in df_bus_info.columns:
            df_bus_incidence = df_bus_info.drop(columns={'BaseVolt', 'Country', 'Name', 'lat', 'long'})

            df_bus_info.drop(columns={'BaseVolt', 'Country', 'Name', 'bus'}, inplace=True)
        else:
            df_bus_incidence = df_bus_info.drop(columns={'lat', 'long'})
        #     df_bus_info.drop(columns={'bus'}, inplace=True)
        #
            df_bus_info = df_bus_info.groupby('cluster').mean()
            df_bus_info.reset_index(inplace=True, drop=False)
            df_bus_info.rename(columns={'cluster': 'bus'}, inplace=True)
            df_bus_info.to_csv(os.path.join(inp_fold, out_case, 'bus_info.csv'), sep=';', decimal=',', index=False)


    # map demand to new buses and sum up
    aggregate_demand(inp_fold, inp_case, res_fold, out_case, df_bus_incidence, num_of_ts)

    # map imports to new buses and sum up
    aggregate_import(inp_fold, inp_case, res_fold, out_case, df_bus_incidence, num_of_ts)

    # assign thermal production to clustered nodes
    assign_thermal_production(inp_fold, inp_case, res_fold, out_case, df_bus_incidence)

    assign_res_production(inp_fold, inp_case, res_fold, out_case, df_bus_incidence)

    aggregate_nsp_cost(inp_fold, inp_case, res_fold, out_case, df_bus_incidence)

    define_opf_parameters(inp_fold, inp_case, out_case)

    __copy_csv_files__(inp_fold, inp_fold, inp_case, out_case, 'gen_costs.csv')

    # copy all csv files in 'cf' folder
    file = os.listdir(os.path.join(inp_fold, inp_case, 'cf'))

    for f in file:
        __copy_csv_files__(inp_fold, inp_fold, os.path.join(inp_case, 'cf'), os.path.join(out_case, 'cf'), f)


    return None


if __name__ == '__main__':
    single_node = False
    number_of_clusters = 20
    # scenario_name = 'austriaImEx_agg17'
    scenario_name = 'austriaImEx_246_imex20_p251'
    output_case = 'austriaImEx_246_imex20_p251_red_spectralClustering' + str(number_of_clusters)
    number_of_timesteps = 1
    begin_timestep = 1
    # name_additional_info = '_test'  # always start with '_...
    input_folder = 'data'
    results_folder = os.path.join('C:\\', 'BeSt', 'results')

    variable_cost_manipulation = 'sameDifference'  # original, sameDifference or increasing difference

    # case_name = (scenario_name
    #              + '_' + str(begin_timestep)
    #              + '-' + str(number_of_timesteps + begin_timestep-1)
    #              + ('_' + variable_cost_manipulation if variable_cost_manipulation != '' else '')
    #              + ('_single_node' if single_node else ''))

    # aggregate_clustered_case(input_folder, results_folder, scenario_name, output_case, number_of_timesteps,
    #                          opf='PTDF')

    assign_clustered_case(input_folder, results_folder, scenario_name, output_case, number_of_timesteps,
                          opf='PTDF')

    # gen_load_aggregation(scenario_name, 'austria_219', number_of_timesteps,
    #                      res_folder=input_folder, single_node=single_node,
    #                      vc_cost=variable_cost_manipulation, begin_timestep=begin_timestep)