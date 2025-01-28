import numpy as np
import pandas as pd
import os
import gen_load_aggregation as gla


def __buses_from_lines__(df: pd.DataFrame):
    d = []

    for i, row in df.iterrows():
        if row['bus1'] not in d:
            d.append(row['bus1'])
        if row['bus2'] not in d:
            d.append(row['bus2'])

    return d


def convert_demand(input_name, file_name, output, num_of_ts, begin_ts=1, scenario_name="unnamed_scenario",
                   demand_scaling_factor=1):

    # load LEGO demand data
    df_demand = gla.load_demand_data(input_name, file_name)

    # filter nodes outside of Austria
    df_demand = df_demand[df_demand['MILES_Node'].str.startswith('AT')]

    # clip data if less time steps are needed
    if num_of_ts is not len(df_demand.columns) - 1:

        df_demand = df_demand.iloc[:, [0] + list(range(begin_ts, begin_ts + num_of_ts))]

        # multiply demand by factor if needed
    if demand_scaling_factor != 1:
        df_demand_aux = df_demand.loc[:, df_demand.columns != 'MILES_Node']
        df_demand_aux = df_demand_aux * demand_scaling_factor
        df_demand_aux.insert(0, column=df_demand.columns.values[0], value=df_demand[df_demand.columns.values[0]].values)
        df_demand = df_demand_aux


    # export demand to pyomo format
    gla.demand_pyomo_export(df_demand, output, scenario_name)


def convert_thermal_gen(input_name, file_name, output, scenario_name="unnamed_scenario"):

    # load LEGO thermal generation data
    df_thermal_gen = gla.load_thermal_data(input_name, file_name)

    df_thermal_gen.reset_index(inplace=True, drop=True)
    df_thermal_gen.insert(0, 'unit', '')
    df_thermal_gen.loc[:, 'unit'] = 'th_' + df_thermal_gen['LEGO_id']
    
    # export thermal generation to pyomo format
    gla.thermal_gen_pyomo_export(df_thermal_gen, False, output, scenario_name, without_ramping=True)

    # export generation cost for thermal units
    gla.gen_cost_export(df_thermal_gen, output, scenario_name, variable_cost='sameDifference', vc_difference=0.001)


def convert_res_gen(input_name, file_name, output, scenario_name="unnamed_scenario"):
    # load LEGO renewable generation data
    df_res_gen = gla.load_res_gen_data(input_name, file_name)

    df_res_gen.reset_index(inplace=True, drop=True)
    df_res_gen.rename(columns={'Bus': 'aggNode'}, inplace=True)

    # export renewable generation to pyomo format
    gla.res_gen_pyomo_export(df_res_gen, output, scenario_name, res_agg=False)

    # export generation cost
    gla.res_gen_cost_export(df_res_gen, output, scenario_name, variable_cost='sameDifference', vc_difference=0.001)


def convert_res_cf(input_name, file_name, output, name, num_of_hours, begin_ts, scenario_name="unnamed_scenario",
                   wind_factor=1.5):

    # load cf time lines from MILES data
    df_res_cf = gla.load_res_cf_data(input_name, file_name, 'VRES-profiles')

    # get nodes assigned to res generation units
    df_res_nodes = gla.get_res_nodes(input_name, file_name)

    # clip data if less time steps are needed
    if num_of_hours is not len(df_res_cf.columns) - 2:
        df_res_cf = df_res_cf.iloc[:, [0, 1] + list(range(begin_ts + 1, begin_ts + num_of_hours + 1))]

    df_res_cf.rename(columns={'orig_node': 'aggNode'}, inplace=True)

    df_res_gen_cf = pd.DataFrame()

    df_res_cf.loc[df_res_cf['type'] == 'Solar', 'type'] = 'solar'
    df_res_nodes.loc[df_res_nodes['Type'] == 'Photovoltaik', 'Type'] = 'solar'

    # assign unit ID to cf nodes
    for i, row in df_res_nodes.iterrows():
        df_aux = df_res_cf[(df_res_cf['aggNode'] == row['Bus']) & (df_res_cf['type'] == row['Type'])]
        df_aux.insert(0, 'ID', row['orig_id'])

        df_res_gen_cf = pd.concat([df_aux, df_res_gen_cf], ignore_index=True)

    # rename ID column to be in line for the pyomo export function
    df_res_gen_cf.pop('aggNode')  # delete line with nodes
    df_res_gen_cf.rename(columns={'ID': 'aggNode'}, inplace=True)

    # multiply Wind cf by factor if needed
    if wind_factor != 1:
        df_res_gen_cf.loc[df_res_gen_cf['type'] == 'Wind', df_res_gen_cf.columns[2:]] = \
            df_res_gen_cf.loc[df_res_gen_cf['type'] == 'Wind', df_res_gen_cf.columns[2:]] * wind_factor


    # export aggregated res cf to file structure for PYOMO
    gla.res_cf_pyomo_export(df_res_gen_cf, output, name)

    return df_res_cf


def convert_hydro_gen(input_name, file_name, output, scenario_name="unnamed_scenario", num_of_hours=8760, begin_ts=1):
    # load LEGO hydro generation data
    df_hydro_gen = gla.load_hydro_gen_data(input_name, file_name)

    df_hydro_gen.reset_index(inplace=True, drop=True)
    df_hydro_gen.rename(columns={'Bus': 'aggNode'}, inplace=True)

    # export hydro generation to pyomo format
    gla.hydro_gen_pyomo_export(df_hydro_gen, folder=output, folder_name=scenario_name, hydro_gen_aggregated=False,
                               hydro_as_res=True)

    # export generation cost for hydro units
    gla.res_gen_cost_export(df_hydro_gen, output, scenario_name, variable_cost='sameDifference', vc_difference=0.001)

    ## export hydro cf data
    # load cf time lines from MILES data
    df_hydro_cf = gla.load_hydro_cf_data(input_name, file_name, 'Inflows')

    # clip data if less time steps are needed
    if num_of_hours is not len(df_hydro_cf.columns) - 2:
        df_hydro_cf = df_hydro_cf.iloc[:, [0] + list(range(begin_ts, begin_ts + num_of_hours))]

    df_hydro_cf.set_index('orig_hydro_pp', inplace=True)

    for i, row in df_hydro_cf.iterrows():
        df_hydro_cf.loc[i] = df_hydro_cf.loc[i] / df_hydro_gen.loc[df_hydro_gen['LEGO_id'] == i,
                                'MaxProd'].values[0]

    df_hydro_cf = df_hydro_cf.round(5)  # round ouptut to reduce unnecessary data

    df_hydro_cf.reset_index(inplace=True)
    df_hydro_cf.rename(columns={'orig_hydro_pp': 'aggNode'}, inplace=True)

    # export aggregated res cf to file structure for PYOMO
    gla.hydro_cf_pyomo_export(df_hydro_cf, True, folder=output, folder_name=scenario_name)


def convert_hydro_cf(input_name, file_name, output, name, num_of_hours, begin_ts, scenario_name="unnamed_scenario"):

    # load cf time lines from MILES data
    df_hydro_cf = gla.load_hydro_cf_data(input_name, file_name, 'Inflows')

    # clip data if less time steps are needed
    if num_of_hours is not len(df_hydro_cf.columns) - 2:
        df_hydro_cf = df_hydro_cf.iloc[:, [0] + list(range(begin_ts, begin_ts + num_of_hours))]

    df_hydro_cf.rename(columns={'orig_hydro_pp': 'aggNode'}, inplace=True)

    # export aggregated res cf to file structure for PYOMO
    gla.hydro_cf_pyomo_export(df_hydro_cf, True, folder=output, folder_name=name)

    return df_hydro_cf


def convert_scenario_to_pyomo(input_name, base_f, output, scenario_name, begin_ts, num_of_ts, vc_cost='sameDifference'):

    gla.check_if_folder_exists_and_create(output, scenario_name)

    gla.__copy_csv_files__(output, output, base_f, scenario_name, 'opf_parameters.csv')
    gla.__copy_csv_files__(output, output, base_f, scenario_name, 'lines.csv')

    if 'bus_info.csv' in os.listdir(os.path.join(output, scenario_name)):
        gla.__copy_csv_files__(output, output, base_f, scenario_name, 'bus_info.csv')

    convert_demand(input_name, '03_Demand.xlsx', output, num_of_ts, begin_ts, scenario_name,
                   demand_scaling_factor=1)

    convert_thermal_gen(input_name, '04_ThermalGen.xlsx', output, scenario_name)

    convert_res_gen(input_name, '06_VRES_RESprofiles.xlsx', output, scenario_name)

    convert_res_cf(input_name, '06_VRES_RESprofiles.xlsx', output, scenario_name, num_of_ts, begin_ts)

    convert_hydro_gen(input_name, '05_Storage_RoR_Inflows.xlsx', output, scenario_name, num_of_ts,
                      begin_ts)

    df_lines = pd.read_csv(os.path.join(output, scenario_name, 'lines.csv'), sep=';', decimal=',')

    l_buses = __buses_from_lines__(df_lines)
    df_buses = pd.DataFrame(l_buses, columns=['aggNode'])
    df_buses.insert(1, 'originalNode', 0)

    gla.create_nsp_cost_for_nodes(df_buses, 5000, output, scenario_name, variable_cost='sameDifference')


def export_export(df_export, out_folder, scenario_name):
    '''
    Export data in Pyomo format as demand on foreign nodes
    '''

    gla.check_if_folder_exists_and_create(out_folder, scenario_name, 'loads')

    df_export.reset_index(inplace=True, drop=True)

    for node in df_export.columns:
        df_temp = pd.DataFrame(columns=['hour', 'load'])
        df_temp['hour'] = range(1, len(df_export.index) + 1)
        df_temp['load'] = abs(df_export[node].astype(float).round(2))
        df_temp.to_csv(os.path.join(out_folder, scenario_name, 'loads', node + '.csv'),
                       index=False, sep=';', decimal=',')


def export_import(df_import, out_folder, scenario_name):

    gla.check_if_folder_exists_and_create(out_folder, scenario_name, 'imports')

    df_import.reset_index(inplace=True, drop=True)

    for node in df_import.columns:
        df_temp = pd.DataFrame(columns=['hour', 'imports'])
        df_temp['hour'] = range(1, len(df_import.index) + 1)
        df_temp['imports'] = df_import[node].astype(float).round(2)
        df_temp.to_csv(os.path.join(out_folder, scenario_name, 'imports', str(node) + '.csv'),
                       index=False, sep=';', decimal=',')


def create_import_export(in_folder, out_folder, scenario_name, begin_ts, num_of_ts, im_scaling_factor=0.5, ex_scaling_factor=1):

    df_imex_data = pd.read_excel(os.path.join(in_folder, 'MILES_IMEX.xlsx'))
    df_imex_data.set_index('hour', inplace=True)

    # clip data if less time steps are needed
    if num_of_ts is not len(df_imex_data.index):
        df_imex_data = df_imex_data.iloc[list(range(begin_ts - 1, begin_ts + num_of_ts - 1)), :]

    df_import = df_imex_data.where(df_imex_data >= 0, 0) * im_scaling_factor

    df_export = df_imex_data.where(df_imex_data <= 0, 0) * ex_scaling_factor

    export_export(df_export, out_folder, scenario_name)
    export_import(df_import, out_folder, scenario_name)


if __name__ == '__main__':

    input_folder = 'dem-load_aggregation'
    output_folder = 'data'
    scenario = 'austriaImEx_246_wind50Im05_p1300'
    base_folder = 'austriaImEx_246'
    begin_timestep = 1300
    number_of_timesteps = 1

    convert_scenario_to_pyomo(input_folder, base_folder, output_folder, scenario, begin_timestep, number_of_timesteps)

    create_import_export(input_folder, output_folder, scenario, begin_timestep, number_of_timesteps,)
