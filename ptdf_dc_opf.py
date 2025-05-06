#!interpreter [optional-arg]
# -*- coding: utf-8 -*-

__author__ = ["David Cardona-Vasquez, Benjamin Stöckl"]
__copyright__ = "Copyright 2023, Graz University of Technology"
__credits__ = ["David Cardona-Vasquez", "Benjamin Stöckl"]
__license__ = "MIT"
__maintainer__ = "Benjamin Stöckl"
__status__ = "Development"

import data
from data import *
import export_results
from datetime import datetime
import numpy as np
import os
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverStatus
import gen_load_aggregation as gla
from sys import exit



def ptdfDcOpf(renewables,
              thermals,
              opf,
              lines,
              cf,
              demand,
              imports,
              costs,
              periods,
              ptdf,
              name='ptdf',
              warmstarting_data_dict=None,
              ramping=False,
              hourly_cost=False,
              activate_NSP=False,
              nsp_on_slack_bus=False,
              relax_line_limits=False,
              df_bus_info=pd.DataFrame()
              ):
    '''

    model based on the transport_model, but with an implementation of the DC-OPF

    '''

    # create pyomo model
    model = pyo.ConcreteModel(name="(" + name + ")")

    # get dictionary with input data
    d_data = fill_model_data_opf(renewables, thermals, opf, lines, cf, demand, imports, periods, df_bus_info)

    # ========================== Define Parameters =====================================================================
    model.PERIODS = periods  # number of periods
    model.pT = pyo.Set(initialize=list(range(1, model.PERIODS + 1)))  # set with periods
    model.pG = pyo.Set(initialize=d_data['generators'])  # set with generators

    if len(d_data['bus']) == 0:
        model.pB = pyo.Set(initialize=['1'])
    else:
        model.pB = pyo.Set(initialize=d_data['bus'])  # set with bus

    model.pL = pyo.Set(initialize=d_data['line_lim'].keys())  # set with power lines

    model.pPTDF = pyo.Param(model.pL, model.pB, initialize=ptdf)

    # sets with generator units
    model.pGR = pyo.Set(within=model.pG, initialize=d_data['renewables'])  # set with renewable generators
    model.pGT = pyo.Set(within=model.pG, initialize=d_data['thermals'])  # set with thermal generators

    # parameters for costs
    model.pCN = pyo.Param(initialize=costs['CN'])  # transport cost
    model.pCG = pyo.Param(model.pG, initialize=costs['CG'])  # generation units' variable cost

    model.pD = pyo.Param(model.pB, model.pT, initialize=d_data['demand'], default=0,
                         domain=pyo.NonNegativeReals)  # demand at each bus for each period
    model.pCF = pyo.Param(model.pGR, model.pT, initialize=d_data['cf'])  # capacity factor for each unit, time
    model.pRU = pyo.Param(model.pGT, initialize=d_data['rmpup'], within=pyo.Any)  # set with ramp-up limits
    model.pRD = pyo.Param(model.pGT, initialize=d_data['rmpdn'], within=pyo.Any)  # set with ramp-down limits
    model.pGMAX = pyo.Param(model.pG, initialize=d_data['gmax'], within=pyo.Any)  # set with max generation for each un
    model.pGMIN = pyo.Param(model.pG, initialize=d_data['gmin'], within=pyo.Any)  # set with min generation for each un

    model.pIMP_NODE = pyo.Set(initialize=d_data['imp_nodes'])  # imports at each bus for each period
    model.pIMP = pyo.Param(model.pIMP_NODE, model.pT, initialize=d_data['imports'], default=0, domain=pyo.NonNegativeReals)  # imports at each bus for each period
    # model.vImp = pyo.Var(model.pIMP_NODE, model.pT, domain=pyo.NonNegativeReals)  # imports at each bus for each period

    # define network parameters
    model.pLL = pyo.Param(model.pL, initialize=d_data['line_lim'])                  # flow limits between bus
    model.pXline = pyo.Param(model.pL, initialize=d_data['Xline'], within=pyo.Any)  # reactance of the line
    model.pBP = pyo.Param(initialize=d_data['bpower'])                                      # parameter for base power
    model.pSlackBus = pyo.Param(initialize=d_data['slckbus'], within=pyo.Any)               # parameter for slack bus
    # model.pMAD = pyo.Param(initialize=d_data['maxangdiff'])

    model.pGB = pyo.Param(model.pG, model.pB, initialize=d_data['gb'],  # set with units connected to a bus
                          default=0)

    # parameters for variable bounds
    model.flow_limits_bounds = pyo.Param(model.pL, initialize=d_data['line_lim'])

    # ========================== Define Variables ======================================================================
    model.vGen = pyo.Var(model.pG, model.pT, domain=pyo.NonNegativeReals,
                         initialize=(None if warmstarting_data_dict is None else warmstarting_data_dict['vGen']))  # generation from each unit at each time

    if activate_NSP and nsp_on_slack_bus:
        model.vNSP = pyo.Var(model.pB, model.pT, domain=pyo.NonNegativeReals)  # non-supplied power at bus at each time
        model.pCNSP = pyo.Param(model.pB, initialize=costs['CNSP'], default=6000)  # non-supplied power cost at each bus

        for i in model.vNSP:
            model.vNSP[i].fix(0)
        model.vNSP[model.pSlackBus, :].unfix()
    elif activate_NSP:
        model.vNSP = pyo.Var(model.pB, model.pT, domain=pyo.NonNegativeReals)  # non-supplied power at bus at each time
        model.pCNSP = pyo.Param(model.pB, initialize=costs['CNSP'], default=6000)  # non-supplied power cost at each buss

    #define slack variable for line limits
    model.slack = pyo.Var(model.pL, model.pT, domain=pyo.NonNegativeReals)

    if not relax_line_limits:
        for i in model.slack:
            model.slack[i].fix(0)


    # ========================== Objective Function ====================================================================
    def obj_rule(mdl):
        return sum(
            + sum(mdl.vGen[g, p] * mdl.pCG[g] for g in mdl.pG)
            + (sum(mdl.vNSP[b, p] * mdl.pCNSP[b] for b in mdl.pB) if activate_NSP else 0)
            + sum(mdl.slack[l, p] * 10000 for l in mdl.pL)  # try to add slack variable for linie limit constraint
            for p in mdl.pT)

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # ========================== Constraints ===========================================================================
    def eHourly_Cost(mdl, p):
        return (
                + sum(mdl.vGen[g, p] * mdl.pCG[g] for g in mdl.pG)
                + (sum(mdl.vNSP[b, p] * mdl.pCNSP[b] for b in mdl.pB) if activate_NSP else 0)
                >= 0
                )

    if hourly_cost:
        model.eHourly_Cost = pyo.Constraint(model.pT, rule=eHourly_Cost)

    # ------------------- bus balance constraints ----------------------------------------------------------------------
    def ePowerBalance(mdl, p):
        ''' Power balance for whole system '''
        return (
                + sum(
                    + mdl.pD[b, p]
                    - (mdl.pIMP[b, p] if b in mdl.pIMP_NODE else 0)
                    - (mdl.vNSP[b, p] if activate_NSP else 0)
                    # - sum(mdl.vGen[g, p] * mdl.pGB[g, b] for g in mdl.pG)
                    - sum((mdl.vGen[g, p] if mdl.pGB[g, b] == 1 else 0) for g in mdl.pG)
                for b in mdl.pB)
                # - sum(mdl.vGen[g, p] * mdl.pGB[g, b] for g in mdl.pG) for b in mdl.pB)
                == 0
                )

    model.ePowerBalance = pyo.Constraint(model.pT, rule=ePowerBalance)

    # ------------------- generation constraints -----------------------------------------------------------------------
    def eMaxProd(mdl, g, p):
        if g in mdl.pGR:
            return mdl.vGen[g, p] <= mdl.pGMAX[g] * mdl.pCF[g, p]
        else:
            return mdl.vGen[g, p] <= mdl.pGMAX[g]

    model.eMaxProd = pyo.Constraint(model.pG, model.pT, rule=eMaxProd)

    def eMinProd(mdl, g, p):
        if mdl.pGMIN[g] != 0:
            return mdl.vGen[g, p] >= mdl.pGMIN[g]
        else:
            return pyo.Constraint.Skip

    model.eMinProd = pyo.Constraint(model.pG, model.pT, rule=eMinProd)

    # ------------------- power flow constraints -----------------------------------------------------------------------
    def eMaxTransport(mdl, b1, b2, p):

        return (
                sum(mdl.pPTDF[(b1, b2), b] * (sum((mdl.vGen[g, p] if mdl.pGB[g, b] == 1 else 0) for g in mdl.pG)
                                             + (mdl.pIMP[b, p] if b in mdl.pIMP_NODE else 0)
                                             + (mdl.vNSP[b, p] if activate_NSP else 0)
                                             - mdl.pD[b, p]
                                             ) for b in mdl.pB)
                <= mdl.pLL[b1, b2] + mdl.slack[b1, b2, p]
                )

    model.eMaxTransport = pyo.Constraint(model.pL, model.pT, rule=eMaxTransport)

    def eMinTransport(mdl, b1, b2, p):

        return (sum(mdl.pPTDF[(b1, b2), b] * (sum((mdl.vGen[g, p] if mdl.pGB[g, b] == 1 else 0) for g in mdl.pG)
                                             + (mdl.pIMP[b, p] if b in mdl.pIMP_NODE else 0)
                                             + (mdl.vNSP[b, p] if activate_NSP else 0)
                                             - mdl.pD[b, p]
                                             ) for b in mdl.pB)
                >= -mdl.pLL[b1, b2] - mdl.slack[b1, b2, p]
                )

    model.eMinTransport = pyo.Constraint(model.pL, model.pT, rule=eMinTransport)


    model.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return model



def calculate_power_flow_original_grid(case, data_folder, df_vGen, periods, df_NSP, original_case='', df_vGen_agg=None,
                                       df_vDemAdd=None):

    '''

    Calculation of the power flow in the original grid based on the dispatch determined in aggregated grid

    case:           name of the aggregated case
    df_vGen:        DataFrame with the dispatch of the aggregated case
    periods:        number of periods
    original_case:  name of the original case (if not specified it is the same as the aggregated case)

    '''

    # import the PTDF matrix of the full model
    df_ptdf_full = pd.read_csv(os.path.join(data_folder, 'data', (original_case if original_case != '' else case),
                                            'PTDF.csv'), sep=';', decimal=',')
    df_ptdf_full = df_ptdf_full.astype({'bus1': str, 'bus2': str})
    df_ptdf_full.set_index(['bus1', 'bus2'], inplace=True)

    # DataFrame with original buses and unit names for thermal & renewable units
    df_orig_res = pd.read_csv(os.path.join(data_folder, 'data', (original_case if original_case != '' else case),
                                           'renewables.csv'), sep=';', decimal=',')[['unit', 'bus']]
    df_orig_th = pd.read_csv(os.path.join(data_folder, 'data', (original_case if original_case != '' else case),
                                          'thermals.csv'), sep=';', decimal=',')[['unit', 'bus']]

    # DataFrame with original buses and unit names for all generating units
    df_orig_gen = pd.concat([df_orig_res, df_orig_th], axis=0)
    df_orig_gen.set_index('unit', inplace=True)
    df_orig_gen['bus'] = df_orig_gen['bus'].astype(int).astype(str)
    # df_orig_gen = df_orig_gen.astype({'bus': str})
    df_orig_gen.rename(columns={'bus': 'orig_bus'}, inplace=True)
    df_vGen = df_vGen.join(df_orig_gen, on='unit', how='left')

    df_assign_node = df_vGen[['bus', 'orig_bus']]

    # get the sum of the production at each bus of the original grid
    df_vGen.set_index(['unit', 'orig_bus'], inplace=True)
    df_vGen = df_vGen.pivot(columns='period', values='value')
    df_vGen.reset_index(inplace=True)
    df_vGen.drop(columns={'unit'}, inplace=True)
    df_vGen['orig_bus'] = df_vGen['orig_bus'].astype(int).astype(str)
    df_vGen = df_vGen.groupby('orig_bus').sum()



    # get import data as df
    #TODO: check if the import data is correct

    # import_data = load_import(original_case, ',', data_folder)
    # df_import_data = data.import_data_to_df(import_data, periods)
    # df_import_data.rename(columns={1: 'inj'}, inplace=True)

    orig_demand_data = load_demand(original_case, ',', data_folder)
    df_dem = data.demand_dict_to_df(orig_demand_data)

    df_inj = df_vGen.sub(df_dem, fill_value=0)
    # df_inj = df_inj.add(df_import_data, fill_value=0)

    # add generation of aggregated model to injection
    if df_vGen_agg is not None:
        df_vGen_agg['bus'] = df_vGen_agg['bus'].astype(int).astype(str)
        df_vGen_agg_bus = df_vGen_agg.join(df_orig_gen, on='unit', how='left')
        df_vGen_agg_bus = df_vGen_agg_bus[['period', 'value', 'unit', 'orig_bus']].set_index(['unit', 'orig_bus'])
        df_vGen_agg_bus = df_vGen_agg_bus.pivot(columns='period', values='value')
        df_vGen_agg_bus.reset_index(inplace=True)
        df_vGen_agg_bus.drop(columns=['unit'], inplace=True)
        df_vGen_agg_bus['orig_bus'] = df_vGen_agg_bus['orig_bus'].astype(int).astype(str)
        df_vGen_agg_bus = df_vGen_agg_bus.groupby('orig_bus').sum()

        df_inj = df_inj.add(df_vGen_agg_bus, fill_value=0)

    # subtract additional demand to the injection, if it exists
    if df_vDemAdd is not None:

        df_vDemAdd.rename(columns={'level_0': 'unit', 'level_1': 'period', 0: 'value'}, inplace=True)
        df_vDemAdd = df_vDemAdd.join(df_orig_gen, on='unit', how='left').set_index(['unit', 'orig_bus'])
        df_vDemAdd = df_vDemAdd.pivot(columns='period', values='value')
        df_vDemAdd.reset_index(inplace=True)
        df_vDemAdd.drop(columns=['unit'], inplace=True)
        df_vDemAdd['orig_bus'] = df_vDemAdd['orig_bus'].astype(int).astype(str)
        df_vDemAdd = df_vDemAdd.groupby('orig_bus').sum()

        df_inj = df_inj.sub(df_vDemAdd, fill_value=0)

    # load NSP data
    # if df_NSP is not None:
    #     df_NSP = df_NSP.pivot_table(values='value', index='bus', columns='period')
    #     df_inj = df_inj.add(df_NSP, fill_value=0)

    # add because there is no demand at this node, but there is a line to the node

    # check if the sum of the injections in every period is zero
    if (df_inj.sum().abs() > 1e-6).any():
        exit('Power balance not fulfilled!' + str(df_inj.sum()))

    # check if for every bus an injection value is defined
    if len(df_inj.index) < len(df_ptdf_full.columns):
        for i in df_ptdf_full.columns:
            if i not in df_inj.index:
                df_inj.loc[i] = 0


    df_orig_power_flow = df_ptdf_full @ df_inj

    return df_orig_power_flow


def get_nodal_injection(df_vGen, import_data, demand_data, df_NSP, model):

    # get sum of production at each bus
    df_inj = df_vGen.drop(columns=['type', 'unit'])

    # get import data as df
    df_import_data = data.import_data_to_df(import_data, model.PERIODS)

    df_inj = df_inj.pivot_table(values='value', index='bus', columns='period', aggfunc="sum")

    if model.pSlackBus.value not in df_inj.index:
        df_inj.loc[model.pSlackBus.value] = [0.0 for _ in range(model.PERIODS)]
    df_inj.sort_index(inplace=True)

    df_dem = data.demand_dict_to_df(demand_data)

    # if NSP exists add to the injection
    if df_NSP is not None:
        df_NSP = df_NSP.pivot_table(values='value', index='bus', columns='period')
        df_inj = df_inj.add(df_NSP, fill_value=0)


    # set same column names
    df_dem.columns = df_inj.keys()
    df_dem.index.name = 'bus'

    # get net injection for each bus
    df_inj = df_inj.sub(df_dem, fill_value=0)
    df_inj = df_inj.add(df_import_data, fill_value=0)
    df_inj.fillna(0, inplace=True)

    return df_inj


def get_production_error_by_technology(df_vGen, df_diff_vGen):

    tec_error = {}
    # get production types
    prod_tec = list(set([s.split('_')[0] for s in df_diff_vGen.index.to_list()])) #  set only takes unique values from list
    df_vGen.index = df_vGen.index.astype(str)

    for tec in prod_tec:
        df_diff = df_diff_vGen[[i for i in df_diff_vGen.index.to_list() if i.startswith(tec)]]
        df_vGen_tec = df_vGen.loc[[i for i in df_vGen.index.to_list() if i.startswith(tec)]]
        df_diff = df_diff / df_vGen_tec['max']  # get percentage error
        tec_error[tec] = np.sqrt(sum(df_diff ** 2) / len(df_diff))
        tec_error[tec + '_max_dev'] = max(abs(df_diff))

    return tec_error


def export_model_statistics(model, case, original_case, number_of_clusters, res_folder, d_flow_error, d_vGen_error, tec_error, l_violated_lines):

    gla.check_if_folder_exists_and_create(res_folder)

    df_agg_stats = gla.load_file_or_gen_df(res_folder, 'aggregation_stats.csv',
                                           ['timestamp', 'aggregated_case', 'original_case',
                                            'number_of_agg_nodes', 'obj_func_value', 'gen_agg_type',
                                            'cluster_method']
                                           + list(d_flow_error.keys())
                                           + list(d_vGen_error.keys())
                                           + list(tec_error.keys())
                                           + ['violated_line_lims'],
                                           folder_name='_'.join(case.split('_')[:-2]), )
    df_agg_stats = df_agg_stats.astype({'flow_rel_error': float})

    dict = {'timestamp': datetime.now(),
            'aggregated_case': case,
            'original_case': original_case,
            'number_of_agg_nodes': number_of_clusters,
            'obj_func_value': pyo.value(model.obj),
            'gen_agg_type': case.split('_')[-1],
            'cluster_method': case.split('_')[-3],
            'violated_line_lims': str([l1 + '->' + l2 for l1, l2 in l_violated_lines]),
            }
    dict = {**dict, **d_flow_error, **d_vGen_error, **tec_error}  # combine the dicts to one

    df = pd.DataFrame(dict, index=[0])
    df_agg_stats = pd.concat([df_agg_stats, df], ignore_index=True)

    gla.check_if_folder_exists_and_create(res_folder, '_'.join(case.split('_')[:-2]))

    # export the stats of the case to csv
    df_agg_stats.to_csv(os.path.join(res_folder, '_'.join(case.split('_')[:-2]), 'aggregation_stats.csv'), index=False,
                        sep=';', decimal=',')


def get_generation_difference(original_case, res_folder, data_folder, aggregated_case='', df_vGen=pd.DataFrame(),
                              df_gen_data=pd.DataFrame()):

    # get the production of the full model
    df_orig_vGen = pd.read_csv(os.path.join(res_folder, original_case, 'vGen.csv'), sep=';', decimal=',')
    df_orig_vGen.drop(columns={'Unnamed: 0'}, inplace=True)
    df_orig_vGen.set_index('unit', inplace=True)

    # load data if missing
    if df_vGen.empty and aggregated_case != '':
        df_vGen = pd.read_csv(os.path.join(res_folder, aggregated_case, 'vGen.csv'), sep=';', decimal=',')

    if df_gen_data.empty and aggregated_case != '':
        sep_dec = ','
        res_data = load_renewables(aggregated_case, sep_dec, data_folder)
        th_data = load_thermals(aggregated_case, sep_dec, data_folder)

        df_gen_data = pd.concat([res_data, th_data], axis=0)
        df_gen_data.set_index('unit', inplace=True)

    df_agg_vGen = df_vGen.set_index('unit')

    # get the difference of the production of agg/orig system
    df_diff_vGen = df_orig_vGen['value'] - df_agg_vGen['value']
    df_diff_vGen.fillna(0, inplace=True)
    df_diff_vGen_perc = df_diff_vGen / df_gen_data['max']  # the difference of the generation relative to the installed capacity

    return df_diff_vGen, df_diff_vGen_perc, df_gen_data


def get_generation_error(original_case, res_folder, data_folder, df_gen_data=pd.DataFrame(),
                         df_vGen=pd.DataFrame(), aggregated_case=''):

    df_diff_vGen, df_diff_vGen_perc, df_gen_data = get_generation_difference(original_case, res_folder, data_folder,
                                                                             aggregated_case, df_vGen, df_gen_data)

    # the vGen error gets calculated by the root-mean-square of the generation differences
    # vGen_rel_error = np.sqrt(sum(df_diff_vGen_perc ** 2) / len(df_diff_vGen))

    # mean of the absolute values of the relative errors
    vGen_rel_error = sum(df_diff_vGen_perc.abs()) / len(df_diff_vGen_perc)
    vGen_rel_error_max_dev = max(abs(df_diff_vGen_perc))  # maximum deviation of generation in aggregated and original

    tec_error = get_production_error_by_technology(df_gen_data, df_diff_vGen)

    d_vGen_error = {'vGen_rel_error': vGen_rel_error, 'vGen_rel_error_max_dev': vGen_rel_error_max_dev}

    return d_vGen_error, tec_error


def run_model(case: str,
              periods: int,
              power_flow_cost=0,
              mult_with_Xline=False,
              ramping=True,
              res_folder='results',
              data_folder='',
              original_case='',
              number_of_clusters=0,
              activate_NSP=False,
              relax_line_limits=False,
              export_model_definition=False,
              warmstarting_case=''
              ):

    # define Input case name
    s_aux = case.split('_')
    input_case = case

    if 'basis' in s_aux:
        input_case = case
        period_weights = pd.read_csv(os.path.join(data_folder, 'data', input_case, 'basis_weights.csv'), sep=';', decimal=',')

    #
    # ----------------------------- Load Model Data --------------------------------------------------------------------

    # load data renewables, thermals, capacity factors, demand, network, line data and opf parameters
    sep_dec = ','
    renewables_data = load_renewables(input_case, sep_dec, data_folder)
    thermals_data = load_thermals(input_case, sep_dec, data_folder)
    cf_data = load_cf(input_case, sep_dec, data_folder)
    demand_data = load_demand(input_case, sep_dec, data_folder)
    line_data = load_lines(input_case, sep_dec, data_folder)
    opf_data = load_opf_parameters(input_case, sep_dec, data_folder)
    import_data = load_import(input_case, sep_dec, data_folder)
    print('Data loaded!')

    df_ptdf = pd.read_csv(os.path.join(data_folder, 'data', input_case, 'ptdf.csv'), sep=';', decimal=',')
    df_ptdf = df_ptdf.astype({'bus1': str, 'bus2': str})

    ptdf = {}

    # make dict from ptdf dataframe
    for idx, row in df_ptdf.iterrows():
        for b in df_ptdf.columns[2:]:
            ptdf[str(row['bus1']), str(row['bus2']), b] = row[b]

    # load generation costs and cost for non supplied power
    folder_content = os.listdir(os.path.join(data_folder, 'data', input_case))
    if 'gen_costs.csv' in folder_content and 'nsp_costs.csv' in folder_content:
        gen_costs = pd.read_csv(os.path.join(data_folder, 'data', input_case, 'gen_costs.csv'), sep=';', decimal=sep_dec)
        gen_costs.set_index('resource', inplace=True)
        nsp_costs = pd.read_csv(os.path.join(data_folder, 'data', input_case, 'nsp_costs.csv'), sep=';', decimal=sep_dec)
        nsp_costs = nsp_costs.astype({'zone': 'str', 'CNSP': 'float'})
        nsp_costs.set_index('zone', inplace=True)
    elif 'gen_costs.xlsx' in folder_content and 'nsp_costs.xlsx' in folder_content:
        gen_costs = pd.read_excel(os.path.join(data_folder, 'data', case, 'gen_costs.xlsx'), index_col=0)
        nsp_costs = pd.read_excel(os.path.join(data_folder, 'data', case, 'nsp_costs.xlsx'), index_col=0)
    else:
        exit('Can\'t load files gen_costs.* or nsp_costs.*!')

    costs = {'CN': 0.1}
    costs['CNSP'] = list(nsp_costs.to_dict().items())[0][1]
    costs['CG'] = list(gen_costs.to_dict().items())[0][1]

    # for warmstarting the model load the unit dispatch and the power flow in the original grid
    warmstarting_data = get_vGen_warmstarting_data(res_folder, warmstarting_case)


    #
    # ----------------------------- Create Model -----------------------------------------------------------------------

    import time
    start = time.time()

    # create model depending on model type (transportModel, dcOpf, augmentedDcOpf)

    model = ptdfDcOpf(renewables_data, thermals_data, opf_data, line_data, cf_data, demand_data, import_data,
                      costs, periods, ptdf, case, warmstarting_data,
                      ramping=ramping,
                      hourly_cost=False,
                      activate_NSP=activate_NSP,
                      relax_line_limits=relax_line_limits,
                      )

    #
    # ------------------------------- Solve Model ----------------------------------------------------------------------
    # hand model to the solver
    end = time.time()
    model_build_time = end - start  # time to build the model
    print('Model created in: {:.2f} seconds'.format(model_build_time))
    solver = pyo.SolverFactory('gurobi_persistent')

    gurobi_solver = 'barrier'
    solver.options['Method'] = 2

    if export_model_definition:
        with open(os.path.join(res_folder, case, 'model_description.txt'), 'w') as f:
            model.pprint(ostream=f)
        f.close()


    start = time.time()
    solver.set_instance(model)

    res = solver.solve(tee=True)
    end = time.time()


    model_solving_time = end - start  # time to solve the model
    print('Model solved in: {:.2f} seconds'.format(model_solving_time))
    print('Solver termination condition was: {}'.format(res.solver.termination_condition))
    print('Objective function value was {:.2f}'.format(pyo.value(model.obj)))

    #
    # ------------------------------- model results --------------------------------------------------------------------
    res.write()
    model.solutions.load_from(res)

    # check if model was feasible otherwise exit
    if res.solver.status == SolverStatus.ok:
        pass
    else:
        print('Model was infeasible!')
        exit()

    # check if folder 'results' exits --> create if not
    gla.check_if_folder_exists_and_create(res_folder, case)

    #export vNSP results
    if activate_NSP:
        df_vNSP = export_results.extract_var(model, 'vNSP')
        df_vNSP.columns = ['bus', 'period', 'value']
        df_vNSP.to_csv(os.path.join(res_folder, case, 'NSP.csv'), sep=';', decimal=',', index=False)
    else:
        df_vNSP = None

    # export calculation time
    export_results.export_calculation_time(res_folder, case, 'PTDF_warmstarting', model_build_time,
                                           model_solving_time,
                                           warmstarting_case)

    df_model_stats = pd.DataFrame()
    df_model_stats.loc[0, 'model_parameter'] = 'ofv'
    df_model_stats.loc[0, 'parameter_value'] = pyo.value(model.obj)

    # export objective function value
    df_model_stats.to_csv(os.path.join(res_folder, case, 'model_stats.csv'), sep=';', decimal=',', index=False)


    # export vGen results
    df_gen = export_results.extract_gen_units(input_case, data_folder)
    df_vGen = export_results.extract_vGen(model, case, df_gen)

    df_vGen.to_csv(os.path.join(res_folder, case, 'vGen.csv'), sep=';', decimal=',')

    # ------------------------------- analyze duals and basis identification -------------------------------------------

    duals = export_results.extract_duals(model)
    duals.to_csv(os.path.join(res_folder, case, 'duals.csv'), index=False, sep=';', decimal=',')

    # change ptdf df to indexed df
    df_ptdf.set_index([df_ptdf['bus1'], df_ptdf['bus2']], inplace=True)
    df_ptdf.drop(columns=['bus1', 'bus2'], inplace=True)
    df_ptdf.sort_index(inplace=True)

    # make array from df_ptdf dataframe
    ptdf = df_ptdf.to_numpy()

    # --------- BEGIN Evaluate power flows in aggregated Grid and original Grid ----------------------------------------

    # get nodal injections of  aggregated grid
    df_inj = get_nodal_injection(df_vGen, import_data, demand_data, df_vNSP, model)
    df_inj.to_csv(os.path.join(res_folder, case, 'node_inj.csv'), sep=';', decimal=',')

    #check if for every node a injection is defined
    if len(df_inj.index) < len(df_ptdf.columns):
        for i in df_ptdf.columns:
            if i not in df_inj.index:
                df_inj.loc[i] = 0

    # calc power flow in aggregated grid and export to csv
    df_flow = df_ptdf @ df_inj
    df_flow.rename(columns={'1': 'flow [MW]'}, inplace=True)
    df_flow.T.to_csv(os.path.join(res_folder, case, 'lineP.csv'), sep=';', decimal=',', index=False)

    # get relative power flow
    df_line_limits = line_data.set_index(['bus1', 'bus2'])
    df_flow_perc = df_flow.div(df_line_limits['Pmax'], axis=0)
    df_flow_perc.T.to_csv(os.path.join(res_folder, case, 'lineP_perc.csv'), sep=';', decimal=',', index=False)

    # determine the difference of the power flow in the aggregated grid and the original grid
    if original_case != '':
        # calculate power flows in original model
        # ATTENTION: in the function some nodes gets dropped from the ptdf AND the inj df, because there is no demand
        df_orig_power_flow = calculate_power_flow_original_grid(case, data_folder, df_vGen, periods, df_vNSP,
                                                                original_case)
        df_orig_power_flow.to_csv(os.path.join(res_folder, case, 'lineP_orig.csv'), sep=';', decimal=',')

        # load line limits of full grid
        df_orig_line_limits = load_lines(original_case, sep_dec, data_folder)
        df_orig_line_limits.set_index(['bus1', 'bus2'], inplace=True)

        # calculate how much the line limits are violated
        # calculate relative power flow in original grid
        df_orig_power_flow_perc = df_orig_power_flow.div(df_orig_line_limits['Pmax'], axis=0)

        # filter lines with power flows below or above the limits
        df_orig_power_flow_violated_limits = df_orig_power_flow_perc[(df_orig_power_flow_perc > 1) |
                                                                     (df_orig_power_flow_perc < -1)]

        # get the lines with violated line limits
        df_orig_limit_violation = df_orig_power_flow_perc.abs() >= 1
        l_violated_lines = df_orig_power_flow.index[df_orig_limit_violation.any(axis=1)].to_list()
        print('\nLines with violated line limits:' + str(l_violated_lines) + '\n')

        # calculate the percentage how much the line limits are violated
        df_orig_power_flow_violated_limits = df_orig_power_flow_violated_limits.abs() - 1
        ll_violation_error_max = df_orig_power_flow_violated_limits.max().max()


        # calculate difference to original power flow with an error metric!
        df_orig_case_power_flow = pd.read_csv(os.path.join(res_folder, original_case, 'lineP.csv'),
                                              sep=';', decimal=',', header=[0, 1])
        df_orig_case_power_flow.index = range(1, periods + 1)
        df_orig_case_power_flow.index.name = 'period'
        df_orig_case_power_flow = df_orig_case_power_flow.T
        df_orig_case_power_flow.index.names = ['bus1', 'bus2']

        df_diff_power_flow = df_orig_power_flow - df_orig_case_power_flow
        df_diff_power_flow_perc = df_diff_power_flow.div(df_orig_line_limits['Pmax'], axis=0)

        # flow_rel_error = sum(df_diff_power_flow_perc.abs()) / len(df_diff_power_flow_perc)
        flow_rel_error = (df_diff_power_flow_perc.abs().sum().sum() / df_diff_power_flow_perc.size)
        flow_rel_error_max_dev = df_diff_power_flow_perc.abs().max().max()  # maximum deviation of power flow in aggregated and original

        # calculate difference in flows between original and aggregated model for the congested lines
        df_om_power_flow_perc = pd.read_csv(os.path.join(res_folder, original_case, 'lineP_perc.csv'),
                                            sep=';', decimal=',', header=[0, 1])
        df_om_power_flow_perc = df_om_power_flow_perc.T


        d_flow_error = {'flow_rel_error': flow_rel_error, 'flow_rel_error_max_dev': flow_rel_error_max_dev,
                        'll_violation_error_max': ll_violation_error_max}

    # --------- END Evaluate power flows in aggregated Grid and original Grid ------------------------------------------
    # --------- BEGIN Evaluate production in aggregated grid and original Grid ----------------------------------------

    if original_case != '':  # only compute, when an original case is given

        df_gen_data = pd.concat([renewables_data, thermals_data], axis=0)
        df_gen_data.set_index('unit', inplace=True)

        # calculate error of generation values
        d_vGen_error, tec_error = get_generation_error(original_case, res_folder, data_folder, df_gen_data=df_gen_data,
                                                       df_vGen=df_vGen)
        # export model statistics
        export_model_statistics(model, case, original_case, number_of_clusters, res_folder, d_flow_error, d_vGen_error,
                            tec_error, l_violated_lines)
    # --------- END Evaluate production in aggregated grid and original Grid ------------------------------------------


    return {}


if __name__ == '__main__':

    all_number_of_clusters = False
    periods = 1
    number_of_clusters = -1
    tec_type = 'tecAss'
    clustering_method = 'lmpAnacPTDF'
    orig_case = 'IEEE_24_p19'


    ''' 
    For large scenarios define file path outside of the git repo, because files with sizes bigger than 100 MB 
    are not uploaded!!
    '''
    results_folder = os.path.join('..', '..', 'GridAggregation', 'results')
    data_folder = os.path.join('..', '..', 'GridAggregation')

    case = f'{orig_case}_red_{clustering_method}_{number_of_clusters}_{tec_type}'

    gla.check_if_folder_exists_and_create(results_folder, folder=case)
    print('run agg model: ', results_folder, str(number_of_clusters))

    results = run_model(case, periods,
                        res_folder=results_folder,
                        data_folder=data_folder,
                        original_case=orig_case,
                        number_of_clusters=number_of_clusters,
                        activate_NSP=False,
                        ramping=False,
                        relax_line_limits=False,
                        )
