#!interpreter [optional-arg]
# -*- coding: utf-8 -*-

__author__ = ["David Cardona-Vasquez"]
__copyright__ = "Copyright 2023, Graz University of Technology"
__credits__ = ["David Cardona-Vasquez"]
__license__ = "MIT"
__maintainer__ = "David Cardona-Vasquez"
__status__ = "Development"


import data

import numpy as np
import os
import pandas as pd
import pyomo.environ as pyo

from pyomo.core.base import ConcreteModel
from pyomo.core.base import Var



def export_model_results(mdl: ConcreteModel, case: str):

    df_gen = extract_gen_units(case)
    df_vGen = extract_vGen(mdl, case, df_gen)
    df_vCost = extract_costs()

    dict_res = {'case': case, 'vGen': df_vGen}

    return dict_res


def extract_vGen(mdl: ConcreteModel, case: str, df_gen: pd.DataFrame):

    df_vGen = extract_var(mdl, 'vGen')
    df_vGen.columns = ['unit', 'period', 'value']

    df_vGen = pd.merge(df_gen, df_vGen, on=['unit'], how='left')

    df_vGen = df_vGen.astype({'bus': str, 'unit': str, 'type': str})

    return df_vGen


def extract_vNSP(mdl: ConcreteModel):

    df_vNSP = extract_var(mdl, 'vNSP')
    df_vNSP.columns = ['bus', 'period', 'value']

    return df_vNSP


def extract_gen_units(case: str, folder=""):

    renewables_data = data.load_renewables(case, ',', folder)
    thermals_data = data.load_thermals(case, ',', folder)

    renewables_data = renewables_data[['unit', 'bus', 'type']]
    thermals_data = thermals_data[['unit', 'bus']]

    thermals_data['type'] = 'thermal'

    df_gen = pd.concat([renewables_data, thermals_data])

    return df_gen


def extract_costs():

    return None


def extract_var(mdl: ConcreteModel, var_name: str):

    var = getattr(mdl, var_name)
    # get variables and values in a dataframe
    var_dict = {}
    for i in var:
        var_dict[i] = var[i].value
    idx_var = pd.MultiIndex.from_tuples(var_dict.keys())
    df_var = pd.DataFrame(index=idx_var, data=var_dict.values())
    df_var.reset_index(inplace=True, drop=False)

    return df_var


def extract_constraint(mdl: ConcreteModel, ctr_name: str):

    ctr = getattr(mdl, ctr_name)
    # get a constraint and their values in a dataframe
    ctr_dict = {}
    for c in ctr:
        ctr_dict[c] = pyo.value(ctr[c])
    idx_ctr = pd.Index(ctr_dict.keys())
    df_ctr = pd.DataFrame(index=idx_ctr, data=ctr_dict.values())
    df_ctr.reset_index(inplace=True, drop=False)

    return ctr_dict


def export_line_flow(pyo_model, case_name, folder=""):
    d_lineP = {}
    d_lineP_perc = {}

    dcOpfTestModel = 'dcOpfTestModel' in case_name

    for l in pyo_model.pL:
        aux_l = []
        aux_perc = []
        for p in pyo_model.pT:

            if dcOpfTestModel:
                flow = pyo_model.vFlow[l, p].value * pyo_model.pBP
            else:
                flow = pyo_model.vFlow[l, p].value

            aux_l.append(flow)
            aux_perc.append(flow / pyo_model.pLL[l])
        d_lineP[l] = aux_l
        d_lineP_perc[l] = aux_perc

    df_lineP = pd.DataFrame.from_dict(d_lineP, orient='columns').reset_index(drop=True)
    df_lineP.to_csv(os.path.join(folder, case_name, 'lineP.csv'), index=False, sep=';', decimal=',')

    df_line_abs_sum = df_lineP.abs().sum(axis=0).round(2)
    df_line_abs_sum.loc['Total line flows'] = df_line_abs_sum.sum()
    df_line_abs_sum.to_csv(os.path.join(folder, case_name, 'power_flow_abs_sum.csv'), sep=';', decimal=',')

    df_lineP_perc = pd.DataFrame.from_dict(d_lineP_perc, orient='columns').reset_index(drop=True)
    df_lineP_perc.to_csv(os.path.join(folder, case_name, 'lineP_perc.csv'), index=False, sep=';', decimal=',')

    df_cong_lines = df_lineP_perc[((df_lineP_perc == 1) | (df_lineP_perc == -1))]
    df_cong_lines = df_cong_lines.abs().sum(axis=0).astype(int)
    df_cong_lines.index.names = ('bus1', 'bus2')
    df_cong_lines.to_csv(os.path.join(folder, case_name, 'congested_lines.csv'), sep=';', decimal=',')

    if df_cong_lines.sum() == 0:
        print('No congested lines!')
    else:
        print('\nCongested lines detected!')

    return df_lineP_perc


def export_line_flow_np_flow(pyo_model, case_name, folder=""):
    d_lineP_pos = {}
    d_lineP_neg = {}
    d_lineP_perc = {}

    for l in pyo_model.pL:
        aux_l_pos = []
        aux_l_neg = []
        aux_perc = []
        for p in pyo_model.pT:
            aux_l_pos.append(pyo_model.vFlowPos[l, p].value)
            aux_l_neg.append(pyo_model.vFlowNeg[l, p].value)
        d_lineP_pos[l] = aux_l_pos
        d_lineP_neg[l] = aux_l_neg

    df_lineP_pos = pd.DataFrame.from_dict(d_lineP_pos, orient='columns').reset_index(drop=True)
    df_lineP_pos.to_csv(os.path.join(folder, case_name, 'lineP_pos.csv'), index=False, sep=';', decimal=',')

    df_lineP_neg = pd.DataFrame.from_dict(d_lineP_neg, orient='columns').reset_index(drop=True)
    df_lineP_neg.to_csv(os.path.join(folder, case_name, 'lineP_neg.csv'), index=False, sep=';', decimal=',')

    return df_lineP_pos, df_lineP_neg


def export_line_flow_augmented(model, case, folder):

    df_vTheta = model.vTheta.extract_var(model, 'vTheta')
    df_vTheta.pivot(index='level_1', columns='level_0', values=0)

    for l in model.pL:
        pass

    return None


def export_calculation_time(res_folder: str, case: str, model_type: str, model_build_time: float,
                            model_solving_time: float, case_property: str):

    # export calculation time
    if os.path.exists(os.path.join(res_folder, case, 'calc_time_'+model_type+'.csv')):
        df_calc_time = pd.read_csv(os.path.join(res_folder, case, 'calc_time_'+model_type+'.csv'), sep=';', decimal=',')
    else:
        df_calc_time = pd.DataFrame()
    df_calc_time['time_def'] = ['model_build', 'model_solving', 'total']
    df_calc_time[case_property] = [model_build_time, model_solving_time, model_build_time + model_solving_time]

    df_calc_time.to_csv(os.path.join(res_folder, case, 'calc_time_'+model_type+'.csv'), sep=';', decimal=',', index=False)


def extract_duals(pyo_model):
    aux_d = {}
    for k, v in pyo_model.dual.items():
        aux_name = k.parent_component().local_name
        # print(k.index(), len(k.index()), pyo.value(k.index()[1]))
        if isinstance(k.index(), tuple):
            if len(k.index()) == 3:
                aux_name = aux_name + "_" + str(pyo.value(k.index()[0])) + "_" + str(pyo.value(k.index()[1]))
            elif len(k.index()) == 2:
                aux_name = aux_name + "_" + str(pyo.value(k.index()[0]))
            # aux_name = aux_name + "_" + str(pyo.value(k.index()[0]))

        # print(aux_name)

        # if aux_d[aux_name] ist not defined, create a dict in it
        try:
            aux_d[aux_name]
        except KeyError as e:
            aux_d[aux_name] = {}
        if isinstance(k.index(), tuple):
            if len(k.index()) > 2:
                aux_d[aux_name][pyo.value(k.index()[2])] = v
            else:
                aux_d[aux_name][pyo.value(k.index()[1])] = v
        else:
            aux_d[aux_name][pyo.value(k.index())] = v

    df_duals = pd.DataFrame.from_dict(aux_d, orient='columns').reset_index(drop=False)

    return df_duals


# insert main
if __name__ == '__main__':

    results_dict = export_model_results(model, case)

    df_vGen = results_dict['vGen']

    df_vGen_agg = df_vGen.groupby(by=['bus', 'type']).agg({'value': 'sum'}).reset_index(drop=False)

    # pivot vGen_agg to have type as columns
    df_vGen_agg_pivot = df_vGen_agg.pivot(index='bus', columns='type', values='value').reset_index(drop=False)


