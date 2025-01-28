#!interpreter [optional-arg]
# -*- coding: utf-8 -*-

__author__ = ["David Cardona-Vasquez, Benjamin Stöckl"]
__copyright__ = "Copyright 2023, Graz University of Technology"
__credits__ = ["David Cardona-Vasquez", "Benjamin Stöckl"]
__license__ = "MIT"
__maintainer__ = "David Cardona-Vasquez"
__status__ = "Development"


from data import *
import export_results
import numpy as np
import os
import pandas as pd
import pyomo.environ as pyo
import gen_load_aggregation as gla
from sys import exit
import ptdf_dc_opf as pdo



def dc_opf_model(renewables, thermals, opf, lines, cf, demand, imports, costs, periods, name='transport', ramping=True,
                 hourly_cost=True, single_node=False, df_vGen_agg=None,
                 activate_NSP=True):

    '''

    model based on the transport_model, but with an implementation of the DC-OPF

    '''

    ### CHANGE IF NEEDED
    # select if the flow-/anglebounds should be defined by variable bounds or constraints
    # True -> variable bounds | False -> constraints
    useFlowBounds = False
    useAngleBounds = True

    nsp_on_slack_bus = False  # usually False

    # create pyomo model
    model = pyo.ConcreteModel(name="(" + name + ")")

    # get dictionary with input data
    d_data = fill_model_data_opf(renewables, thermals, opf, lines, cf, demand, imports, periods)

    # ========================== Define Parameters =====================================================================
    model.PERIODS = periods  # number of periods
    model.pT = pyo.Set(initialize=list(range(1, model.PERIODS + 1)))  # set with periods
    model.pG = pyo.Set(initialize=d_data['generators'])  # set with generators
    model.pB = pyo.Set(initialize=d_data['bus'])  # set with bus
    model.pL = pyo.Set(initialize=d_data['line_lim'].keys())  # set with power lines

    # sets with generator units
    model.pGR = pyo.Set(within=model.pG, initialize=d_data['renewables'])  # set with renewable generators
    model.pGT = pyo.Set(within=model.pG, initialize=d_data['thermals'])  # set with thermal generators

    # parameters for costs
    if activate_NSP:
        model.pCNSP = pyo.Param(model.pB, initialize=costs['CNSP'], default=6000)  # non-supplied power cost at each bus
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
    model.pLL = pyo.Param(model.pL, initialize=d_data['line_lim'])  # flow limits between bus
    model.pXline = pyo.Param(model.pL, initialize=d_data['Xline'])  # reactance of the line
    model.pBP = pyo.Param(initialize=d_data['bpower'])  # parameter for base power
    model.pSlackBus = pyo.Param(initialize=d_data['slckbus'], within=pyo.Any)  # parameter for slack bus
    model.pMAD = pyo.Param(initialize=d_data['maxangdiff'])

    model.pGB = pyo.Param(model.pG, model.pB, initialize=d_data['gb'],  # set with units connected to a bus
                          default=0)

    # parameters for variable bounds
    model.flow_limits_bounds = pyo.Param(model.pL, initialize=d_data['line_lim'])

    # ========================== Define Variables ======================================================================
    model.vGen = pyo.Var(model.pG, model.pT, domain=pyo.NonNegativeReals, initialize=0)  # generation from each unit at each time
    model.vNSP = pyo.Var(model.pB, model.pT, domain=pyo.NonNegativeReals, initialize=0)  # non-supplied power at bus at each time

    model.vDemAdd = pyo.Var(model.pB, model.pT, domain=pyo.NonNegativeReals, initialize=0)  # demand shed at bus at each time


    # fix_generation_variables by results from aggregated model to calculate redispatch costs in the original model


    print(df_vGen_agg is not None)
    # fix the generation variables to results
    if df_vGen_agg is not None:
        print('Fix generation values!')
        for _, row in df_vGen_agg.iterrows():
            model.vGen[row['unit'], row['period']].fix(row['value'])
    else:
        model.vDemAdd.fix(0)
        if not activate_NSP:
            print('NSP is not activated!')
            model.vNSP.fix(0)



    if nsp_on_slack_bus:
        for i in model.vNSP:
            model.vNSP[i].fix(0)
        model.vNSP[model.pSlackBus, :].unfix()


    if useFlowBounds:
        def flow_bounds(model, b1, b2, p):
            return(- model.flow_limits_bounds[(b1, b2)], model.flow_limits_bounds[(b1, b2)])
            #return (None, None)
        model.vFlow = pyo.Var(model.pL, model.pT, domain=pyo.Reals, bounds=flow_bounds)  # flow between bus
    else:
        model.vFlow = pyo.Var(model.pL, model.pT, domain=pyo.Reals)  # flow between bus

    # model.vPosFlow = pyo.Var(model.pL, model.pT, domain=pyo.NonNegativeReals)#,

    if useAngleBounds:
        def angle_bounds(mdl, b, p):
            if b == pyo.value(mdl.pSlackBus):
                return (0, 0)
            else:
                return (-d_data['maxangdiff'], d_data['maxangdiff'])
                # return (None, None)

        model.vTheta = pyo.Var(model.pB, model.pT, domain=pyo.Reals, bounds=angle_bounds)

    else:
        model.vTheta = pyo.Var(model.pB, model.pT, domain=pyo.Reals)

    # model.vPosTheta = pyo.Var(model.pB, model.pT, domain=pyo.Reals)
    #model.vTheta[model.pSlackBus, :].fix(0)  # set slack bus angle to 0

    # ========================== Objective Function ====================================================================
    def obj_rule(mdl):
        return sum(
            sum(mdl.vGen[g, p] * mdl.pCG[g] for g in mdl.pG) + sum((mdl.vNSP[b, p] * mdl.pCNSP[b] if activate_NSP else 0)
                                                                   + mdl.vDemAdd[b, p] * 1 for b in mdl.pB)
            for p in mdl.pT)

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # ========================== Constraints ===========================================================================
    def eHourly_Cost(mdl, p):
        return (sum(mdl.vGen[g, p] * mdl.pCG[g] for g in mdl.pG)
                + sum((mdl.vNSP[b, p] * mdl.pCNSP[b] if activate_NSP else 0) for b in mdl.pB)
                >= 0)

    if hourly_cost:
        model.eHourly_Cost = pyo.Constraint(model.pT, rule=eHourly_Cost)

    # ------------------- bus balance constraints ----------------------------------------------------------------------
    def eBalance_Bus(mdl, b1, p):
        inflow_sum = sum(mdl.vFlow[b2, b1, p] for b2, x in mdl.pL if x == b1)
        outflow_sum = sum(mdl.vFlow[b1, b2, p] for x, b2 in mdl.pL if x == b1)

        return (
                mdl.pD[b1, p]
                - (mdl.pIMP[b1, p] if b1 in mdl.pIMP_NODE else 0)
                + mdl.vDemAdd[b1, p]
                - inflow_sum
                + outflow_sum
                - (mdl.vNSP[b1, p] if activate_NSP else 0)
                - sum(mdl.vGen[g, p] * mdl.pGB[g, b1] for g in mdl.pG)
                == 0
                )

    model.eBalance_Bus = pyo.Constraint(model.pB, model.pT, rule=eBalance_Bus)

    # ------------------- generation constraints -----------------------------------------------------------------------
    def eMaxProd(mdl, g, p):
        if g in mdl.pGR:
            return mdl.vGen[g, p] <= mdl.pGMAX[g] * mdl.pCF[g, p] #+ 1
        else:
            return mdl.vGen[g, p] <= mdl.pGMAX[g] #+ 1

    model.eMaxProd = pyo.Constraint(model.pG, model.pT, rule=eMaxProd)

    def eMinProd(mdl, g, p):
        if mdl.pGMIN[g] != 0:
            return mdl.vGen[g, p] >= mdl.pGMIN[g]
        else:
            return pyo.Constraint.Skip

    model.eMinProd = pyo.Constraint(model.pG, model.pT, rule=eMinProd)

    # ------------------- power flow constraints -----------------------------------------------------------------------
    def ePowerFLow(mdl, b1, b2, p):

        return mdl.vFlow[b1, b2, p] == ((mdl.vTheta[b1, p] - mdl.vTheta[b2, p]) * mdl.pBP) / mdl.pXline[b1, b2] #+ 1

    model.ePowerFlow = pyo.Constraint(model.pL, model.pT, rule=ePowerFLow)

    def eMaxTransport(mdl, b1, b2, p):

       return mdl.vFlow[b1, b2, p] <= mdl.pLL[b1, b2]

    if not useFlowBounds:
        model.eMaxTransport = pyo.Constraint(model.pL, model.pT, rule=eMaxTransport)

    def eMinTransport(mdl, b1, b2, p):

        return mdl.vFlow[b1, b2, p] >= -mdl.pLL[b1, b2]

    if not useFlowBounds:
        model.eMinTransport = pyo.Constraint(model.pL, model.pT, rule=eMinTransport)

    # # constraints to define positive power flow
    # def ePosFlow(mdl, b1, b2, p):
    #     return mdl.vFlow[b1, b2, p] <= mdl.vPosFlow[b1, b2, p]
    #
    # model.ePosFlow = pyo.Constraint(model.pL, model.pT, rule=ePosFlow)
    #
    # def ePosFlow2(mdl, b1, b2, p):
    #     return - mdl.vFlow[b1, b2, p] <= mdl.vPosFlow[b1, b2, p]
    # model.ePosFlow2 = pyo.Constraint(model.pL, model.pT, rule=ePosFlow2)

    def eMaxAngleDiff(mdl, b, p):

        if b == pyo.value(mdl.pSlackBus):
            return mdl.vTheta[b, p] == 0
            #return pyo.Constraint.Skip

        else:
            return mdl.vTheta[b, p] <= mdl.pMAD #+ 1

    if not useAngleBounds:
        model.eMaxAngleDiff = pyo.Constraint(model.pB, model.pT, rule=eMaxAngleDiff)

    def eMinAngleDiff(mdl, b, p):

        if b == pyo.value(mdl.pSlackBus):
            return mdl.vTheta[b, p] == 0
            # return pyo.Constraint.Skip
        else:
            return mdl.vTheta[b, p] >= - mdl.pMAD #+ 1

    if not useAngleBounds:
        model.eMinAngleDiff = pyo.Constraint(model.pB, model.pT, rule=eMinAngleDiff)

    # # test constraints to define positive angles
    # def ePosAngle(mdl, b, p):
    #     return + mdl.vTheta[b, p] <= mdl.vPosTheta[b, p]
    # model.ePosAngle = pyo.Constraint(model.pB, model.pT, rule=ePosAngle)
    #
    # def ePosAngle2(mdl, b, p):
    #     return - mdl.vTheta[b, p] <= mdl.vPosTheta[b, p]
    # model.ePosAngle2 = pyo.Constraint(model.pB, model.pT, rule=ePosAngle2)

    # ------------------- ramping constraints --------------------------------------------------------------------------
    def eRmpUp(mdl, g, p):
        if p == 1:
            return pyo.Constraint.Skip
        else:
            return mdl.vGen[g, p] - mdl.vGen[g, p - 1] <= mdl.pRU[g]

    if ramping:
        model.eRmpUp = pyo.Constraint(model.pGT, model.pT, rule=eRmpUp)

    def eRmpDn(mdl, g, p):
        if p == mdl.PERIODS:
            return pyo.Constraint.Skip
        else:
            return mdl.vGen[g, p] - mdl.vGen[g, p + 1] <= mdl.pRD[g]

    if ramping:
        model.eRmpDn = pyo.Constraint(model.pGT, model.pT, rule=eRmpDn)

    # def eImport(mdl, b, p):
    #     return mdl.vImp[b, p] <= mdl.pIMP[b, p]

    # model.eImport = pyo.Constraint(model.pIMP_NODE, model.pT, rule=eImport)

    model.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return model


def idx_match(row: pd.Series, df: pd.DataFrame, threshold=0):

    idxs = []
    for idx, r in df.iterrows():
        if threshold == 0:
            if np.all(row.values == r.values):
                idxs.append(idx)
        else:
            if np.all((row.values - r.values) < threshold):
                idxs.append(idx)
    return idxs


def extract_duals_binding(model, binding):

    names = []
    indices = []
    duals = []
    for b in binding:
        names.append(b.name)
        indices.append(b.index())
        duals.append(model.dual[b])

    df = pd.DataFrame(data = {'constraint': names, 'index': indices, 'dual': duals})

    return df


def export_aggregation_stats(model, df_vGen, case, original_case, res_folder, original_res_folder, renewables_data,
                             thermals_data):

    # get flow errors
    # get aggregated relative power flows
    df_lineP_perc = export_results.export_line_flow(model, case, folder=res_folder)
    df_lineP_perc = df_lineP_perc.T

    # get original relative power flows
    df_lineP_perc_orig = pd.read_csv(os.path.join(original_res_folder, original_case, 'lineP_perc.csv'), sep=';', decimal=',',
                                     header=[0, 1])
    df_lineP_perc_orig = df_lineP_perc_orig.T

    df_diff_power_flow_perc = df_lineP_perc_orig - df_lineP_perc
    flow_rel_error = df_diff_power_flow_perc.abs().sum() / len(df_diff_power_flow_perc)
    flow_rel_error_max_dev = df_diff_power_flow_perc.abs().max()

    d_flow_error = {'flow_rel_error': flow_rel_error, 'flow_rel_error_max_dev': flow_rel_error_max_dev}

    # get generation errors
    df_vGen_orig = pd.read_csv(os.path.join(original_res_folder, original_case, 'vGen.csv'), sep=';', decimal=',')
    df_vGen_orig.drop(columns={'Unnamed: 0'}, inplace=True)
    df_vGen_orig.set_index('unit', inplace=True)

    df_agg_vGen = df_vGen.set_index('unit')

    df_gen_data = pd.concat([renewables_data, thermals_data], axis=0)
    df_gen_data.set_index('unit', inplace=True)

    # get the difference of the production of agg/orig system
    df_diff_vGen = df_vGen_orig['value'] - df_agg_vGen['value']
    df_diff_vGen_perc = df_diff_vGen / df_gen_data['max']

    # mean of the absolute values of the relative errors
    vGen_rel_error = df_diff_vGen_perc.abs().sum() / len(df_diff_vGen_perc)
    vGen_rel_error_max_dev = df_diff_vGen_perc.abs().max()  # maximum deviation of generation in aggregated and original

    d_vGen_error = {'vGen_rel_error': vGen_rel_error, 'vGen_rel_error_max_dev': vGen_rel_error_max_dev}

    tec_error = pdo.get_production_error_by_technology(df_gen_data, df_diff_vGen)

    df_vDemAdd = export_results.extract_var(model, 'vDemAdd')
    vDemAdd_tot = df_vDemAdd[0].sum()

    d_flow_error['vDemAdd'] = vDemAdd_tot

    return d_flow_error, d_vGen_error, tec_error, []


def run_model(case: str, periods: int, model_type='dcOpf', power_flow_cost=0,
              mult_with_Xline=False, ramping=True, hourly_cost=True, res_folder='results', res_agg_case='',
              activate_NSP=True):

    # define Input case name
    s_aux = case.split('_')
    input_case = case

    #
    # ----------------------------- Load Model Data --------------------------------------------------------------------

    # load data renewables, thermals, capacity factors, demand, network, line data and opf parameters
    sep_dec = ','
    renewables_data = load_renewables(input_case, sep_dec)
    thermals_data = load_thermals(input_case, sep_dec)
    cf_data = load_cf(input_case, sep_dec)
    demand_data = load_demand(input_case, sep_dec)
    line_data = load_lines(input_case, sep_dec)
    opf_data = load_opf_parameters(input_case, sep_dec)
    import_data = load_import(input_case, sep_dec)
    print('Data loaded!')


    # load generation costs and cost for non supplied power
    folder_content = os.listdir(os.path.join('data', input_case))
    if 'gen_costs.csv' in folder_content and 'nsp_costs.csv' in folder_content:
        gen_costs = pd.read_csv(os.path.join('data', input_case, 'gen_costs.csv'), sep=';', decimal=sep_dec,
                                index_col=0)
        gen_costs.index = gen_costs.index.astype('str')
        gen_costs = gen_costs.astype({'CG': 'float'})

        if activate_NSP:
            nsp_costs = pd.read_csv(os.path.join('data', input_case, 'nsp_costs.csv'), sep=';', decimal=sep_dec,
                                    index_col=0)
            nsp_costs.index = nsp_costs.index.astype('str')
            nsp_costs = nsp_costs.astype({'CNSP': 'float'})
        else:
            nsp_costs = pd.DataFrame()

    elif 'gen_costs.xlsx' in folder_content and 'nsp_costs.xlsx' in folder_content:
        gen_costs = pd.read_excel(os.path.join('data', case, 'gen_costs.xlsx'), index_col=0)

        if activate_NSP:
            nsp_costs = pd.read_excel(os.path.join('data', case, 'nsp_costs.xlsx'), index_col=0)
        else:
            nsp_costs = pd.DataFrame()
    else:
        exit('Can\'t load files gen_costs.* or nsp_costs.*!')

    costs = {'CN': 0.1}
    if activate_NSP:
        costs['CNSP'] = list(nsp_costs.to_dict().items())[0][1]
    costs['CG'] = list(gen_costs.to_dict().items())[0][1]

    # import production results from aggregated run
    if res_agg_case != '':
        df_vGen_agg = pd.read_csv(os.path.join(res_folder, res_agg_case, 'vGen.csv'), sep=';', decimal=',')
    else:
        print('df_vGen_agg is None')
        df_vGen_agg = None

    print(df_vGen_agg)


    # if model run for redispatch change results folder so the results won't overwrite the original results

    original_case = ''
    original_res_folder = ''
    if res_agg_case != '':
        original_case = case
        original_res_folder = res_folder
        res_folder = os.path.join(res_folder, 'redispatch')
        case = res_agg_case
        gla.check_if_folder_exists_and_create(res_folder, scenario_folder=case)


    # ----------------------------- Create Model -----------------------------------------------------------------------

    import time
    start = time.time()

    # create model depending on model type (dcOpf)
    if model_type == "dcOpf":
        model = dc_opf_model(renewables_data, thermals_data, opf_data, line_data, cf_data, demand_data, import_data,
                             costs, periods, case,
                             ramping=ramping,
                             hourly_cost=hourly_cost,
                             df_vGen_agg=df_vGen_agg,
                             activate_NSP=activate_NSP,
                             )
    else:
        exit('Choose a valid model type: dcOpf')


    # ------------------------------- Solve Model ----------------------------------------------------------------------
    # hand model to the solver
    end = time.time()
    model_build_time = end - start  # time to build the model
    print('Model created in: {:.2f} seconds'.format(model_build_time))
    solver = pyo.SolverFactory('gurobi_persistent')

    start = time.time()
    solver.set_instance(model)

    res = solver.solve()
    end = time.time()
    model_solving_time = end - start  # time to solve the model
    print('Model solved in: {:.2f} seconds'.format(model_solving_time))
    print('Solver termination condition was: {}'.format(res.solver.termination_condition))
    print('Objective function value was {:.2f}'.format(pyo.value(model.obj)))

    df_model_stats = pd.DataFrame()
    df_model_stats.loc[0, 'model_parameter'] = 'ofv'
    df_model_stats.loc[0, 'parameter_value'] = pyo.value(model.obj)

    # ------------------------------- model results --------------------------------------------------------------------
    res.write()
    model.solutions.load_from(res)

    # check if folder 'results' exits --> create if not
    gla.check_if_folder_exists_and_create('results', case)


    if activate_NSP:
        df_vNSP = export_results.extract_var(model, 'vNSP')
        df_vNSP.columns = ['bus', 'period', 'value']

        if df_vNSP['value'].sum() != 0:
            df_vNSP.to_csv(os.path.join(res_folder, case, 'NSP.csv'), sep=';', decimal=',', index=False)
        else:
            if 'NSP.csv' in os.listdir(os.path.join(res_folder, case)):
                os.remove(os.path.join(res_folder, case, 'NSP.csv'))
                print('No NSP - existing NSP.csv file removed!')
            else:
                print('No NSP!')
    else:
        print('NSP not activated!')
        df_vNSP = None

    # export objective function value
    df_model_stats.to_csv(os.path.join(res_folder, case, 'model_stats.csv'), sep=';', decimal=',', index=True)

    # export power flows
    if model_type == 'dcOpf':
        df_lineP_perc = export_results.export_line_flow(model, case, folder=res_folder)

    # export vGen results
    df_gen = export_results.extract_gen_units(input_case)
    df_vGen = export_results.extract_vGen(model, case, df_gen)
    df_vGen.to_csv(os.path.join(res_folder, case, 'vGen.csv'), sep=';', decimal=',')

    # calculate objective function for basis runs
    obj_value = float(pyo.value(model.obj))
    print('Objective function value was: ' + str(obj_value))

    duals = export_results.extract_duals(model)
    duals.to_csv(os.path.join(res_folder, case, 'duals.csv'), index=False, sep=';', decimal=',')

    df_model_stats.to_csv(os.path.join(res_folder, case, 'model_stats.csv'), sep=';', decimal=',', index=False)

    return {}


if __name__ == '__main__':

    periods = 1
    # case = 'IEEE_24'
    case = 'IEEE_24_p19'

    results_folder = os.path.join('.', 'results')  # chose path outside the repo for big case studies

    gla.check_if_folder_exists_and_create(results_folder, scenario_folder=case)

    results = run_model(case, periods,
                          model_type='dcOpf',
                          ramping=False,
                          hourly_cost=True,
                          res_folder=results_folder,
                          activate_NSP=False,
                        )
