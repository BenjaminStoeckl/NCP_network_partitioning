#!interpreter [optional-arg]
# -*- coding: utf-8 -*-

__author__ = ["Benjamin Stöckl"]
__copyright__ = "Copyright 2025, Graz University of Technology"
__credits__ = ["Benjamin Stöckl"]
__license__ = "MIT"
__maintainer__ = "Benjamin Stöckl"
__status__ = "Development"


import os.path
import pandas as pd
import numpy as np
import data


def calculate_ptdf(B_line, B_bus_inv):

    return B_line.dot(B_bus_inv)


def calculate_bus_susceptance_matrix_B(l_bus:list, d_line_react:dict):

    '''

    function to calculate the Bus Susceptance Matrix B
    source: Optimization in Modern Power Systems, Spyros Chatzivasileiadis, 2018, p. 13

    :param l_bus: list of buses
    :param d_line_react: dictionary of line reactances

    '''

    B = pd.DataFrame(index=l_bus, columns=l_bus)
    # B = np.zeros(l_bus, l_bus)

    for i in l_bus:
        for j in l_bus:
            if i == j:
                print('test')
                temp = sum([1/d_line_react[(b1, b2)] for b1, b2 in d_line_react.keys() if (b1 == i or b2 == i)])
                print(temp)
                B.loc[i, j] = temp
            else:
                B.loc[i, j] = -sum([1/d_line_react[(b1, b2)] for b1, b2 in d_line_react.keys() if (b1 == i and b2 == j) or (b1 == j and b2 == i)])

    return B


def calculate_bus_reactance_matrix_X(B:pd.DataFrame, sb:str):

    '''

    function to calculate the Bus Reactance Matrix X
    source: Optimization in Modern Power Systems, Spyros Chatzivasileiadis, 2018, p. 20

    :param B: Bus Susceptance Matrix
    :param sb: slack bus as string

    '''

    # remove slack bus entries from B
    B_ = B.drop(sb, axis=0).drop(sb, axis=1).astype(float)

    # invert reduced matrix B_
    B_inv = pd.DataFrame(np.linalg.pinv(B_.values), B_.columns, B_.index)

    # add slack bus entries as zeros
    X = pd.DataFrame(B_inv, index=B.index, columns=B.columns).fillna(0)

    return X


def get_line_susceptance_matrix(l_bus, d_line_react):

    B_line = pd.DataFrame(0, index=d_line_react.keys(), columns=l_bus, dtype=float)

    for i in l_bus:
        for b1, b2 in d_line_react.keys():
            if i == b1:
                B_line.loc[(b1, b2), i] = 1 / d_line_react[(b1, b2)]
            if i == b2:
                B_line.loc[(b1, b2), i] = -1 / d_line_react[(b1, b2)]

    return B_line


def get_generator_mapping_matrix(l_bus, df_gen):

    G = pd.DataFrame(0, index=l_bus, columns=df_gen['unit'].values, dtype=float)

    for i in l_bus:
        for g in df_gen['unit'].values:
            if df_gen.loc[df_gen['unit'] == g, 'bus'].values == i:
                G.loc[i, g] = 1

    return G


if __name__ == "__main__":

    # case = 'IEEE_24_p19'
    case = 'IEEE_24_p19_red_KMedoids_5_tecAss'

    # load grid data
    lines = data.load_lines(case, ',')
    df_opf_parameters = pd.read_csv(os.path.join('data', case, 'opf_parameters.csv'), sep=';', decimal=',')
    slack_bus = str(df_opf_parameters.loc[df_opf_parameters['parameter_name'] == 'SlackBus', 'parameter_value'].values[0])

    l_bus = data.__buses_from_lines__(lines)
    d_line_power, d_line_react = data.__lines_to_dict__(lines)

    B = calculate_bus_susceptance_matrix_B(l_bus, d_line_react)
    print('\nB:\n', B)

    X = calculate_bus_reactance_matrix_X(B, slack_bus)
    print('\nX:\n', X)

    B_line = get_line_susceptance_matrix(l_bus, d_line_react)
    print(B_line)

    PTDF = calculate_ptdf(B_line, X)
    PTDF.index.names = ['bus1', 'bus2']
    print(PTDF)

    PTDF.to_csv(os.path.join('data', case, 'PTDF_susc.csv'), sep=';', decimal=',')


    print('\nfinished!')

