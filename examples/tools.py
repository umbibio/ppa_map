"""Helper methods for PPA mapping symbolic computations"""

import sympy as sp
from sympy.polys.polytools import terms_gcd
# from sympy import *
import time
from IPython.display import display, Math
import numpy as np


def distribute(e, x):
    """Helper method to apply distributive multiplication.
    This helps by allowing the use of 'subs' instead of 'limit'
    in special circumstances
    """

    # traverse the operations graph
    if e.is_Add:
        return sp.Add(*[distribute(a, x) for a in e.args])
    elif e.is_Mul:
        o = sp.Integer(1)
        done = False
        for a in e.args:
            if x in a.atoms() and not done:
                # prefer multiply where x is found
                # seek for simplification
                o = sp.Mul(o, distribute(a, x))
                done = True
            else:
                o = sp.Mul(o, a)
        return o if done else sp.Mul(o, x)
    else:
        return sp.Mul(e, x)


def show_expr(expr, plain_latex=False):
    """Helper function to display Math expressions"""
    if plain_latex:
        print(sp.printing.latex(expr))
        return None#sp.printing.latex(expr)
    return display(Math(sp.printing.latex(expr)))


def compositions(n, k):
    """
    Returns all compositions of size 'k' for number 'n'.
    We use it to discover all possible states in a given
    reduced model with k states, and n mRNAs in it. 
    Very useful for bursty mRNA arrivals.
    """

    if n < 0 or k < 0:
        return
    elif k == 0:
        # the empty sum, by convention, is zero, so only return something if
        # n is zero
        if n == 0:
            yield []
        return
    elif k == 1:
        yield [n]
        return
    else:
        for i in range(0,n+1):
            for comp in compositions(n-i,k-1):
                yield [i] + comp


def clean_from_N(M):
    """
    Simplifies sympy matrices or vectors by taking
    limit N->inf when possible, without losing first order
    """

    N = sp.symbols('N')
    M = M.as_mutable()
    for i, elem in enumerate(M):
        if N in M[i].atoms():
            # first try taking limit directly
            tmp = (M[i]).subs(N, sp.S.Infinity)
            if tmp != 0:
                M[i] = tmp
                continue

            # now try multiplying by N before taking the limit
            nmi = distribute(M[i], N)

            tmp = (nmi).subs(nmi, sp.S.Infinity)/N
            if not (-sp.S.Infinity in tmp.atoms() or sp.S.Infinity in tmp.atoms()):
                M[i] = tmp
    return M


def compute_steady_state_moments(model_transition_rates, model_protein_production_rates, model_protein_degradation_rate):

    g = model_protein_degradation_rate
    N = sp.symbols('N')

    model_transition_rates_by_deltas = {}
    tran_rates = {}

    Nmrna_stat_red = len(model_protein_production_rates)

    protein_mean = {}
    protein_fano = {}

    for burst_size in [1, 2]:
        print("\n")
        # The size of the mRNA burst
        bs = burst_size
        print(f"\nWorking on Burst Size = {bs}\n")

        tran_rates[bs] = {}

        # These are all of the possible states of the reduced model
        states = np.array(list(compositions(burst_size, Nmrna_stat_red)))

        # Build a dictionary of transitions
        for i in range(len(states)):
            si = tuple(states[i])
            for j in range(len(states)):
                delta = states[j] - states[i]
                sj = tuple(states[j])
                sd = tuple(delta)

                # the index for the source state for this transition
                src_idx = np.argmin(delta)

                try:
                    if bs == 1:
                        rate = model_transition_rates[(si, sj)]
                        model_transition_rates_by_deltas[sd] = rate
                    else:
                        if delta[0] == - burst_size:
                            sd = tuple((delta/burst_size).astype(int))
                            rate = model_transition_rates_by_deltas[sd]
                        else:
                            # total rate is proportional to the number of mRNAs in source state
                            rate = model_transition_rates_by_deltas[sd] * states[i, src_idx]
                except KeyError:
                    continue

                msg = f"{states[i]} --> {states[j]}\tdelta = {delta}\t{rate}"
                if max(abs(delta)) == 1 and delta[0] >= 0:
                    print(msg)
                elif delta[0] == - burst_size:
                    print(msg + "\t(arrival)")
                else:
                    continue

                tran_rates[bs][(si, sj)] = rate

        # will contain the actual protein production rates for all posible
        # states of the system
        prod_rates = {}

        # the set of posible states of the system
        states = set([node for edge in tran_rates[bs].keys() for node in edge])

        # protein production rates at each state
        for state in states:
            # initialize
            prod_rates[state] = 0
            for i, n in enumerate(state):
                # add up all protein production rate contributions
                # n is the number of mRNAs in i-th mRNA state
                prod_rates[state] += n * model_protein_production_rates[i]

        # Node count, sorted list of nodes
        n_cnt = len(states)
        n_lst = reversed(sorted(list(states)))
        n_idx = range(n_cnt)

        # dict map from node name to node index in sorted list
        n_dct = dict(zip(n_lst, n_idx))

        # Build the "transition" matrix
        K = sp.zeros(n_cnt, n_cnt)
        for t, rt in tran_rates[bs].items():
            K[n_dct[t[1]], n_dct[t[0]]] += rt
            K[n_dct[t[0]], n_dct[t[0]]] -= rt

        print(f"\n\n\nThe transition Matrix K\n")
        show_expr(K)

        R = sp.zeros(n_cnt, n_cnt)
        for s, rt in prod_rates.items():
            R[n_dct[s], n_dct[s]] += rt

        X = K.copy()
        X.row_del(0)
        X = X.row_insert(0, sp.ones(1, K.shape[0]))
        G = sp.eye(K.shape[0])*g
        kr = R*sp.ones(K.shape[0], 1)
        b = sp.zeros(K.shape[0], 1)
        b[0] = 1

        print(f"\n\n\nThe protein production rates\n")
        show_expr(kr.T)

        L, U, P = X.LUdecomposition()
        L = clean_from_N(L)
        U = clean_from_N(U)
        b = L.inv()*b
        m0 = U.LUsolve(b)
        m0 = clean_from_N(m0)

        L, U, P = (K-G).LUdecomposition()
        L = clean_from_N(L)
        U = clean_from_N(U)
        b = L.inv()*R*m0
        b = clean_from_N(b)
        m1 = -U.LUsolve(b)
        m1 = clean_from_N(m1)


        # 1st moment at reduced model
        mean_rm = ((kr.T * m0/g)[0])

        # 1st moment: E[p], the mean
        # for protein number at original model

        mean = sp.factor((distribute(mean_rm, N)).subs(N, sp.S.Infinity))

        print(f"\n\n\nThe protein mean\n")
        show_expr(mean)

        # 2nd moment at reduced model
        secm_rm = ((kr.T * m1/g)[0])
        # secm_rm = sp.factor((kr.T * m1/g)[0])
        # show_expr(secm_rm)

        # 2nd moment: E[p*(p-1)]
        # for protein number at original model
        secm = ((distribute(secm_rm, N)).subs(N, sp.S.Infinity)) + mean**2
        # show_expr(secm)

        # compute the variance
        # for protein number at original model
        variance = (sp.factor(secm/mean - mean) + 1) * mean
        # show_expr(variance)

        # Standard deviation
        stdv = sp.sqrt(variance)
        # show_expr(stdv)

        # compute the fano factor
        # for protein number at original model
        fano = sp.factor(variance/mean -1) +1 # tweak factorization

        print(f"\n\n\nThe protein fano factor\n")
        show_expr(fano)

        protein_mean[bs] = mean
        protein_fano[bs] = fano

    FF2, FF1, FF, Gb2, Gb1, ue, up, kp = sp.symbols("FF_2 FF_1 FF G''_b G'_b mu_e mu_p k_p")

    print(f"\n\n\nThe protein fano factor for burst size 1\n")
    show_expr(sp.Eq(FF1, protein_fano[1]))

    print(f"\n\n\nThe protein fano factor for burst size 2\n")
    show_expr(sp.Eq(FF2, sp.factor(protein_fano[2] - protein_fano[1]) + protein_fano[1]))

    print(f"\n\n\nThe protein fano factor for bursty mRNA arrivals")
    print(f"with Gb the generating function for the burst distribution\n")
    show_expr(sp.Eq(FF, (1 + (protein_fano[1]-1)*(1 + sp.factor(protein_fano[2]-protein_fano[1])/(protein_fano[1]-1)*(Gb2/Gb1)))))

    return protein_mean, protein_fano


def compute_matrices(model_transition_rates, model_protein_production_rates, model_protein_degradation_rate, max_burst_size=1):

    g = model_protein_degradation_rate
    N = sp.symbols('N')

    model_transition_rates_by_deltas = {}
    tran_rates = {}

    Nmrna_stat_red = len(model_protein_production_rates)

    for burst_size in range(1, max_burst_size + 1):

        # The size of the mRNA burst
        bs = burst_size
        print(f"\nWorking on Burst Size = {bs}\n")

        tran_rates[bs] = {}

        # These are all of the possible states of the reduced model
        states = np.array(list(compositions(burst_size, Nmrna_stat_red)))

        # Build a dictionary of transitions
        for i in range(len(states)):
            si = tuple(states[i])
            for j in range(len(states)):
                delta = states[j] - states[i]
                sj = tuple(states[j])
                sd = tuple(delta)

                # the index for the source state for this transition
                src_idx = np.argmin(delta)
                # the index for the target state for this transition
                trg_idx = np.argmax(delta)

                try:
                    if bs == 1:
                        rate = model_transition_rates[(si, sj)]
                        model_transition_rates_by_deltas[sd] = rate
                    else:
                        if delta[0] == - burst_size:
                            sd = tuple((delta/burst_size).astype(int))
                            rate = model_transition_rates_by_deltas[sd]
                        else:
                            # total rate is proportional to the number of mRNAs in source state
                            rate = model_transition_rates_by_deltas[sd] * states[i, src_idx]
                except KeyError:
                    continue

                msg = f"{states[i]} --> {states[j]}\tdelta = {delta}\t{rate}"
                if max(abs(delta)) == 1 and delta[0] >= 0:
                    print(msg)
                elif delta[0] == - burst_size:
                    arrival_state = sj
                    print(msg + "\t(arrival)")
                else:
                    continue

                tran_rates[bs][(si, sj)] = rate

        # will contain the actual protein production rates for all posible
        # states of the system
        prod_rates = {}

        # the set of posible states of the system
        states = set([node for edge in tran_rates[bs].keys() for node in edge])

        # protein production rates at each state
        for state in states:
            # initialize
            prod_rates[state] = 0
            for i, n in enumerate(state):
                # add up all protein production rate contributions
                # n is the number of mRNAs in i-th mRNA state
                prod_rates[state] += n * model_protein_production_rates[i]

    # Node count, sorted list of nodes
    n_cnt = len(states)
    n_lst = reversed(sorted(list(states)))
    n_idx = range(n_cnt)

    # dict map from node name to node index in sorted list
    n_dct = dict(zip(n_lst, n_idx))

    # Build the "transition" matrix
    K = sp.zeros(n_cnt, n_cnt)
    for t, rt in tran_rates[bs].items():
        K[n_dct[t[1]], n_dct[t[0]]] += rt
        K[n_dct[t[0]], n_dct[t[0]]] -= rt

    print(f"\n\n\nThe transition Matrix K\n")
    show_expr(K)

    R = sp.zeros(n_cnt, n_cnt)
    for s, rt in prod_rates.items():
        R[n_dct[s], n_dct[s]] += rt

    G = sp.eye(K.shape[0])*g
    kr = R*sp.ones(K.shape[0], 1)

    return K, G, R, kr, n_dct[arrival_state]-1
