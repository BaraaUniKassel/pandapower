# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


"""Solves the power flow using a full Newton's method.
"""

from numpy import angle, exp, linalg, conj, r_, Inf, arange, zeros, max, zeros_like, column_stack, float64,\
    int64, nan_to_num, flatnonzero, tan, deg2rad, append, array, ones
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix as sparse, vstack, hstack, eye

from pandapower.pf.iwamoto_multiplier import _iwamoto_step
from pandapower.pypower.makeSbus import makeSbus
from pandapower.pf.create_jacobian import create_jacobian_matrix, get_fastest_jacobian_function
from pandapower.pypower.idx_gen import PG
from pandapower.pypower.idx_bus import PD, SL_FAC
from pandapower.pypower.idx_brch import F_BUS, T_BUS, VM_SET_PU, SHIFT


def newtonpf(Ybus, Sbus, V0, ref, pv, pq, ppci, options, makeYbus):
    """Solves the power flow using a full Newton's method.
    Solves for bus voltages given the full system admittance matrix (for
    all buses), the complex bus power injection vector (for all buses),
    the initial vector of complex bus voltages, and column vectors with
    the lists of bus indices for the swing bus, PV buses, and PQ buses,
    respectively. The bus voltage vector contains the set point for
    generator (including ref bus) buses, and the reference angle of the
    swing bus, as well as an initial guess for remaining magnitudes and
    angles.
    @see: L{runpf}
    @author: Ray Zimmerman (PSERC Cornell)
    @author: Richard Lincoln
    Modified by University of Kassel (Florian Schaefer) to use numba
    """

    # options
    tol = options['tolerance_mva']
    max_it = options["max_iteration"]
    numba = options["numba"]
    iwamoto = options["algorithm"] == "iwamoto_nr"
    voltage_depend_loads = options["voltage_depend_loads"]
    dist_slack = options["distributed_slack"]
    v_debug = options["v_debug"]
    use_umfpack = options["use_umfpack"]
    permc_spec = options["permc_spec"]
    trafo_taps = options.get("trafo_taps", False)

    baseMVA = ppci['baseMVA']
    bus = ppci['bus']
    gen = ppci['gen']
    branch = ppci["branch"]
    slack_weights = bus[:, SL_FAC].astype(float64)  ## contribution factors for distributed slack

    # initialize
    i = 0
    V = V0
    Va = angle(V)
    Vm = abs(V)
    dVa, dVm = None, None
    if iwamoto:
        dVm, dVa = zeros_like(Vm), zeros_like(Va)

    if v_debug:
        Vm_it = Vm.copy()
        Va_it = Va.copy()
    else:
        Vm_it = None
        Va_it = None

    # set up indexing for updating V
    if dist_slack and len(ref) > 1:
        pv = r_[ref[1:], pv]
        ref = ref[[0]]

    pvpq = r_[pv, pq]
    # reference buses are always at the top, no matter where they are in the grid (very confusing...)
    # so in the refpvpq, the indices must be adjusted so that ref bus(es) starts with 0
    # todo: is it possible to simplify the indices/lookups and make the code clearer?
    # for columns: columns are in the normal order in Ybus; column numbers for J are reduced by 1 internally
    refpvpq = r_[ref, pvpq]
    # generate lookup pvpq -> index pvpq (used in createJ):
    #   shows for a given row from Ybus, which row in J it becomes
    #   e.g. the first row in J is a PV bus. If the first PV bus in Ybus is in the row 2, the index of the row in Jbus must be 0.
    #   pvpq_lookup will then have a 0 at the index 2
    pvpq_lookup = zeros(max(Ybus.indices) + 1, dtype=int)
    if dist_slack:
        # slack bus is relevant for the function createJ_ds
        pvpq_lookup[refpvpq] = arange(len(refpvpq))
    else:
        pvpq_lookup[pvpq] = arange(len(pvpq))

    # get jacobian function
    createJ = get_fastest_jacobian_function(pvpq, pq, numba, dist_slack)
    
    tap_control_branches = flatnonzero(nan_to_num(branch[:, VM_SET_PU]))
    hv_bus = branch[tap_control_branches, F_BUS].real.astype(int64)
    controlled_bus = branch[tap_control_branches, T_BUS].real.astype(int64)
    # make initial guess for the tap control variables
    vm_set_pu = branch[tap_control_branches, VM_SET_PU].real.astype(float64)
    shift_degree = branch[tap_control_branches, SHIFT].real.astype(float64)
    x_control = r_[Va[controlled_bus], vm_set_pu]
    # x_control = r_[zeros(len(Va[controlled_bus])), ones(len(vm_set_pu))]

    nref = len(ref)
    npv = len(pv)
    npq = len(pq)
    ntap_va = ntap_vm = len(controlled_bus)
    j0 = 0
    j1 = nref if dist_slack else 0
    j2 = j1 + npv  # j1:j2 - V angle of pv buses
    j3 = j2
    j4 = j2 + npq  # j3:j4 - V angle of pq buses
    j5 = j4
    j6 = j4 + npq  # j5:j6 - V mag of pq buses
    j7 = j6 + ntap_va  # trafo tap: modification of va_degree
    j8 = j7 + ntap_vm  # trafo tap: modification of vm_pu

    # make initial guess for the slack
    slack = (gen[:, PG].sum() - bus[:, PD].sum()) / baseMVA

    # evaluate F(x0)
    F = _evaluate_Fx(Ybus, V, Va, Vm, Sbus, ref, pv, pq, slack_weights, dist_slack, slack, trafo_taps, x_control,
                     hv_bus, controlled_bus, vm_set_pu, shift_degree)
    converged = _check_for_convergence(F, tol)

    Ybus = Ybus.tocsr()


    J = None

    # do Newton iterations
    while (not converged and i < max_it):
        # update iteration counter
        i = i + 1
        
        print("V:\t", r_[Va, Vm, x_control])
        print("F:\t", F)

        if trafo_taps:
            Ybus_m = _Ybus_modification(Ybus,tap_control_branches,hv_bus,trafo_taps,controlled_bus)        
            # V = append(V,x_control)
            # pq = r_[pq, len(pq)+len(x_control)]
            # pvpq = r_[pv, pq]
            Ybus_m = Ybus_m.tocsr()
    
        J = create_jacobian_matrix(Ybus, V, ref, refpvpq, pvpq, pq, createJ, pvpq_lookup, nref, npv, npq, numba,
                                   slack_weights, dist_slack, trafo_taps, x_control, Ybus_m, hv_bus, controlled_bus,
                                   Vm,Va, branch,tap_control_branches)

        dx = -1 * spsolve(J, F, permc_spec=permc_spec, use_umfpack=use_umfpack)
        # update voltage
        if dist_slack:
            slack = slack + dx[j0:j1]
        if npv and not iwamoto:
            Va[pv] = Va[pv] + dx[j1:j2]
        if npq and not iwamoto:
            Va[pq] = Va[pq] + dx[j3:j4]
            Vm[pq] = Vm[pq] + dx[j5:j6]
        if trafo_taps:
            x_control[0:ntap_va] += dx[j6:j7]
            x_control[ntap_va:ntap_vm + ntap_va] += dx[j7:j8]
        # iwamoto multiplier to increase convergence
        if iwamoto:
            Vm, Va = _iwamoto_step(Ybus, J, F, dx, pq, npv, npq, dVa, dVm, Vm, Va, pv, j1, j2, j3, j4, j5, j6)
            
        

        V = Vm * exp(1j * Va)
        Vm = abs(V)  # update Vm and Va again in case
        Va = angle(V)  # we wrapped around with a negative Vm

        if v_debug:
            Vm_it = column_stack((Vm_it, Vm))
            Va_it = column_stack((Va_it, Va))

        if voltage_depend_loads:
            Sbus = makeSbus(baseMVA, bus, gen, vm=Vm)

        F = _evaluate_Fx(Ybus, V, Va, Vm, Sbus, ref, pv, pq, slack_weights, dist_slack, slack, trafo_taps, x_control,
                         hv_bus, controlled_bus, vm_set_pu, shift_degree)
        

        converged = _check_for_convergence(F, tol)

    return V, converged, i, J, Vm_it, Va_it, None, None

def _Ybus_modification(Ybus,tap_control_branches,hv_bus,trafo_taps,controlled_bus):
    ##### modify the Ybus to consider the voltage source at regulating Transformer  dfd
    

    YS = Ybus.shape[0]  

    YM_ROW = vstack([Ybus, sparse((int(len(tap_control_branches)),YS))], format="csr")   ### add zero raws
    YM_COL = hstack([YM_ROW, sparse((YS +int(len(tap_control_branches)),int(len(tap_control_branches))))], format="csr")  ### add zero column

    c = 0

    for i,j in zip(hv_bus,controlled_bus):

        YT_d = Ybus[i,i]   ####TODO what if there is multi regulating transformers 
        YT_nd = YT_d * -1 
        
        ##### exchange zeros with the extended Trafo Ybus values

        YM_COL[i,YS + c] = YT_nd
        YM_COL[j,YS + c] = YT_d
        YM_COL[YS + c,i] = YT_d
        YM_COL[YS + c,j] = YT_nd
        YM_COL[YS + c,YS + c] = YT_nd

        c =+ 1

    Ybus_m = YM_COL

    return Ybus_m

def _evaluate_Fx(Ybus, V, Va, Vm, Sbus, ref, pv, pq, slack_weights=None, dist_slack=False, slack=None, trafo_taps=False, x_control=None, hv_bus=None, controlled_bus=None, vm_set_pu=None, shift_degree=None):
    # evalute F(x)
    if dist_slack:
        # we include the slack power (slack * contribution factors) in the mismatch calculation
        mis = V * conj(Ybus * V) - Sbus + slack_weights * slack
        F = r_[mis[ref].real, mis[pv].real, mis[pq].real, mis[pq].imag]
    else:
        mis = V * conj(Ybus * V) - Sbus
        F = r_[mis[pv].real, mis[pq].real, mis[pq].imag]

    if trafo_taps:
        # todo: check if the Va indexing needs to have a lookup
        Va_q = x_control[:len(controlled_bus)]
        #F1 =  tan(Va[hv_bus] - Va[controlled_bus]) - tan(deg2rad(shift_degree))
        # F1 =  tan(Va[hv_bus] - Va[controlled_bus]) - tan(deg2rad(Va_q))
        F1 = tan(Va[controlled_bus] - Va_q) - tan(deg2rad(shift_degree))
        # F1 = tan(Va[controlled_bus] - Va_q) - tan(deg2rad(Va_q))
        F2 = Vm[controlled_bus] - vm_set_pu  # low-volrtage bus of the transformer
        F = r_[F, F1, F2]

    return F


def _check_for_convergence(F, tol):
    # calc infinity norm
    return linalg.norm(F, Inf) < tol
