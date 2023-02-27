# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


"""Solves the power flow using a full Newton's method.
"""
import numpy as np
from numpy import float64, array, angle, sqrt, square, exp, linalg, conj, r_, Inf, arange, zeros, max, \
    zeros_like, column_stack, flatnonzero, nan_to_num
from scipy.sparse import csr_matrix, eye, vstack
from scipy.sparse.linalg import spsolve

from pandapower.pf.iwamoto_multiplier import _iwamoto_step
from pandapower.pypower.makeSbus import makeSbus
from pandapower.pf.create_jacobian import create_jacobian_matrix, get_fastest_jacobian_function
from pandapower.pypower.idx_gen import PG
from pandapower.pypower.idx_bus import PD, SL_FAC, BASE_KV, SVC, SET_VM_PU, SVC_THYRISTOR_FIRING_ANGLE, BS, SVC_X_L, SVC_X_CVAR
from pandapower.pypower.idx_brch import BR_R, BR_X, F_BUS, T_BUS
from pandapower.pypower.idx_brch_tdpf import BR_R_REF_OHM_PER_KM, BR_LENGTH_KM, RATE_I_KA, T_START_C, R_THETA, \
    WIND_SPEED_MPS, ALPHA, TDPF, OUTER_DIAMETER_M, MC_JOULE_PER_M_K, WIND_ANGLE_DEGREE, SOLAR_RADIATION_W_PER_SQ_M, \
    GAMMA, EPSILON, T_AMBIENT_C, T_REF_C
from pandapower.pypower.idx_tcsc import F_BUS_TCSC, T_BUS_TCSC, TCSC_X_L, TCSC_X_CVAR, TCSC_SET_P, \
    TCSC_THYRISTOR_FIRING_ANGLE, TCSC_STATUS, TCSC_CONTROLLABLE, tcsc_cols, TCSC_MIN_FIRING_ANGLE, \
    TCSC_MAX_FIRING_ANGLE, TCSC_PF, TCSC_QF, TCSC_PT, TCSC_QT, TCSC_IF, TCSC_IT, TCSC_X_PU

from pandapower.pf.create_jacobian_tdpf import calc_g_b, calc_a0_a1_a2_tau, calc_r_theta, \
    calc_T_frank, calc_i_square_p_loss, create_J_tdpf

from pandapower.pf.create_jacobian_facts import create_J_modification_svc, calc_y_svc_pu, create_J_modification_tcsc, \
    calc_tcsc_p_pu


def newtonpf(Ybus, Sbus, V0, ref, pv, pq, ppci, options, makeYbus=None):
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

    baseMVA = ppci['baseMVA']
    bus = ppci['bus']
    gen = ppci['gen']
    branch = ppci['branch']
    tcsc = ppci['tcsc']
    slack_weights = bus[:, SL_FAC].astype(float64)  ## contribution factors for distributed slack
    tdpf = options.get('tdpf', False)

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

    pq_lookup = zeros(max(refpvpq) + 1, dtype=int)
    pq_lookup[pq] = arange(len(pq))

    # get jacobian function
    createJ = get_fastest_jacobian_function(pvpq, pq, numba, dist_slack)

    svc_buses = flatnonzero(nan_to_num(bus[:, SVC]))
    svc_set_vm_pu = bus[svc_buses, SET_VM_PU]
    x_control_svc = bus[svc_buses, SVC_THYRISTOR_FIRING_ANGLE]
    svc_controllable = np.ones_like(x_control_svc, dtype=bool)
    svc_x_l_pu = bus[svc_buses, SVC_X_L]
    svc_x_cvar_pu = bus[svc_buses, SVC_X_CVAR]
    Sbus_backup = Sbus.copy()  # todo: make work without this

    tcsc_branches = flatnonzero(nan_to_num(tcsc[:, TCSC_STATUS]))
    tcsc_fb = tcsc[tcsc_branches, [F_BUS_TCSC]].real.astype(int)
    tcsc_tb = tcsc[tcsc_branches, [T_BUS_TCSC]].real.astype(int)

    tcsc_controllable = tcsc[tcsc_branches, TCSC_CONTROLLABLE].real.astype(bool)

    tcsc_set_p_pu = tcsc[tcsc_branches[tcsc_controllable], TCSC_SET_P].real

    tcsc_min_x = tcsc[tcsc_branches[tcsc_controllable], TCSC_MIN_FIRING_ANGLE].real
    tcsc_max_x = tcsc[tcsc_branches[tcsc_controllable], TCSC_MAX_FIRING_ANGLE].real

    # todo differentiate controllable or not
    x_control_tcsc = tcsc[tcsc_branches, TCSC_THYRISTOR_FIRING_ANGLE].real

    tcsc_x_l_pu = tcsc[tcsc_branches, TCSC_X_L].real
    tcsc_x_cvar_pu = tcsc[tcsc_branches, TCSC_X_CVAR].real
    num_svc_controllable = len(x_control_svc[svc_controllable])
    num_tcsc_controllable = len(x_control_tcsc[tcsc_controllable])
    num_facts_controllable = num_svc_controllable + num_tcsc_controllable
    num_facts_total = len(x_control_svc) + len(x_control_tcsc)

    tcsc_in_pq_f = np.isin(branch[tcsc_branches, F_BUS].real.astype(int), pq)
    tcsc_in_pq_t = np.isin(branch[tcsc_branches, T_BUS].real.astype(int), pq)
    tcsc_in_pvpq_f = np.isin(branch[tcsc_branches, F_BUS].real.astype(int), pvpq)
    tcsc_in_pvpq_t = np.isin(branch[tcsc_branches, T_BUS].real.astype(int), pvpq)
    #else:
     #   tcsc_fb = tcsc_tb = tcsc_i = tcsc_j = None

    nref = len(ref)
    npv = len(pv)
    npq = len(pq)
    j0 = 0
    j1 = nref if dist_slack else 0
    j2 = j1 + npv  # j1:j2 - V angle of pv buses
    j3 = j2
    j4 = j2 + npq  # j3:j4 - V angle of pq buses
    j5 = j4
    j6 = j4 + npq  # j5:j6 - V mag of pq buses
    j6a = j6 + num_svc_controllable  # svc
    j6b = j6a + num_tcsc_controllable  # svc
    j7 = j6b

    # make initial guess for the slack
    slack = (gen[:, PG].sum() - bus[:, PD].sum()) / baseMVA
    # evaluate F(x0)

    Ybus_tcsc = makeYbus_tcsc(Ybus, x_control_tcsc, tcsc_x_l_pu, tcsc_x_cvar_pu, tcsc_fb, tcsc_tb)
    F = _evaluate_Fx(Ybus, V, Sbus, ref, pv, pq, slack_weights, dist_slack, slack, svc_buses, svc_set_vm_pu,
                     tcsc_controllable, tcsc_set_p_pu, tcsc_tb, Ybus_tcsc)

    T_base = 100  # T in p.u. for better convergence
    T = 20 / T_base
    r_theta_pu = 0
    if tdpf:
        if len(pq) > 0:
            pq_lookup = zeros(max(refpvpq) + 1, dtype=int)  # for TDPF
            pq_lookup[pq] = arange(len(pq))
        else:
            pq_lookup = array([])
        tdpf_update_r_theta = options.get('tdpf_update_r_theta', True)
        tdpf_delay_s = options.get('tdpf_delay_s')
        tdpf_lines = flatnonzero(nan_to_num(branch[:, TDPF]))
        # set up the necessary parameters for TDPF:
        T0 = branch[tdpf_lines, T_START_C].real / T_base
        t_ref_pu = branch[tdpf_lines, T_REF_C].real / T_base
        t_air_pu = branch[tdpf_lines, T_AMBIENT_C].real / T_base
        alpha_pu = branch[tdpf_lines, ALPHA].real * T_base

        i_max_a = branch[tdpf_lines, RATE_I_KA].real * 1e3
        v_base_kv = bus[branch[tdpf_lines, F_BUS].real.astype(int), BASE_KV]
        z_base_ohm = square(v_base_kv) / baseMVA
        r_ref_pu = branch[tdpf_lines, BR_R_REF_OHM_PER_KM].real * branch[tdpf_lines, BR_LENGTH_KM].real / z_base_ohm
        i_base_a = baseMVA / (v_base_kv * sqrt(3)) * 1e3
        i_max_pu = i_max_a / i_base_a
        # p_rated_loss_pu = square(i_max_pu) * r_ref_pu * (1 + alpha_pu * (25/T_base+t_air_pu - t_ref_pu))
        # p_rated_loss_mw = square(branch[tdpf_lines, RATE_I_KA].real * sqrt(3)) * branch[tdpf_lines, BR_R_REF_OHM_PER_KM].real * branch[tdpf_lines, BR_LENGTH_KM].real * (1 + alpha_pu * (25/T_base+t_air_pu - t_ref_pu))
        # assert np.allclose(p_rated_loss_mw / baseMVA, p_rated_loss_pu)
        # defined in Frank et.al. as T_Rated_Rise / p_rated_loss. Given in net.line based on °C, kA, kV:
        r_theta_pu = branch[tdpf_lines, R_THETA].real * baseMVA / T_base
        x = branch[tdpf_lines, BR_X].real

        # calculate parameters for J:
        Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
        # todo: add parameters to the create function
        a0, a1, a2, tau = calc_a0_a1_a2_tau(t_air_pu=t_air_pu, t_max_pu=80 / T_base, t_ref_pu=t_ref_pu,
                                            r_ref_ohm_per_m=1e-3 * branch[tdpf_lines, BR_R_REF_OHM_PER_KM].real,
                                            conductor_outer_diameter_m=branch[tdpf_lines, OUTER_DIAMETER_M].real,
                                            mc_joule_per_m_k=branch[tdpf_lines, MC_JOULE_PER_M_K].real,
                                            wind_speed_m_per_s=branch[tdpf_lines, WIND_SPEED_MPS].real,
                                            wind_angle_degree=branch[tdpf_lines, WIND_ANGLE_DEGREE].real,
                                            s_w_per_square_meter=branch[tdpf_lines, SOLAR_RADIATION_W_PER_SQ_M].real,
                                            alpha_pu=alpha_pu, solar_absorptivity=branch[tdpf_lines, GAMMA].real,
                                            emissivity=branch[tdpf_lines, EPSILON].real, T_base=T_base,
                                            i_base_a=i_base_a)
        g, b = calc_g_b(r_ref_pu, x)
        i_square_pu, p_loss_pu = calc_i_square_p_loss(branch, tdpf_lines, g, b, Vm, Va)
        if tdpf_update_r_theta:
            r_theta_pu = calc_r_theta(t_air_pu, a0, a1, a2, i_square_pu, p_loss_pu)
        # initial guess for T:
        # T = calc_T_frank(p_loss_pu, t_air_pu, r_theta_pu, tdpf_delay_s, T0, tau)
        T = T0.copy()  # better for e.g. timeseries calculation
        F_t = zeros(len(branch))
        # F_t[tdpf_lines] = T - T0
        F = r_[F, F_t]

    converged = _check_for_convergence(F, tol)

    Ybus = Ybus.tocsr()
    J = None


    # do Newton iterations
    while (not converged and i < max_it):
        # update iteration counter
        i = i + 1

        if tdpf:
            # update the R, g, b for the tdpf_lines, and the Y-matrices
            branch[tdpf_lines, BR_R] = r = r_ref_pu * (1 + alpha_pu * (T - t_ref_pu))
            Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
            g, b = calc_g_b(r, x)

        # here: if J is "Jacobian for the original system", then it should be based on Ybus
        J = create_jacobian_matrix(Ybus, V, ref, refpvpq, pvpq, pq, createJ, pvpq_lookup, nref, npv, npq, numba, slack_weights, dist_slack)
        # J = create_jacobian_matrix(Ybus+Ybus_tcsc, V, ref, refpvpq, pvpq, pq, createJ, pvpq_lookup, nref, npv, npq, numba, slack_weights, dist_slack)

        if tdpf:
            # p.u. values for T, a1, a2, I, S
            # todo: distributed_slack works fine if sn_mva is rather high (e.g. 1000), otherwise no convergence. Why?
            J = create_J_tdpf(branch, tdpf_lines, alpha_pu, r_ref_pu, refpvpq if dist_slack else pvpq, pq, pvpq_lookup,
                              pq_lookup, tau, tdpf_delay_s, Vm, Va, r_theta_pu, J, r, x, g)

        if num_facts_controllable > 0:
            K_J = vstack([eye(J.shape[0], format="csr"),
                          csr_matrix((num_facts_controllable, J.shape[0]))], format="csr")
            J = K_J * J * K_J.T  # this extends the J matrix with 0-rows and 0-columns
        if len(svc_buses):
            # todo: fix this
            J_m_svc = create_J_modification_svc(J, svc_buses, pvpq, pq, pq_lookup, V, x_control_svc, svc_controllable,
                                                svc_x_l_pu, svc_x_cvar_pu)
            J = J + J_m_svc
        if len(tcsc_branches) > 0:
            J_m_tcsc = create_J_modification_tcsc(V, Ybus_tcsc, x_control_tcsc, tcsc_controllable,
                                                  tcsc_x_l_pu, tcsc_x_cvar_pu, tcsc_fb, tcsc_tb,
                                                  pvpq, pq, pvpq_lookup, pq_lookup)
            J = J + J_m_tcsc

        dx = -1 * spsolve(J, F, permc_spec=permc_spec, use_umfpack=use_umfpack)
        # update voltage
        if dist_slack:
            slack = slack + dx[j0:j1]
        if npv and not iwamoto:
            Va[pv] = Va[pv] + dx[j1:j2]
        if npq and not iwamoto:
            Va[pq] = Va[pq] + dx[j3:j4]
            Vm[pq] = Vm[pq] + dx[j5:j6]
        if num_svc_controllable > 0:
            x_control_svc[svc_controllable] += dx[j6:j6a]
        if num_tcsc_controllable > 0:
            x_control_tcsc[tcsc_controllable] += dx[j6a:j6b]
        if tdpf:
            T = T + dx[j7:][tdpf_lines]

        # iwamoto multiplier to increase convergence
        if iwamoto and not tdpf:
            Vm, Va = _iwamoto_step(Ybus, J, F, dx, pq, npv, npq, dVa, dVm, Vm, Va, pv, j1, j2, j3, j4, j5, j6)

        V = Vm * exp(1j * Va)
        Vm = abs(V)  # update Vm and Va again in case
        Va = angle(V)  # we wrapped around with a negative Vm

        if v_debug:
            Vm_it = column_stack((Vm_it, Vm))
            Va_it = column_stack((Va_it, Va))

        if len(svc_buses) > 0:
            y_svc = calc_y_svc_pu(x_control_svc, svc_x_l_pu, svc_x_cvar_pu)
            q_svc = square(abs(V[svc_buses])) * y_svc
            Sbus[svc_buses] = Sbus_backup[svc_buses] - q_svc * 1j

        if voltage_depend_loads:
            Sbus = makeSbus(baseMVA, bus, gen, vm=Vm)

        if num_tcsc_controllable > 0:
            Ybus_tcsc = makeYbus_tcsc(Ybus, x_control_tcsc, tcsc_x_l_pu, tcsc_x_cvar_pu, tcsc_fb, tcsc_tb)
        F = _evaluate_Fx(Ybus, V, Sbus, ref, pv, pq, slack_weights, dist_slack, slack, svc_buses, svc_set_vm_pu,
                         tcsc_controllable, tcsc_set_p_pu, tcsc_tb, Ybus_tcsc)

        if tdpf:
            i_square_pu, p_loss_pu = calc_i_square_p_loss(branch, tdpf_lines, g, b, Vm, Va)
            if tdpf_update_r_theta:
                r_theta_pu = calc_r_theta(t_air_pu, a0, a1, a2, i_square_pu, p_loss_pu)
            T_calc = calc_T_frank(p_loss_pu, t_air_pu, r_theta_pu, tdpf_delay_s, T0, tau)
            F_t[tdpf_lines] = T - T_calc
            F = r_[F, F_t]

        converged = _check_for_convergence(F, tol)

    # write q_svc, x_control in ppc["bus"] and then later calculate q_mvar for net.res_shunt
    if len(svc_buses) > 0:
        bus[svc_buses, BS] += -q_svc * baseMVA * (1 / svc_set_vm_pu) ** 2
        bus[svc_buses, SVC_THYRISTOR_FIRING_ANGLE] = x_control_svc

    if len(tcsc_branches) > 0:
        # todo: move to pf.run_newton_raphson_pf.ppci_to_pfsoln
        baseI = baseMVA / (bus[tcsc_tb, BASE_KV] * sqrt(3))
        Ibus_tcsc = Ybus_tcsc * V
        Sbus_tcsc = V * conj(Ibus_tcsc) * baseMVA
        tcsc[tcsc_branches, TCSC_THYRISTOR_FIRING_ANGLE] = x_control_tcsc
        tcsc[tcsc_branches, TCSC_PF] = Sbus_tcsc[tcsc_fb].real
        tcsc[tcsc_branches, TCSC_QF] = Sbus_tcsc[tcsc_fb].imag
        tcsc[tcsc_branches, TCSC_PT] = Sbus_tcsc[tcsc_tb].real
        tcsc[tcsc_branches, TCSC_QT] = Sbus_tcsc[tcsc_tb].imag
        tcsc[tcsc_branches, TCSC_IF] = np.abs(Ibus_tcsc[tcsc_fb]) * baseI
        tcsc[tcsc_branches, TCSC_IT] = np.abs(Ibus_tcsc[tcsc_tb]) * baseI
        tcsc[tcsc_branches, TCSC_X_PU] = 1 / calc_y_svc_pu(x_control_tcsc, tcsc_x_l_pu, tcsc_x_cvar_pu)

    return V, converged, i, J, Vm_it, Va_it, r_theta_pu / baseMVA * T_base, T * T_base


def _evaluate_Fx(Ybus, V, Sbus, ref, pv, pq, slack_weights=None, dist_slack=False, slack=None,
                 svc_buses=None, svc_set_vm_pu=None, tcsc_controllable=None, tcsc_set_p_pu=None, tcsc_tb=None, Ybus_tcsc=None):
    # evalute F(x)
    if dist_slack:
        # we include the slack power (slack * contribution factors) in the mismatch calculation
        mis = V * conj(Ybus * V) - Sbus + slack_weights * slack
        F = r_[mis[ref].real, mis[pv].real, mis[pq].real, mis[pq].imag]
    elif tcsc_tb is not None and len(tcsc_tb) > 0:
        # p_tcsc, *_ = calc_tcsc_p_pu(Ybus, V, tcsc_fb, tcsc_tb)
        mis = V * conj((Ybus + Ybus_tcsc) * V) - Sbus
        F = r_[mis[pv].real, mis[pq].real, mis[pq].imag]

        if np.any(tcsc_controllable):
            Sbus_tcsc = V * conj(Ybus_tcsc * V)
            mis_tcsc = Sbus_tcsc[tcsc_tb[tcsc_controllable]].real - tcsc_set_p_pu
            F = r_[F, mis_tcsc]
    else:
        mis = V * conj(Ybus * V) - Sbus
        F = r_[mis[pv].real, mis[pq].real, mis[pq].imag]
    if svc_buses is not None and len(svc_buses) > 0:
        Fc_svc = abs(V[svc_buses]) - svc_set_vm_pu
        F = r_[F, Fc_svc]
    return F


def _check_for_convergence(F, tol):
    # calc infinity norm
    return linalg.norm(F, Inf) < tol


def makeYbus_tcsc(Ybus, x_control_tcsc, tcsc_x_l_pu, tcsc_x_cvar_pu, tcsc_fb, tcsc_tb):
    Ybus_tcsc = np.zeros(Ybus.shape, dtype=np.complex128)
    Y_TCSC = calc_y_svc_pu(x_control_tcsc, tcsc_x_l_pu, tcsc_x_cvar_pu)
    Y_TCSC_c = -1j * Y_TCSC
    for y_tcsc_pu_i, i, j in zip(Y_TCSC_c, tcsc_fb, tcsc_tb):
        Ybus_tcsc[i, i] += y_tcsc_pu_i
        Ybus_tcsc[i, j] += -y_tcsc_pu_i
        Ybus_tcsc[j, i] += -y_tcsc_pu_i
        Ybus_tcsc[j, j] += y_tcsc_pu_i
    return csr_matrix(Ybus_tcsc)

