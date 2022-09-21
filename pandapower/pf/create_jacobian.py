from numpy import complex128, float64, int32, r_, int64
from numpy.core.multiarray import zeros, empty, array
import numpy as np
from scipy.sparse import csr_matrix as sparse, vstack, hstack, eye

from pandapower.pypower.dSbus_dV import dSbus_dV
from pandapower.pypower.idx_brch import F_BUS, T_BUS

try:
    # numba functions
    from pandapower.pf.create_jacobian_numba import create_J, create_J2, create_J_ds
    from pandapower.pf.dSbus_dV_numba import dSbus_dV_numba_sparse
except ImportError:
    pass


def _create_J_with_numba(Ybus, V, refpvpq, pvpq, pq, createJ, pvpq_lookup, nref, npv, npq, slack_weights, dist_slack):
    Ibus = zeros(len(V), dtype=complex128)
    # create Jacobian from fast calc of dS_dV
    dVm_x, dVa_x = dSbus_dV_numba_sparse(Ybus.data, Ybus.indptr, Ybus.indices, V, V / abs(V), Ibus)
    # data in J, space preallocated is bigger than acutal Jx -> will be reduced later on
    Jx = empty(len(dVm_x) * 4, dtype=float64)
    # row pointer, dimension = pvpq.shape[0] + pq.shape[0] + 1
    if dist_slack:
        Jp = zeros(refpvpq.shape[0] + pq.shape[0] + 1, dtype=int32)
    else:
        Jp = zeros(pvpq.shape[0] + pq.shape[0] + 1, dtype=int32)
    # indices, same with the preallocated space (see Jx)
    Jj = empty(len(dVm_x) * 4, dtype=int32)

    # fill Jx, Jj and Jp
    createJ(dVm_x, dVa_x, Ybus.indptr, Ybus.indices, pvpq_lookup, refpvpq, pvpq, pq, Jx, Jj, Jp, slack_weights)

    # resize before generating the scipy sparse matrix
    Jx.resize(Jp[-1], refcheck=False)
    Jj.resize(Jp[-1], refcheck=False)

    # todo: why not replace npv by pv.shape[0] etc.?
    # generate scipy sparse matrix
    if dist_slack:
        dimJ = nref + npv + npq + npq
    else:
        dimJ = npv + npq + npq
    J = sparse((Jx, Jj, Jp), shape=(dimJ, dimJ))

    return J


def _create_J_without_numba(Ybus, V, ref, pvpq, pq, slack_weights, dist_slack):
    # create Jacobian with standard pypower implementation.
    dS_dVm, dS_dVa = dSbus_dV(Ybus, V)

    ## evaluate Jacobian

    if dist_slack:
        rows_pvpq = array(r_[ref, pvpq]).T
        cols_pvpq = r_[ref[1:], pvpq]
        J11 = dS_dVa[rows_pvpq, :][:, cols_pvpq].real
        J12 = dS_dVm[rows_pvpq, :][:, pq].real
    else:
        rows_pvpq = array([pvpq]).T
        cols_pvpq = pvpq
        J11 = dS_dVa[rows_pvpq, cols_pvpq].real
        J12 = dS_dVm[rows_pvpq, pq].real
    if len(pq) > 0 or dist_slack:
        J21 = dS_dVa[array([pq]).T, cols_pvpq].imag
        J22 = dS_dVm[array([pq]).T, pq].imag
        if dist_slack:
            J10 = sparse(slack_weights[rows_pvpq].reshape(-1,1))
            J20 = sparse(zeros(shape=(len(pq), 1)))
            J = vstack([
                hstack([J10, J11, J12]),
                hstack([J20, J21, J22])
            ], format="csr")
        else:
            J = vstack([
                hstack([J11, J12]),
                hstack([J21, J22])
            ], format="csr")
    else:
        J = vstack([
            hstack([J11, J12])
        ], format="csr")
    return J


def _create_J_modification_trafo_taps(Ybus_m, V, ref, pvpq, pq, slack_weights, dist_slack, len_J, len_control, x_control, hv_bus, controlled_bus):
    # todo

    """
    J_m has the shape
    | Jcpd | Jcpu | Jnpc + Jcpc |                      
    | Jcqd | Jcqu | Jnqc + Jcqc |                      
    | Jccd | Jccu |   Jccc      |  
    
    """

    import numpy as np    
    Va = np.angle(V)
    
    V_control = x_control[len_control:] * np.exp(1j*x_control[:len_control])

    dS_dVm, dS_dVa = dSbus_dV(Ybus_m, np.r_[V, V_control])
        
    rows_pvpq = array([pvpq]).T
    cols_pvpq = pvpq
    
   
    # controller_bus_idx = [2]   #### TODO : this must be generlized later and also take into consedaration multi controllers
    #rows_controller_bus = rows_pvpq[np.searchsorted(pvpq, controller_bus_idx)]
    #cols_controller_bus = pvpq[np.searchsorted(pvpq, controller_bus_idx)]

    len_buses = np.max(pvpq)
    rows_controller_bus = np.arange(1+len_buses, 1+len_buses+len_control).reshape(-1,1)
    cols_controller_bus = np.arange(1+len_buses, 1+len_buses+len_control)
    
    dpc_dVa = dS_dVa[rows_controller_bus,cols_controller_bus].real   ##### partial derivatives of the controler active power with repect to controler voltage angel 
    dqc_dVa = dS_dVa[rows_controller_bus,cols_controller_bus].imag   ##### partial derivatives of the controler reactive power with repect to controler voltage angel 
  
    dpc_dVm = dS_dVm[rows_controller_bus,cols_controller_bus].real   ##### partial derivatives of the controler active power with repect to controler voltage magnitude 
    dqc_dVm = dS_dVm[rows_controller_bus,cols_controller_bus].imag   ##### partial derivatives of the controler reactive power with repect to controler voltage magnitude 

    #pvpq_without_controller_bus = np.delete(pvpq, np.searchsorted(pvpq, rows_controller_bus))   #### keep in mind that the slack bus in excluded 
    
    ## formulating the Jm submetrises

    ###  Jcpd: patial derivatives of the active power of the controller bus with respect to Volage angels of the original system.
    Jcpd =  dS_dVa[rows_controller_bus,cols_pvpq].real

    ### Jcqd: patial derivatives of the reactive power of the controller bus with respect to Volage angels of the original system.
    Jcqd =  dS_dVa[rows_controller_bus,cols_pvpq].imag
    
    ### Jcpu: patial derivatives of the active power of the controller bus with respect to Volage magnitude of the original system.
    Jcpu = dS_dVm[rows_controller_bus,cols_pvpq].real   
    
    ### Jcqu: patial derivatives of the reactive power of the controller bus with respect to Volage magnitude of the original system.
    Jcqu = dS_dVm[rows_controller_bus,cols_pvpq].imag   
    
    ### Jcpc: patial derivatives of the active power at the controller bus with respect to controler states variables (x_control)  
    Jcpc = hstack([dpc_dVa, dpc_dVm])

    ### Jcqc: patial derivatives of the reactive power at the controller bus with respect to controler states variables (x_control)
    Jcqc = hstack([dqc_dVa, dqc_dVm])   
        
    ### Jnpc: partial derivatives of the active power mismatch equations of the orginal system with respect to controler states variables (x_control)

    Jnpc = 0  
   
    ### Jnqc: partial derivatives of the reactive power mismatch equations of the orginal system with respect to controler states variables (x_control)
    Jnqc = 0   
    
    
    ### Jccd: partial derivatives of the controller missmatch equations with respect to the voltage angel of the original system

    Va_q = x_control[:len_control] 
    t= np.tan(Va[controlled_bus] - Va_q)
    # t= np.tan(Va[hv_bus] - Va[controlled_bus])
    Jccd =  sparse(((1+t**2),(np.array([0]*len_control),np.arange(0,len(x_control))[len_control:])), shape=(2,len(x_control)))
    
    Jccd = Jccd[:, controlled_bus]
    
    ### Jccu: partial derivatives of the controoler missmatch equations with respect to the voltage magnitude of the original system
    Jccu = sparse((np.array([1]) ,(np.array([1]*len_control),np.arange(0,len(x_control))[len_control:])), shape=(2,len(x_control)))   ###TODO
    
    Jccu = Jccu[:, controlled_bus]       
    ### Jccc: partial derivatives of the controller mismatch eqation with respect to controller states variables (x_control) 
    Jccc = sparse(((-1-t**2),(np.array([0]*len_control),np.arange(0,len(x_control))[:len_control])), shape=(2,len(x_control)))      
    
    ###### stacking all sub metrices together 

    J_m = vstack([hstack([Jcpd,Jcpu,Jnpc+Jcpc]),
                  hstack([Jcqd,Jcqu,Jnqc+Jcqc]),
                  hstack([Jccd, Jccu ,Jccc])], format='csr')
    
    # J_m = sparse((len_J + len_control, len_J + len_control))


    return J_m


def J_modified(Vm, Va, x_control, Ybus_m, branch, tap_control_branches, pvpq, pvpq_lookup, refpvpq, pq, hv_bus, controlled_bus):


    # J = np.zeros(shape=(len(pvpq), len(pvpq)), dtype=np.float64)
    # len_control = int(len(x_control)/2)
    # Va_q = x_control[:len_control]
    # Vm_q = x_control[len_control:]

    # # q_RT_ij =
    # for i in pvpq:
    #     for q, ij in enumerate(tap_control_branches):
    #         f = branch[ij, F_BUS]
    #         t = branch[ij, T_BUS]
    #         if i == f:
    #             j = t
    #         elif i == t:
    #             j = f
    #         else:
    #             continue
    #         q_RT_ij = Vm[i] * np.real(Ybus_m[i,j]) * Vm[j] * np.sin(Va[i] - Va[j] + np.angle(Ybus_m[i,j]))
    #         q_RT_iq = Vm[i] * np.real(Ybus_m[i,q+len(pvpq)+1]) * Vm_q[q] * np.sin(Va[i]-Va_q[q]+ np.angle(Ybus_m[i, q+len(pvpq)+1]))
    #         J[pvpq_lookup[i], pvpq_lookup[i]] = -q_RT_ij - q_RT_iq
    #         J[pvpq_lookup[i], pvpq_lookup[j]] = q_RT_ij


      # import numpy as np

      """   | J_C_Pd | J_C_Pu |               | (pvpq, pvpq) | (pvpq, pq) |
            | --------------- | = dimensions: | ------------------------- |
            | J_C_Qd | J_C_Qu |               |  (pq, pvpq)  |  (pq, pq)  |
      """

      """   | J_N_Pc + J_C_Pc |               | (pvpq, x_control) + (pvpq, x_control)|
            | --------------- | = dimensions: | -------------------------------------|
            | J_N_Qc + J_C_Qc |               |  (pq, x_control)  +  (pq, x_control) |
      """

      """   | J_C_Cd | J_C_Cu | J_C_Cc |  = dimensions:  | (x_control, pvpq) | (x_control, pq) | (x_control, x_control) |
            

      """

      #### create zero n*n matrix where n is the number of system buses
      #### afterward the requried values will be choosen
      #### this is just until we get the right results, then the submatrix can be reshaped.

      J_C_Pd =  np.zeros(shape=(len(pvpq), len(pvpq)), dtype=np.float64)
      J_C_Pu =  np.zeros(shape=(len(pvpq), len(pq)), dtype=np.float64)
      J_C_Qd =  np.zeros(shape=(len(pq), len(pvpq)), dtype=np.float64)
      J_C_Qu =  np.zeros(shape=(len(pq), len(pq)), dtype=np.float64)

      J_C_Pc =  np.zeros(shape=(len(pvpq), len(x_control)), dtype=np.float64)
      J_C_Qc =  np.zeros(shape=(len(pq),  len(x_control)), dtype=np.float64)

      # J_C_Cd =  np.zeros(shape=(len(x_control), len(refpvpq)), dtype=np.float64)
      # J_C_Cu =  np.zeros(shape=(len(x_control), len(refpvpq)), dtype=np.float64)
      # J_C_Cc =  np.zeros(shape=(len(x_control), len(x_control)), dtype=np.float64)



      len_control = int(len(x_control)/2)
      Va_q = x_control[:len_control]
      Vm_q = x_control[len_control:]


      # refpvpq = np.sort(refpvpq)


      for i in pvpq:      ###### TODO modify the dimentions for other submetrix
          for q, ij in enumerate(tap_control_branches):
              f = branch[ij, F_BUS].real.astype(int64)
              t = branch[ij, T_BUS].real.astype(int64)
              if i == f:
                  j = t
              elif i == t:
                  j = f
              else:
                  continue
              
              p_RT_ii = Vm[i]**2 * np.abs(Ybus_m[i,i]) * np.cos(np.angle(Ybus_m[i,i]))
              p_RT_ij = Vm[i] * np.abs(Ybus_m[i,j]) * Vm[j] * np.cos(Va[i] - Va[j] + np.angle(Ybus_m[i,j]))
              p_RT_iq = Vm[i] * np.abs(Ybus_m[i,q+len(pvpq)+1]) * Vm_q[q] * np.cos(Va[i]-Va_q[q]+ np.angle(Ybus_m[i, q+len(pvpq)+1]))
              
              
          
              q_RT_ii = Vm[i]**2 * np.abs(Ybus_m[i,i]) * np.sin(np.angle(Ybus_m[i,i]))
              q_RT_ij = Vm[i] * np.abs(Ybus_m[i,j]) * Vm[j] * np.sin(Va[i] - Va[j] + np.angle(Ybus_m[i,j]))
              q_RT_iq = Vm[i] * np.abs(Ybus_m[i,q+len(pvpq)+1]) * Vm_q[q] * np.sin(Va[i]-Va_q[q]+ np.angle(Ybus_m[i, q+len(pvpq)+1]))
              


             # J_C_Pd[refpvpq[i], refpvpq[j]] = q_RT_ij
              #J_C_Pd[refpvpq[i], refpvpq[i]] = -q_RT_ij - q_RT_iq       

              J_C_Pd[pvpq_lookup[i], pvpq_lookup[j]] = q_RT_ij
              J_C_Pd[pvpq_lookup[i], pvpq_lookup[i]] = -q_RT_ij - q_RT_iq

              # J_C_Pu[refpvpq[i], refpvpq[j]]= p_RT_ij    
              # J_C_Pu[refpvpq[i], refpvpq[i]] = 2*p_RT_ii + p_RT_ij + p_RT_iq

              
              J_C_Pu[pvpq_lookup[i], pvpq_lookup[j]] = p_RT_ij    
              J_C_Pu[pvpq_lookup[i], pvpq_lookup[i]] = 2*p_RT_ii + p_RT_ij + p_RT_iq


              J_C_Qd[pvpq_lookup[i], pvpq_lookup[j]] = -p_RT_ij   
              J_C_Qd[pvpq_lookup[i], pvpq_lookup[i]] =  p_RT_ij + p_RT_iq

              J_C_Qu[pvpq_lookup[i], pvpq_lookup[j]] = q_RT_ij   
              J_C_Qu[pvpq_lookup[i], pvpq_lookup[i]] = 2*q_RT_ii + q_RT_ij + q_RT_iq



      ###### here we exclude the slack bus row and column values
      ###### and reshape the matrices according to the dimentions.
 
      #J_C_Pd = J_C_Pd[pvpq.reshape(-1,1),pvpq]
      J_C_Pd = sparse(J_C_Pd)              

      #J_C_Pu = J_C_Pu[pvpq.reshape(-1,1),pq]
      J_C_Pu = sparse(J_C_Pu)              


      #J_C_Qd = J_C_Qd[pq.reshape(-1,1),pvpq]
      J_C_Qd = sparse(J_C_Qd)              


      #J_C_Qu = J_C_Qu[pq.reshape(-1,1),pq]
      J_C_Qu = sparse(J_C_Qu)              






#########################################################################################




      x_control_lookup = np.arange(len(x_control))
      x_control_lookup_Va_q = x_control_lookup[:len_control]
      x_control_lookup_Vm_q = x_control_lookup[len_control:]


      for i in pvpq:      ###### TODO modify the dimentions for other submetrix
          for q, ij in enumerate(tap_control_branches):
              f = branch[ij, F_BUS].real.astype(int64)
              t = branch[ij, T_BUS].real.astype(int64)
              if i == f:
                  j = t
              elif i == t:
                  j = f
              else:
                  continue
              
                
              
              p_RT_iq = Vm[i] * np.abs(Ybus_m[i,q+len(pvpq)+1]) * Vm_q[q] * np.cos(Va[i]-Va_q[q]+ np.angle(Ybus_m[i, q+len(pvpq)+1]))
              q_RT_iq = Vm[i] * np.abs(Ybus_m[i,q+len(pvpq)+1]) * Vm_q[q] * np.sin(Va[i]-Va_q[q]+ np.angle(Ybus_m[i, q+len(pvpq)+1]))

            

              J_C_Pc[pvpq_lookup[i], x_control_lookup_Va_q[q]]  = q_RT_iq
              J_C_Pc[pvpq_lookup[i], x_control_lookup_Vm_q[q]] = p_RT_iq
           
              J_C_Qc[pvpq_lookup[i], x_control_lookup_Va_q[q]] = -p_RT_iq
              J_C_Qc[pvpq_lookup[i], x_control_lookup_Vm_q[q]]  = q_RT_iq  

                  

      # J_C_Pc = np.reshape(np.stack((J_C_Pc[:,hv_bus],J_C_Pc[:,controlled_bus]), axis=1),(len(pvpq),len(x_control)))
      # J_C_Pc = J_C_Pc[pvpq]

      J_C_Pc = sparse(J_C_Pc)              

      
      # J_C_Qc = np.reshape(np.stack((J_C_Qc[:,hv_bus],J_C_Qc[:,controlled_bus]), axis=1),(len(refpvpq),len(x_control)))
      # J_C_Qc = J_C_Qc[pq]            

      J_C_Qc = sparse(J_C_Qc)              



############################################################################################################################################



      t = np.tan(Va[controlled_bus] - Va_q)
    # t = np.tan(Va[hv_bus] - Va[controlled_bus])


      ### J_C_Cd: partial derivatives of the controller missmatch equations with respect to the voltage angel of the original system

      J_C_Cd =  sparse(((1+t**2),(np.arange(0,len_control),controlled_bus)), shape=(len(x_control),len(refpvpq)))
      J_C_Cd = J_C_Cd[:, pvpq]

      
      ### J_C_Cu: partial derivatives of the controoler missmatch equations with respect to the voltage magnitude of the original system
      J_C_Cu = sparse((np.array([1]*len_control),(np.arange(0,len(x_control))[len_control:],controlled_bus)), shape=(len(x_control),len(refpvpq)))   
      J_C_Cu = J_C_Cu[:, pq]
    
      ### J_C_Cc: partial derivatives of the controller mismatch eqation with respect to controller states variables (x_control) 
      J_C_Cc = sparse((r_[(-1-t**2),np.array([-1]*len_control)],(np.arange(0,len(x_control)),np.arange(0,len(x_control)))), shape=(len(x_control),len(x_control)))

      J_C_Cc[len_control:, len_control:] = 0
      

      
      J_m = vstack([hstack([J_C_Pd,J_C_Pu,J_C_Pc]),
                    hstack([J_C_Qd,J_C_Qu,J_C_Qc]),
                    hstack([J_C_Cd, J_C_Cu ,J_C_Cc])], format='csr')
      
  
      return J_m




def create_jacobian_matrix(Ybus, V, ref, refpvpq, pvpq, pq, createJ, pvpq_lookup, nref, npv, npq, numba, slack_weights, dist_slack, trafo_taps, x_control, Ybus_m, hv_bus, controlled_bus,
                            Vm,Va, branch,tap_control_branches):
    if numba:
        J = _create_J_with_numba(Ybus, V, refpvpq, pvpq, pq, createJ, pvpq_lookup, nref, npv, npq, slack_weights, dist_slack)
    else:
        J = _create_J_without_numba(Ybus, V, ref, pvpq, pq, slack_weights, dist_slack)
    if trafo_taps:
        # todo: implement J_m for trafo taps
        # J_m = _create_J_modification_trafo_taps(Ybus_m, V, ref, pvpq, pq, slack_weights, dist_slack, J.shape[0],
        #                                         int(len(x_control) / 2), x_control, hv_bus, controlled_bus)


        J_m = J_modified(Vm, Va, x_control, Ybus_m, branch, tap_control_branches, pvpq, pvpq_lookup, refpvpq, pq, hv_bus, controlled_bus)

        K_J = vstack([eye(J.shape[0], format="csr"), sparse((len(x_control), J.shape[0]))], format="csr")
        J_nr = K_J * J * K_J.T  # this extends the J matrix with 0-rows and 0-columns

        J = J_nr + J_m
    return J


def get_fastest_jacobian_function(pvpq, pq, numba, dist_slack):
    if numba:
        if dist_slack:
            create_jacobian = create_J_ds
        elif len(pvpq) == len(pq):
            create_jacobian = create_J2
        else:
            create_jacobian = create_J
    else:
        create_jacobian = None
    return create_jacobian
