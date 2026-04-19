import socket
import time
import numpy as np
import matplotlib.pyplot as plt

#want to generate cells correctly for pvmismatch
from pvmismatch import pvsystem
from pvmismatch.pvmismatch_lib import pvmodule, pvstring, pvcell
from pvmismatch.pvmismatch_lib.pvconstants import PVconstants

#for generating own panel
import pvlib
from refactored_whole_module import Module
from DPSO_MPPT import DPSO_MPPT

#generate a dashboard with results 
def build_figure(mismatch_panel, tracker, module, irr_list):
    #a 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Single-Step Solar Simulation Dashboard', fontsize=16)

    #the 6 columns of the panel
    cols = 6 
    cells = len(irr_list)
    rows = int(np.ceil(cells / cols))
    padded_irr = irr_list + [0] * ((rows * cols) - cells)

    #reshape into a matrix for the figure
    irr_matrix = np.array(padded_irr).reshape((rows, cols))
    im = axs[0, 0].imshow(irr_matrix, cmap='inferno', interpolation='nearest', vmin=0, vmax=1000, origin='lower')
    axs[0, 0].set_title(f'Panel Shading Profile ({cells} cells)')
    fig.colorbar(im, ax=axs[0, 0], label='Irradiance (W/m²)')

    #generate the pvmismatch model
    try:
        raw_suns = np.array(irr_list[:48]) / 1000.0
        suns_array = np.clip(raw_suns, 0.001, None)
        mismatch_panel.setSuns({0: {0: suns_array}})
        V_pvmm = mismatch_panel.Vsys.flatten()
        P_pvmm = mismatch_panel.Psys.flatten()

        #clean negative voltages
        valid_points = P_pvmm >= 0
        V_clean = V_pvmm[valid_points]
        P_clean = P_pvmm[valid_points]
        
        #plot P-V curve
        axs[0, 1].plot(V_clean, P_clean, label='P-V Curve', color='red', linestyle='--')
        
        axs[0, 1].set_title(f'PVMismatch Output {np.max(P_clean):.3f}W')
        axs[0, 1].set_xlabel('Voltage (V)')
        axs[0, 1].set_ylabel('Power (W/m^2)', color='blue')
        
    except Exception as e:
        axs[0, 1].set_title('PVMismatch Error')
        axs[0, 1].text(0.5, 0.5, str(e), ha='center', va='center', wrap=True)

    #set conditions for modules and tracker
    module.set_cell_conditions(irr_array=irr_list)

    #calculate iv 
    original_v, original_p = module.calculate_iv()
    track_v, track_p = tracker.track_mpp()

    #generate curves and mppt point
    axs[1, 0].set_title(f'Implicit Model Output & MPPT {track_p:.3f}W')
    axs[1, 0].plot(original_v, original_p, 'g-', label='Implicit P-V')
    axs[1, 0].plot(track_v, track_p, 'ro', markersize=6, label='Tracked MPP')
    axs[1, 0].set_xlabel('Voltage (V)')
    axs[1, 0].set_ylabel('Power (W/m^2)', color='blue')
    axs[1, 0].legend()

    #calculate explicit curve
    refactored_v, refactored_p = module.refactored_iv()
    axs[1, 1].plot(refactored_v, refactored_p, 'g-', label='Explicit P-V')

    axs[1, 1].set_title(f'Explicit Model Output {np.max(refactored_p):.3f}W')
    axs[1, 1].set_xlabel('Voltage (V)')
    axs[1, 1].set_ylabel('Power (W/m^2)', color='blue')

    plt.tight_layout()
    plt.show()

def run_udp_server(mismatch_panel, tracker, module):
    #configure network
    UDP_IP = "127.0.0.1"
    UDP_PORT = 5005

    #create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    print(f'Data Harvester listening on {UDP_IP}:{UDP_PORT}')
    print("Waiting for Unity to start sending data...")

    try:
        # wati for unity packet
        data, addr = sock.recvfrom(4096)

        message = data.decode('utf-8')

        #convert to irradiance list
        irr_list = [float(x) for x in message.split(',')]

        #handshake format
        reply_message = "0.00,DataSaved"
        sock.sendto(reply_message.encode('utf-8'), addr)
        print(f"Harvested 1 step with {len(irr_list)} cells.")

        sock.close()
        build_figure(mismatch_panel, tracker, module, irr_list)
            
    except Exception as e:
        print(f"Error processing packet: {e}")
        sock.close()

    #let write early
    except KeyboardInterrupt:
        print("\nSimulation manually stopped by user (Ctrl+C).")
        sock.close()

if __name__ == "__main__":
    #create the module
    cec_modules = pvlib.pvsystem.retrieve_sam('CECmod')
    module = cec_modules['Prism_Solar_Technologies_Bi48_267BSTC']
    module_name = 'Prism_Solar_Technologies_Bi48_267BSTC'
    datasheet_conditions = (
        module['I_sc_ref'], 
        module['V_mp_ref'], 
        module['V_oc_ref'], 
        module['I_mp_ref'],
        module['N_s']
    )
    module = Module(datasheet_conditions, 'Prism_Solar_Technologies_Bi48_267BSTC')

    #create the mppt tracker
    tracker = DPSO_MPPT(module.d, module, 0, module.voc)

    #create the pvmismatch version
    custom_const = PVconstants()
    a, iph, isat, rs, rsh = module.cell_list[0].get_params()
    T = 298.15 
    Vt_standard = (custom_const.k * T) / custom_const.q
    n_ideal = a / Vt_standard
    custom_const.k = custom_const.k * n_ideal

    lg_cells = [
        pvcell.PVcell(Rs=rs, Rsh=rsh, Isat1_T0=isat, Isat2_T0=0, Isc0_T0=iph, pvconst=custom_const, alpha_Isc=datasheet_conditions[0]) 
        for _ in range(48)
    ]

    custom_layout = pvmodule.standard_cellpos_pat(nrows=8, ncols_per_substr=[2, 2, 2])
    panel_48 = pvmodule.PVmodule(cell_pos=custom_layout, pvcells=lg_cells)
    panel_string = pvstring.PVstring(numberMods=1, pvmods=[panel_48])
    pvmm_sys = pvsystem.PVsystem(numberStrs=1, pvstrs=[panel_string])

    run_udp_server(pvmm_sys, tracker, module)