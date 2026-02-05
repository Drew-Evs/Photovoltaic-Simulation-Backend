import numpy as np
import pvlib
from scipy.optimize import least_squares, fsolve
import matplotlib.pyplot as plt

##using least square method to minimise the residuals
##residuals based on equations that require accurate voltage and current to = 0
##will allow the model to find voltage and current per cell in a model
##based on work: https://www.sciencedirect.com/science/article/pii/S0038092X06003070

#getting a panel to test
cec_modules = pvlib.pvsystem.retrieve_sam('CECmod')
module = cec_modules['SunPower_SPR_X20_327']
Ns = module['N_s']

#reference boltzmans and electrical charge
k = 1.38e-23
q = 1.6e-19

#a cell class to store data - start by simply using the pv_lib
#takes in cell temperature and irradiance to calculate values
class cell():
    def __init__(self, irr, temp):
        self.irr = irr
        self.temp = temp
        self.kT = temp + 273.15
        self.calc_params()
        self.get_resist()
        self.get_max_voltage()
        print(f"Iph={self.iph}, Isat={self.isat}, Rs={self.rs}, Rsh={self.rsh}, n={self.n}")

    #test run using pv libs
    def calc_params(self):
        Iph, Isat, Rs, Rsh, nNsVth = pvlib.pvsystem.calcparams_desoto(
            effective_irradiance=self.irr,
            temp_cell=self.temp,
            alpha_sc=module['alpha_sc'],
            a_ref=module['a_ref'],
            I_L_ref=module['I_L_ref'],
            I_o_ref=module['I_o_ref'],
            R_sh_ref=module['R_sh_ref'],
            R_s=module['R_s'],
            EgRef=1.121,
            dEgdT=-0.0002677
        )

        #this calculates whole module
        #current stays constant, but rsh and rs dont scale linearly
        self.nNsVth = nNsVth
        self.iph = Iph
        self.isat = Isat
        self.rs = Rs
        self.rsh = Rsh
        self.n = q*nNsVth/(k*self.kT*Ns)

    #get the full voltage of the panel and divide by number of cells to get the correct voltage
    #need to get a series of points
    #then from taht with a known isat, n and iph can extract Rs and Rsh
    def get_points(self):
        max_v = 0
        Vt = k * self.kT / q
        voltages = []

        #using i-v law
        def iv_equation(V, I):
            exponent = (V + I * self.rs)/(self.nNsVth)
            exponent_term = self.isat * (np.exp(np.clip(exponent, -50, 50)) -1)
            rsh_term = (V + I * self.rs)/self.rsh
            return (self.iph - exponent_term - rsh_term - I)

        current_targets = np.linspace(0, self.iph*0.99, 30)
        for I in current_targets:
            v_guess = 0.3
            sol = fsolve(iv_equation, v_guess, args=(I,))[0]

            voltages.append(sol)

            if sol > max_v:
                max_v = sol

        #getting the correct points
        voltages = [V/Ns for V in voltages]
        powers = [V*I for V, I in zip(voltages, current_targets)]

        #pmp is a known point
        max_index = np.argmax(powers)
        max_point = (voltages[max_index], current_targets[max_index])

        #find current where voltage = 0
        voltages = np.array(voltages)
        idx = np.where(voltages==0)[0]
        min_point = (0, current_targets[idx])

        #output of the 3 points
        output = [(V, I) for V, I in zip(voltages, current_targets)]
        return output

    #using the known points on the curve to find the rs and rsh of a single cell
    #use least square to fit to the curve
    def get_resist(self):

        points = self.get_points()
        Vt = k * self.kT / q

        def residuals(x):
            Rs, Rsh = x
            res = []
            for V, I in points:
                I_calc = self.iph - self.isat*(np.exp(np.clip((V + I*Rs)/(self.n*Vt), -50, 50)) - 1) - (V + I*Rs)/Rsh
                res.append(I_calc - I)
            return res

        # initial guess
        x0 = [0.1, 1000]
        sol = least_squares(residuals, x0, bounds=([0, 0],[np.inf, np.inf]))
        self.rs, self.rsh = sol.x

        # I_points = np.linspace(0, self.iph*0.99, 100)
        # V_points = []

        # def iv_eq(V, I):
        #     exponent = (V + I * self.rs)/(self.n*Vt)
        #     exponent_term = self.isat * (np.exp(np.clip(exponent, -50, 50)) -1)
        #     rsh_term = (V + I * self.rs)/self.rsh
        #     return (self.iph - exponent_term - rsh_term - I)

        # for I in I_points:
        #     V_guess = 0.7
        #     V_sol = fsolve(iv_eq, V_guess, args=(I, ))[0]
        #     V_points.append(V_sol)

        # # Optionally plot it
        # plt.plot(V_points, I_points)
        # plt.xlabel("Voltage (V)")
        # plt.ylabel("Current (A)")
        # plt.title("Single Cell I-V Curve")
        # plt.grid(True)
        # plt.savefig("single_cell_iv_curve.png")

    #gets the max voltage (when current = 0)
    def get_max_voltage(self):
        Vt = k * self.kT / q
        def iv_eq(V):
            exponent = V/(self.n*Vt)
            exponent_term = self.isat * (np.exp(np.clip(exponent, -50, 50)) -1)
            rsh_term = V/self.rsh
            return (self.iph - exponent_term - rsh_term)

        v_guess = 0.7
        self.max_voltage = fsolve(iv_eq, v_guess)[0]

current_target_weight = 5

#minimises to find results
def residuals(x, cells, d_count, current_target):
    #need the number of cells, number of diodes and the number of cells per diode
    c = len(cells)
    d = d_count
    p = c//d

    #gets the current values of voltage and current from x (cells and bypass diodes)
    v_c = x[0:c]
    i_c = x[c:2*c]
    v_bd = x[2*c:2*c+d]
    i_bd = x[2*c+d:2*c+2*d]
    voltage_load = x[-1]

    #stores the residuals
    res = [] 

    #eq 7 load voltage
    res.append(np.sum(v_c) - voltage_load)

    #eq 8 mesh equations for each bd loop
    #the voltage in the byass diode should be opposite to that in the cells
    for i in range(d):
        start, end = i*p, (i+1)*p
        res.append(v_bd[i] + np.sum(v_c[start:end]))

    #eq 9 - each cell should have the same current inside bd loops
    for i in range(c):
        res.append((current_target - i_c[i])*current_target_weight)

    #eq 10-11 current needs to be the same at the junctions
    #i.e. each section needs to match the previouse
    # for i in range(d-1):
    #     current = i_c[i*p] + i_bd[i]
    #     current_next = i_c[(i+1)*p] + i_bd[i+1]
    #     res.append(current - current_next)

    #eq 12 single cell current voltage relation
    #using constant values for boltzmans and electrical charge
    for i, cell in enumerate(cells):
        exponent = q*(v_c[i] + i_c[i]*cell.rs)/(cell.n*k*cell.kT)
        exp_term = cell.isat * np.exp(np.clip(exponent, -50, 50))
        rsh_term = (v_c[i] + i_c[i]*cell.rs)/cell.rsh
        res.append(-i_c[i] + cell.iph - exp_term - rsh_term)

    #eq 13 same for the bypass diodes
    #using constants for saturation current ideality and temperature
    i_sbd = 1.6e-9
    n_bd = 1
    t_bd = 308.5 #(kelvin)
    for i in range(d):
        #adding a resistance to the circuit
        v_eff = v_bd[i] - i_bd[i]*0.05
        arg_bd = np.clip((q * v_bd[i]) / (n_bd * k * t_bd), -50, 50)
        res.append(-i_bd[i] + i_sbd * (np.exp(arg_bd) - 1))

    return res

#test run
module_cells = [cell(1000, 25), cell(700, 25), cell(100, 25), cell(500, 25)]
max_iph = max(cell.iph for cell in module_cells)
for i, c in enumerate(module_cells):
    print(f"Cell {i}")
    print(f" iph = {c.iph:.6e} A")
    print(f" isat = {c.isat:.6e} A")
    print(f" rs = {c.rs:.6f} Ω")
    print(f" rsh = {c.rsh:.6f} Ω")
    print(f" n = {c.n:.4f}")
    print("-" * 30)

c = len(module_cells)
d = 2
p = c//2

low_bound = np.array([-0.7]*c + [-np.inf]*(c + d + d) + [-0.7*c])
low_bound[c:2*c] = 0
high_bound = np.array([np.inf]*(c + c + d + d + 1))
high_bound[0:c] = [cell.max_voltage for cell in module_cells]

powers = []
voltages = []

#test all currents between the max iph and 0
current_targets = np.linspace(0, max_iph, 30)

for current_target in current_targets:
    #intial guess
    x0 = np.concatenate([
        [0.4]*c,
        [current_target]*c,
        [0.0]*d,
        [0.0]*d,
        [np.sum([cell.max_voltage for cell in module_cells])]
    ])

    solution = least_squares(residuals, x0, bounds=(low_bound, high_bound),
        args=(module_cells, d, current_target))

    #get the bd/cells currents and voltages
    v_c = solution.x[0:c]
    i_c = solution.x[c:2*c]
    v_bd = solution.x[2*c:2*c+d]
    i_bd = solution.x[2*c+d:2*c+2*d]
    voltage_load = solution.x[-1]

    print("-"*50)
    print(f"For target {current_target:.2f}")

    print("\nCELLS")
    print(f"{'Index':>5} | {'V_c (V)':>8} | {'I_c (A)':>8}")
    print("-" * 28)
    for i in range(c):
        print(f"{i:5d} | {v_c[i]:8.2f} | {i_c[i]:8.2f}")


    print("\nBYPASS DIODES")
    print(f"{'Index':>5} | {'V_bd (V)':>9} | {'I_bd (A)':>9}")
    print("-" * 31)
    for i in range(d):
        print(f"{i:5d} | {v_bd[i]:9.2f} | {i_bd[i]:9.2f}")

    print("\nLOAD VOLTAGE")
    print(f'Load Voltage is {voltage_load}')

    #set voltage to 0 if bd is on
    for i in range(d):
        if i_bd[i] > 0:
            for j in range(p):
                v_c[p*i+j] = 0

    #calculate power voltage and current
    voltage = np.sum(v_c)
    power = np.sum(v_c*i_c)

    powers.append(power)
    voltages.append(voltage_load)

voltages = np.array(voltages)
currents = np.array(current_targets)
powers   = np.array(powers)

# # I–V
# plt.figure()
# plt.plot(voltages, currents)
# plt.xlabel("Voltage (V)")
# plt.ylabel("Current (A)")
# plt.title("I–V Curve")
# plt.grid(True)
# plt.savefig("iv_curve.png", dpi=300, bbox_inches="tight")
# plt.close()

# # P–V
# plt.figure()
# plt.plot(voltages, powers)
# plt.xlabel("Voltage (V)")
# plt.ylabel("Power (W)")
# plt.title("P–V Curve")
# plt.grid(True)
# plt.savefig("pv_curve.png", dpi=300, bbox_inches="tight")
# plt.close()

# # P–I
# plt.figure()
# plt.plot(currents, powers)
# plt.xlabel("Current (A)")
# plt.ylabel("Power (W)")
# plt.title("P–I Curve")
# plt.grid(True)
# plt.savefig("pi_curve.png", dpi=300, bbox_inches="tight")
# plt.close()