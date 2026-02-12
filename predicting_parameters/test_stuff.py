import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# IV equation
def iv_equation(V, I, iph, isat, rs, rsh, nNsVth):
    exponent = (V + I * rs) / nNsVth
    exponent_term = isat * (np.exp(np.clip(exponent, -50, 50)) - 1)
    rsh_term = (V + I * rs) / rsh
    return iph - exponent_term - rsh_term - I

# Parameters
iph = 9.774064000000001
isat = 6.598872105885762e-10
rs = 0.0040598395061755424
rsh = 12.077061777470085
nNsVth = 0.026539065030651992

# Currents
currents = np.array([0., 0.09872792, 0.19745584, 0.29618376, 0.39491168, 0.4936396,
                     0.59236752, 0.69109543, 0.78982335, 0.88855127, 0.98727919, 1.08600711,
                     1.18473503, 1.28346295, 1.38219087, 1.48091879, 1.57964671, 1.67837463,
                     1.77710255, 1.87583046, 1.97455838, 2.0732863, 2.17201422, 2.27074214,
                     2.36947006, 2.46819798, 2.5669259, 2.66565382, 2.76438174, 2.86310966,
                     2.96183758, 3.06056549, 3.15929341, 3.25802133, 3.35674925, 3.45547717,
                     3.55420509, 3.65293301, 3.75166093, 3.85038885, 3.94911677, 4.04784469,
                     4.14657261, 4.24530053, 4.34402844, 4.44275636, 4.54148428, 4.6402122,
                     4.73894012, 4.83766804, 4.93639596, 5.03512388, 5.1338518, 5.23257972,
                     5.33130764, 5.43003556, 5.52876347, 5.62749139, 5.72621931, 5.82494723,
                     5.92367515, 6.02240307, 6.12113099, 6.21985891, 6.31858683, 6.41731475,
                     6.51604267, 6.61477059, 6.71349851, 6.81222642, 6.91095434, 7.00968226,
                     7.10841018, 7.2071381, 7.30586602, 7.40459394, 7.50332186, 7.60204978,
                     7.7007777, 7.79950562, 7.89823354, 7.99696145, 8.09568937, 8.19441729,
                     8.29314521, 8.39187313, 8.49060105, 8.58932897, 8.68805689, 8.78678481,
                     8.88551273, 8.98424065, 9.08296857, 9.18169648, 9.2804244, 9.37915232,
                     9.47788024, 9.57660816, 9.67533608, 9.774064])

# Solve for voltages
voltages = []
for I in currents:
    v_guess = 0.7
    V_sol = fsolve(iv_equation, v_guess, args=(I, iph, isat, rs, rsh, nNsVth))[0]
    voltages.append(V_sol)

# Convert to NumPy array for convenience
voltages = np.array(voltages)

# Print results
for I, V in zip(currents, voltages):
    print(f"I = {I:.6f} A, V = {V:.6f} V")

plt.figure()
plt.plot(voltages, currents)
plt.xlabel("Voltage (V)")
plt.ylabel("Current (A)")
plt.title("Kill me")
plt.grid(True)

# save
plt.savefig("Kill me2", dpi=300, bbox_inches="tight")
plt.close()
