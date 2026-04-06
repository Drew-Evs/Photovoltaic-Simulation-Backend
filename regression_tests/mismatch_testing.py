from pvmismatch import pvsystem
from pvmismatch.pvmismatch_lib import pvmodule, pvstring
import matplotlib.pyplot as plt

#testing with a custom layout
custom_layout = pvmodule.standard_cellpos_pat(nrows=8, ncols_per_substr=[2, 2, 2])
panel_48 = pvmodule.PVmodule(cell_pos=custom_layout)

#convert to sys
my_string = pvstring.PVstring(numberMods=1, pvmods=[panel_48])
sys = pvsystem.PVsystem(numberStrs=1, pvstrs=[my_string])
print(f"Total cells: {sys.pvstrs[0].pvmods[0].numberCells}")
print(f"Number of bypass diodes: {sys.pvstrs[0].pvmods[0].numSubStr}")
print(f"Unshaded Max Power of Panel: {sys.Pmp:.2f} W")

#create a shading string of cell irradiance - first 3 cells down 20%
cell_irradiance = [1.0] * 48
cell_irradiance[0] = 0.2 
cell_irradiance[1] = 0.2
cell_irradiance[2] = 0.2

#apply the shade
sys.setSuns({0: {0: cell_irradiance}})
print(f"Shaded Max Power: {sys.Pmp:.2f} W")

#plot iv curve
f, ax = plt.subplots(1, 2, figsize=(10, 4))
sys.plotSys(f)
plt.show()