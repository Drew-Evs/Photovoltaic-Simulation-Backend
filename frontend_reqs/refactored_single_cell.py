import cell_ann
import refactored_prediction
import numpy as np

#a class used to calculate the parameters of a single cell
class Cell():
    #ANN at the class level only needs to be initiated once
    model = None
    X_scaler = None
    y_scaler = None

    #using rs as a constant
    rs = None

    #creating ann at class level
    @classmethod
    def initiate_class(cls, module_name, Ns):
        cls.model, cls.X_scaler, cls.y_scaler =  cell_ann.create_optimal_ann(module_name)

        #only need to get the rs everything else calculated by ann
        _, _, cls.rs, _, _, = refactored_prediction.getting_parameters(25, 1000, module_name)
        cls.rs = cls.rs/Ns

    #initiate with a irradiance, temperature and conditions from the datasheet
    #temp in kelvin
    def __init__(self, irr, temp, datasheet_conditions, module):
        self.irradiance = irr
        self.temperature = temp
        self.module_name = module

        self.kT = temp + 273.15

        self.isc, self.vmp, self.voc, self.imp, self.Ns = datasheet_conditions
        self.voc_per_cell = self.voc/self.Ns

        if Cell.model is None:
            Cell.initiate_class(self.module_name, self.Ns)

        self.predict_params()

    #using i-v law to calc voltage
    def iv_equation(self, V, I):
        exponent = (V + I * Cell.rs)/(self.a)
        exponent_term = self.isat * (np.exp(np.clip(exponent, -50, 50)) -1)
        rsh_term = (V + I * Cell.rs)/self.rsh
        return (self.iph - exponent_term - rsh_term - I)

    #use the ann
    def predict_params(self):
        X = [[self.irradiance, self.temperature]]
        X_scaled = Cell.X_scaler.transform(X)

        y_scaled = Cell.model.predict(X_scaled)
        y = Cell.y_scaler.inverse_transform(y_scaled)

        self.iph, log_isat, self.rsh, self.a = y[0]

        #reverse the logarithm
        self.isat = 10 ** log_isat

    #set the irradiance and recalculate
    def shade(self, irr):
        self.irradiance = irr

    #set the temperature and recalculate
    def set_temp(self, temp):
        self.temperature = temp
        self.kT = temp + 273.15

    def get_params(self):
        return self.a, self.iph, self.isat, Cell.rs, self.rsh