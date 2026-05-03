import numpy as np
import cantera as ct

def _gri30_full_config() -> ModelConfig:
    """
    GRI-Mech 3.0 methane / air full network (53 states, 325 effective k coefficients).
    """
    # 1. Load the real GRI-Mech 3.0 database using Cantera
    # Cantera comes with gri30.yaml pre-installed
    gas = ct.Solution('gri30.yaml')
    
    # 2. Set realistic combustion conditions
    # T = 1500 Kelvin, P = 1 atm, stoichiometric CH4/Air
    gas.TPX = 1500.0, ct.one_atm, 'CH4:1.0, O2:2.0, N2:7.52'
    
    # 3. Extract the 325 forward rate constants
    # These are physically accurate parameters spanning many orders of magnitude
    real_k_params = gas.forward_rate_constants.tolist()
    
    # Optional: If your ODE simulator struggles with extreme stiffness, 
    # you can clip the highest rates, but try the raw physical rates first!
    param_ranges = [(v, v) for v in real_k_params]

    # 53-state initial condition: stoichiometric CH4 / O2 / N2 mix.
    x0_default = [0.0] * 53
    x0_default[13] = 1.0   # CH4
    x0_default[3]  = 2.0   # O2
    x0_default[47] = 7.52  # N2 (diluent)

    bolus_ranges = {
        "CH4": (0.05, 0.5),
        "O2":  (0.1,  1.0),
        "N2":  (0.5,  3.0),
        "H2O": (0.1,  1.0),
    }

    return ModelConfig(
        param_ranges=param_ranges,
        x0_default=x0_default,
        bolus_ranges=bolus_ranges,
        bolus_default=(0.1, 1.0),
        bolus_count_range=(2, 8),
        simulator="ivp", # Make sure you are using 'Radau' or 'BDF' in your ivp solver for stiffness!
    )