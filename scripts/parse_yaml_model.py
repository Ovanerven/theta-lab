import re
import numpy as np
import cantera as ct

def sanitize_name(name):
    """Converts chemical symbols into valid Python variable names."""
    safe = re.sub(r'[^a-zA-Z0-9]', '_', name)
    if safe[0].isdigit():
        safe = 'sp_' + safe
    return safe

def generate_explicit_model(yaml_file, output_file, func_name="Aramco_FullModel"):
    print(f"Loading mechanism from {yaml_file}...")
    gas = ct.Solution(yaml_file)
    species_names = gas.species_names
    reactions = gas.reactions()
    
    n_spec = gas.n_species
    n_rxns = gas.n_reactions

    safe_names = [sanitize_name(sp) for sp in species_names]
    name_map = dict(zip(species_names, safe_names))

    lines = []
    
    # 1. Imports
    lines.append("import numpy as np")
    lines.append("from numba import njit\n")

    # 2. The JIT-compiled Pure Math Core
    lines.append("@njit")
    lines.append("def _math_core(t: float, y: np.ndarray, k: np.ndarray) -> np.ndarray:")
    
    # Unpack y using direct indexing to prevent Numba tuple limit errors
    lines.append("    # Unpack species")
    for i, sp in enumerate(safe_names):
        lines.append(f"    {sp} = y[{i}]")
    lines.append("")

    # Unpack k using direct indexing
    lines.append("    # Unpack effective reaction coefficients")
    for i in range(n_rxns):
        lines.append(f"    k{i+1} = k[{i}]")
    lines.append("")

    # Third-body M concentrations
    lines.append("    # Third-body / falloff effective mixture concentrations")
    m_dict = {}
    for i, rxn in enumerate(reactions):
        if hasattr(rxn, 'efficiencies'):
            eff = rxn.efficiencies
            default_eff = rxn.default_efficiency
            m_terms = []
            for sp in species_names:
                eff_val = eff.get(sp, default_eff)
                if eff_val != 0:
                    safe_sp = name_map[sp]
                    if eff_val == 1.0:
                        m_terms.append(safe_sp)
                    else:
                        m_terms.append(f"{eff_val}*{safe_sp}")
            if m_terms:
                m_str = " + ".join(m_terms)
                lines.append(f"    M{i+1} = {m_str}")
                m_dict[i+1] = f"M{i+1}"
    lines.append("")

    # Reaction Rates (r_i)
    lines.append("    # Reaction rates")
    species_derivatives = {sp: [] for sp in species_names}
    
    for i, rxn in enumerate(reactions):
        fwd_terms = []
        for sp, stoich in rxn.reactants.items():
            safe_sp = name_map[sp]
            if stoich == 1: fwd_terms.append(safe_sp)
            else: fwd_terms.append(f"({safe_sp}**{stoich})")
        fwd_str = " * ".join(fwd_terms) if fwd_terms else "1.0"
        
        rev_terms = []
        if rxn.reversible:
            for sp, stoich in rxn.products.items():
                safe_sp = name_map[sp]
                if stoich == 1: rev_terms.append(safe_sp)
                else: rev_terms.append(f"({safe_sp}**{stoich})")
        rev_str = " * ".join(rev_terms) if rev_terms else ""

        if rev_str:
            rate_eq = f"k{i+1} * ({fwd_str} - {rev_str})"
        else:
            rate_eq = f"k{i+1} * ({fwd_str})"

        if (i+1) in m_dict:
            rate_eq += f" * {m_dict[i+1]}"

        lines.append(f"    r{i+1} = {rate_eq}")

        for sp, stoich in rxn.reactants.items():
            species_derivatives[sp].append(f"- {stoich}*r{i+1}" if stoich != 1 else f"- r{i+1}")
        for sp, stoich in rxn.products.items():
            species_derivatives[sp].append(f"+ {stoich}*r{i+1}" if stoich != 1 else f"+ r{i+1}")
            
    lines.append("")

    # Species Balances (dY)
    lines.append("    # Species balances")
    dy_names = []
    for sp in species_names:
        safe_sp = name_map[sp]
        dy_name = f"d_{safe_sp}"
        dy_names.append(dy_name)
        terms = species_derivatives[sp]
        if not terms:
            eq = "0.0"
        else:
            eq = " ".join(terms).replace("+ -", "- ").replace("- -", "+ ")
            if eq.startswith("+ "): eq = eq[2:]
        lines.append(f"    {dy_name} = {eq}")

    lines.append("")
    
    # Return Statement (Strictly typed as float64)
    lines.append(f"    return np.array([{', '.join(dy_names)}], dtype=np.float64)\n")

    # 3. The Standard Python Wrapper
    lines.append(f"def {func_name}(t: float, y: np.ndarray, k: np.ndarray, dim=False):")
    lines.append(f"    if dim:")
    lines.append(f"        observed = ['CH4', 'O2', 'H2O', 'CO', 'CO2', 'H2', 'H', 'O', 'OH', 'HO2', 'H2O2', 'CH3', 'CH2O', 'HCO']")
    lines.append(f"        inputs = ['feed_CH4', 'feed_O2', 'feed_N2', 'feed_H2O', 'Tin', 'pressure', 'residence_time', 'dilution']")
    lines.append(f"        return {n_spec}, {n_rxns}, {species_names}, observed, inputs, '{yaml_file}'")
    lines.append(f"    return _math_core(t, y, k)\n")

    with open(output_file, 'w') as f:
        f.write("\n".join(lines))
    print(f"Successfully generated {output_file}!")

if __name__ == "__main__":
    generate_explicit_model('datasets/methane/aramco_30.yaml', 'aramco_full_model.py')