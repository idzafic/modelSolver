import re
import numpy as np
import xml.etree.ElementTree as ET
import json
import argparse 
import os      
import sys  
import random     

def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="Converts a MATPOWER .m case file to a dTwin .dmodl model file.",
        formatter_class=argparse.RawTextHelpFormatter  # Preserves formatting in help text
    )
    
    # Required input: MATPOWER .m file
    parser.add_argument(
        "matpower_file",
        help="Path to the input MATPOWER .m case file."
    )
    
    # Optional output path for .dmodl file
    parser.add_argument(
        "-o", "--output",
        help="Path for the output .dmodl file. \nIf not specified, it will be created in the same directory as the input file with the same name (e.g., 'case9.m' -> 'case9.dmodl')."
    )

    # Optional output folder for .dmodl file
    parser.add_argument(
        "-r", "--resPath",
        help="Path for the output .dmodl file. \nIf not specified, it will be created in the same directory as the input file with the same name (e.g., 'case9.m' -> 'case9.dmodl')."
    )
    
    args = parser.parse_args()

    # Extract input and config file paths from arguments
    matpower_input_path = r"cases\{}".format(args.matpower_file)
    config_file_path = "config.xml"           # Static path to XML config
    greek_symbols_path = "greek_symbols.json" # Static path to Greek symbols map


    # Determine the output file path
    if args.output:
        dmodl_output_path = args.output
    else:
        # If no output path is given, use the same name as input with .dmodl extension
        base_name = os.path.basename(matpower_input_path)
        file_name_without_ext = os.path.splitext(base_name)[0]
        if args.resPath:
            dmodl_output_path = args.resPath + "/" + file_name_without_ext 
        else:
            dmodl_output_path = f"{file_name_without_ext}" + "_SE"

    # Print progress info
    print(f"Starting conversion...")
    print(f"  > Input MATPOWER file: {matpower_input_path}")


    #   Error handling for file not found
    try:
        # Parsing configuration file for user options and variable naming
        tree = ET.parse(config_file_path)
        rootNode = tree.getroot()
        root = rootNode.find('powerFlow')
        rootES = rootNode.find('stateEstimation')
        rootCommon = rootNode.find('common')
        # Loading map of Greek symbols (used for variable name formatting)
        with open(greek_symbols_path, 'r', encoding='utf-8') as f:
            greek_map = json.load(f)
    except FileNotFoundError as e:
        print(f"\nError: A required file was not found.")
        print(f"Details: {e}")
        print("Please make sure the paths for the input, config, and greek symbols files are correct.")
        sys.exit(1) # Exit with an error code

    # Function to resolve variable name formatting
    def resolve_variable(elem):
        name = elem.attrib['name']
        fmt = elem.attrib.get('format', 'name')
        if fmt == 'symbol':
            if name in greek_map:
                return greek_map[name]
            else:
                print(f"Error: Symbol '{name}' not found in greek_map.")
                sys.exit(1)
        return name

    # Extracting relevant variable names from XML config
    variables = root.find('variables')
    # --- Polar Coordinate Variables ---
    V_mag   = resolve_variable(variables.find('voltage_magnitude'))
    V_angle = resolve_variable(variables.find('voltage_angle'))
    Y_mag   = resolve_variable(variables.find('line_admittance_magnitude'))
    Y_angle = resolve_variable(variables.find('line_admittance_angle'))

    # --- Rectangular Coordinate Variables ---
    e_var   = resolve_variable(variables.find('real_voltage_component'))
    f_var   = resolve_variable(variables.find('imaginary_voltage_component'))
    G_var   = resolve_variable(variables.find('conductance'))
    B_var   = resolve_variable(variables.find('susceptance'))

    # --- Complex Coordinate Variables ---
    v_cplx  = resolve_variable(variables.find('complex_voltage'))
    Y_cplx  = resolve_variable(variables.find('complex_admittance'))

    # Reading configuration options from XML
    options = root.find('options')
    converter_type_element = options.find('converter_type')
    converter_type = converter_type_element.text.strip().lower() if converter_type_element is not None else 'polar'  # default 'polar' ako nije navedeno
    include_limits = options.find('include_limits').text.strip().lower() == 'true'
    comment_equations = options.find('comment_equations').text.strip().lower() == 'true'
    comment_params = options.find('comment_params').text.strip().lower() == 'true'
    zero_loads = options.find('zero_loads').text.strip().lower() == 'true'
    zip_coeff = options.find('zip_coeff').text.strip().lower() == 'true'
    zip_Kpone = options.find('zip_Kpone').text.strip().lower() == 'true'
    calcQOfPVGensInEachIteration = options.find('calcQOfPVGensInEachIteration').text.strip().lower() == 'true'
    useSumOfCurrentsForZI = options.find('useSumOfCurrentsForZI').text.strip().lower() == 'true'
    convertLoadsToImpedance = options.find('convertLoadsToImpedance').text.strip().lower() == 'true'
    
    #print(f"  > Converter type set to: {converter_type}")

    # Reading power limits for bus categorization (if any)
    limits = root.find('limits')
    power_limits = {}

    if limits is not None:
        last_value = -float('inf')  # Initialize to the smallest possible value
        for category in limits.findall('category'):
            group_name = category.attrib.get('name', '').strip()
            max_value_raw = category.attrib.get('max', 'inf')
            # Try to convert max value to float (handles 'inf' as well)
            try:
                max_value = float('inf') if max_value_raw.strip().lower() == 'inf' else float(max_value_raw)
            except ValueError:
                print(f"Warning: Invalid max value in group '{group_name}': {max_value_raw}")
                continue
            # Ensure max values are strictly increasing
            if max_value <= last_value:
                print(f"Error: Non-increasing max value for category '{group_name}' (value: {max_value})")
                sys.exit(1)
            last_value = max_value
            power_limits[group_name] = max_value  # Store the valid limit


    # Parsing ZIP model parameters
    zip_limits = root.find('zip_limits')
    zip_limits_data = {}

    if zip_limits is not None:
        last_value = -float('inf')  # Initialize to lowest possible to ensure increasing order
        for category in zip_limits.findall('category'):
            group_name = category.attrib.get('name', '').strip()
            max_raw = category.attrib.get('max', 'inf')
            # Parse and validate the max value
            try:
                max_val = float('inf') if max_raw.strip().lower() == 'inf' else float(max_raw)
            except ValueError:
                print(f"Warning: Invalid max value in '{group_name}': {max_raw}")
                continue

            if max_val <= last_value:
                print(f"Error: Non-increasing max value for ZIP category '{group_name}' (value: {max_val})")
                sys.exit(1)
            last_value = max_val
            # Extract ZIP coefficients
            try:
                Kz = float(category.attrib.get('Kz', 0.0))
                Ki = float(category.attrib.get('Ki', 0.0))
                Kp = float(category.attrib.get('Kp', 1.0))
            except ValueError:
                print(f"Warning: Invalid ZIP coefficient in group '{group_name}'")
                continue
            # Store ZIP category data
            zip_limits_data[group_name] = {'max': max_val, 'Kz': Kz, 'Ki': Ki, 'Kp': Kp}

    # Checking sum of ZIP coefficients
    if zip_coeff:
        for name, vals in zip_limits_data.items():
            total = vals['Kz'] + vals['Ki'] + vals['Kp']
            if abs(total - 1.0) > 1e-6:
                print(f"Error: Kz + Ki + Kp for '{name}' is not 1 (got: {total})")
                sys.exit(1)

    # Parsing state estimation from config file
    optionsES = rootES.find('options')
    estimation_method = optionsES.find('methodOfEstimation').text
    if estimation_method!="NE" and  estimation_method!="EC":
        print("Warning: invalid method.")
    voltage_meas = float(optionsES.find('voltageMeasurements').text)
    branch_meas= float(optionsES.find('branchMeasurements').text)
    inj_meas = float(optionsES.find('injectionMeasurements').text)

    weightsES = rootES.find('weightFactors')
    inj_weight = float(weightsES.find('injectionWeight').text)
    branch_weight = float(weightsES.find('branchWeight').text)
    voltage_weight = float(weightsES.find('voltageWeight').text)
    zero_inj_weight = float(weightsES.find('zeroInjectionWeight').text)

    deviationsES = rootES.find('deviations')
    inj_deviation = float(deviationsES.find('injectionDeviation').text)
    branch_deviation = float(deviationsES.find('branchDeviation').text)
    voltage_deviation = float(deviationsES.find('voltageDeviation').text)

    # Parsing common from config file
    includeConsumptionCurves = rootCommon.find('includeConsumptionCurves').text.strip().lower() == 'true'
    numberOfLoadConsumptionCurves = int(rootCommon.find('numberOfLoadConsumptionCurves').text)
    numberOfGenConsumptionCurves = int(rootCommon.find('numberOfGenConsumptionCurves').text)

    eps=1e-14
    number_pattern = re.compile(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?')

    def parse_matrix_block(lines, start_index):
        data = []
        i = start_index + 1 # Skip matrix header line
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('%') or line == '':
                i += 1
                continue
            if '];' in line:
                break
            row = line.split('%')[0].strip()
            matches = number_pattern.findall(row)
            if matches:
                data.append([float(num) for num in matches])
            i += 1
        return data, i

    # Initialize containers
    bus, gen, branch = [], [], []
    version = None
    baseMVA = None

    # Read from the path provided by the command line
    try:
        with open(matpower_input_path, 'r') as file_handle:
            lines = file_handle.readlines()
    except FileNotFoundError:
        print(f"\nError: The MATPOWER input file '{matpower_input_path}' was not found.")
        sys.exit(1)

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Parse version string
        if line.startswith("mpc.version"):
            match = re.search(r"'(.*?)'", line)
            if match:
                version = match.group(1)
            i += 1
        # Parse baseMVA value
        elif line.startswith("mpc.baseMVA"):
            match = re.search(r"=\s*([0-9.]+);", line)
            if match:
                baseMVA = float(match.group(1))
            i += 1
        # Parse bus, gen, and branch matrices
        elif line.startswith('mpc.bus'):
            bus, i = parse_matrix_block(lines, i)
        elif line.startswith('mpc.gen ='):
            gen, i = parse_matrix_block(lines, i)
        elif line.startswith('mpc.branch'):
            branch, i = parse_matrix_block(lines, i)
        elif bus and gen and branch:
            break
        else:
            i += 1
            
    # Map bus indices for easier access
    n = len(bus)
    bus_id_map = {int(bus[i][0]): i for i in range(n)}
    index_to_bus_id = {v: k for k, v in bus_id_map.items()}
    pq_nodes, pv_nodes, slack = [], [], []

    # Assign node type
    for i in range(n):
        bus_type = bus[i][1]
        bus_id = int(bus[i][0])
        if bus_type == 1:
            pq_nodes.append(bus_id)
        elif bus_type == 2:
            pv_nodes.append(bus_id)
        elif bus_type == 3:
            slack.append(bus_id)

    # Organizing generators and branches by bus_id
    gen_by_bus = {}
    for g in gen:
        bus_index = int(g[0])
        if bus_index not in gen_by_bus:
            gen_by_bus[bus_index] = []
        gen_by_bus[bus_index].append(g)

    branch_by_bus = {}
    for b in branch:
        from_bus = int(b[0])
        to_bus = int(b[1])
        if from_bus not in branch_by_bus:
            branch_by_bus[from_bus] = []
        branch_by_bus[from_bus].append(b)
        if to_bus not in branch_by_bus:
            branch_by_bus[to_bus] = []
        branch_by_bus[to_bus].append(b)

    # Initialize the admittance matrix Y with zeros, shape (n x n), complex type
    Y = np.zeros((n, n), dtype=complex)

    # Create a set to track transformer connections
    transformer_set = set()

    # Loop over each row in the branch data to construct the admittance matrix
    for row in branch:
        f_bus = int(row[0])  # From bus
        t_bus = int(row[1])  # To bus
        r, x, b = row[2], row[3], row[4]  # Line resistance, reactance, and shunt susceptance
        ratio = row[8] if row[8] != 0 else 1.0  # Tap ratio; default to 1 if zero
        angle_deg = row[9]  # Phase shift angle in degrees
        angle_rad = np.deg2rad(angle_deg)
        a = ratio * np.exp(1j * angle_rad)  # Complex tap ratio for transformer

        # Calculate series admittance and shunt admittance
        y = 1 / complex(r, x) if r != 0 or x != 0 else 0
        b_shunt = complex(0, b / 2)

        # Map from bus IDs to matrix indices
        f_idx = bus_id_map[f_bus]
        t_idx = bus_id_map[t_bus]

        # Identify transformer branches (with tap or phase shift)
        if ratio != 0 and ratio != 1.0 or angle_deg != 0.0:
            transformer_set.add((f_idx, t_idx))
            transformer_set.add((t_idx, f_idx))

        # Update admittance matrix using transformer model
        Y[f_idx][f_idx] += (y + b_shunt) / (a * np.conj(a))
        Y[t_idx][t_idx] += y + b_shunt
        Y[f_idx][t_idx] -= y / np.conj(a)
        Y[t_idx][f_idx] -= y / a

    G = Y.real  # Conductance matrix
    B = Y.imag  # Susceptance matrix

    num_to_select = int(len(branch) * branch_meas)
    random_branches = random.sample(branch, num_to_select)

    # Generating random voltage measurements list
    n = len(index_to_bus_id)
    num_to_select = int(n * voltage_meas)
    random_voltage = sorted(random.sample(range(1, n + 1), num_to_select))

    # Generating random branches for measurement and randon direction
    for b_row in random_branches:
        original_f_bus = int(b_row[0])
        original_t_bus = int(b_row[1])
        # For each row, make a random 50/50 choice to swap them or not
        if random.choice([True, False]):
            # Keep the original order
            b_row[0] = original_f_bus
            b_row[1] = original_t_bus
        else:
            # Swap the order
            b_row[0] = original_t_bus
            b_row[1] = original_f_bus

    # Add shunt conductance (gs) and susceptance (bs) from the bus data to diagonal elements of Y
    for row in bus:
        i_idx = bus_id_map[int(row[0])]
        gs = row[4] / baseMVA
        bs = row[5] / baseMVA
        Y[i_idx][i_idx] += complex(gs, bs)

    # Set slack bus voltage magnitude and angle from generator data or default to bus data
    for real_bus_id in slack:
        row = bus[bus_id_map[real_bus_id]]
        v_slack = gen_by_bus[real_bus_id][0][5] if real_bus_id in gen_by_bus else row[7]
        Va_deg = row[8]
        v_angle = np.deg2rad(Va_deg)
    
    # Begin writing the dTwin .dmodl file
    with open(dmodl_output_path + ".dmodl", "w", encoding='utf-8') as file:
        # Write model header
        file.write("""Header:\n\tmaxIter=1000\n\treport=AllDetails\t//Solved - only final solved solution, All - shows solved and nonSolved with iterations, AllDetails - All + debug information\n\tmaxReps = -1\n\toutToTxt = false\nend\n""")
        file.write("//Generated by MATPOWER to dmodl coverter\n")
        if converter_type=="complex":
            file.write(f"Model [type=WLS reInit=true eps=1e-5 maxIter=50 domain=cmplx name=\"SE With PF SubModel in complex domain\"]:\n")
        else:
            file.write(f"Model [type=WLS reInit=true eps=1e-5 maxIter=50 domain=real name=\"SE With PF SubModel in real domain\"]:\n")

        # Declare variables for voltage angles and magnitudes (except slack bus)
        file.write("Vars [out=true]:\n")
        for bus_id in range(1, n + 1):
            real_bus_id = index_to_bus_id[bus_id - 1]
            if real_bus_id not in slack:
                row = bus[bus_id_map[real_bus_id]]
                Vm = row[7]
                Va_deg = row[8]
                Va_rad = np.deg2rad(Va_deg)
                # Setting voltage angles and magnitudes according to bus data
                #file.write(f"\t{V_angle}_{real_bus_id} = {Va_rad}; ")
                #file.write(f"{V_mag}_{real_bus_id} = {Vm}\n")

                # Setting voltage angles and magnitudes according to initial values of 0 and 1 respectively
                #file.write(f"\t{V_angle}_{real_bus_id} = {0}; ")
                #file.write(f"{V_mag}_{real_bus_id} = {1}\n")

                # Setting voltage angles and magnitudes according to slack values
            for real_bus_id1 in slack:
                if converter_type == "polar":
                    file.write(f"\t{V_angle}_{real_bus_id} = {V_angle}_{real_bus_id1}_sl; ")
                    file.write(f"{V_mag}_{real_bus_id} = {V_mag}_{real_bus_id1}_sl\n")
                elif converter_type == "rectangular":
                    file.write(f"\t{e_var}_{real_bus_id} = {e_var}_{real_bus_id1}_sl; ")
                    file.write(f"{f_var}_{real_bus_id} = {f_var}_{real_bus_id1}_sl\n")
                elif converter_type == "complex":
                    file.write(f"\t{v_cplx}_{real_bus_id} = {v_cplx}_{real_bus_id1}_sl\n")


        # Write ZIP model coefficients as parameters if enabled
        file.write("Params:\n")
        if zip_coeff:
            if zip_Kpone:
                # If enabled, override all ZIP coefficients to make model purely constant power
                for data in zip_limits_data.values():
                    data['Kz'] = 0.0
                    data['Ki'] = 0.0
                    data['Kp'] = 1.0

            for group_name, data in zip_limits_data.items():
                file.write(f"\tKz_{group_name} = {data['Kz']}; Ki_{group_name} = {data['Ki']}; Kp_{group_name} = {data['Kp']} \n")

        # Declare slack bus variables 
        for real_bus_id in slack:
            row = bus[bus_id_map[real_bus_id]]
            v_slack = gen_by_bus[real_bus_id][0][5] if real_bus_id in gen_by_bus else row[7]
            Va_deg = row[8]
            v_angle = np.deg2rad(Va_deg)
            if converter_type == "polar":
                file.write(f"\t{V_angle}_{real_bus_id}_sl = {v_angle} [out=true]; ")
                file.write(f"\t{V_mag}_{real_bus_id}_sl = {v_slack} [out=true]\n")
            elif converter_type == "rectangular":
                file.write(f"\t{e_var}_{real_bus_id}_sl = {v_slack*np.cos(v_angle)} [out=true]; ")
                file.write(f"\t{f_var}_{real_bus_id}_sl = {v_slack*np.sin(v_angle)} [out=true]\n")
            elif converter_type == "complex":
                file.write(f"\t{v_cplx}_{real_bus_id}_sl = {v_slack}*e^(1i*{v_angle}) [out=true]\n")
        
        # Compute power injections from generators (Pg, Qg)
        P_inj = np.zeros(n)
        Q_inj = np.zeros(n)
        for bus_id, gens in gen_by_bus.items():
            for g in gens:
                bus_idx = bus_id_map[bus_id]
                Pg = g[1] / baseMVA
                Qg = g[2] / baseMVA
                P_inj[bus_idx] += Pg
                Q_inj[bus_idx] += Qg # should not change anything --> always zero

        # Compute power demands and subtract from injections
        Pd = np.array([row[2] for row in bus]) / baseMVA
        Qd = np.array([row[3] for row in bus]) / baseMVA
        if zero_loads:          # If enabled, set load values to zero for testing
            Pd = np.zeros(n)
            Qd = np.zeros(n)

        # Apply the loads based on the chosen model
        if convertLoadsToImpedance:
            Vm = np.array([row[7] for row in bus])
            # Add the equivalent admittance (P - jQ) to the Ybus diagonal
            for i_idx in range(n):
                 if Vm[i_idx] > 1e-6:
                    shunt_admittance = (Pd[i_idx] - 1j * Qd[i_idx]) / (Vm[i_idx]**2)
                    Y[i_idx, i_idx] += shunt_admittance
        else: 
            # Subtract the power demands from the generator injections to get net power
            for i_idx in range(n):
                P_inj[i_idx] -= Pd[i_idx]
                Q_inj[i_idx] -= Qd[i_idx]

        # Generating random injection nodes
        eligible_injection_nodes = []
        all_original_nodes = pv_nodes + pq_nodes
        for bus_id in all_original_nodes:
            i_idx = bus_id_map[bus_id]
            # Checking for ZI node
            if not (P_inj[i_idx] == 0 and Q_inj[i_idx] == 0):
                eligible_injection_nodes.append(bus_id)
        num_to_select = int(len(eligible_injection_nodes) * inj_meas)
        selected_nodes = random.sample(eligible_injection_nodes, num_to_select)
        pv_nodes_set = set(pv_nodes)
        random_pv_nodes = [bus_id for bus_id in selected_nodes if bus_id in pv_nodes_set]
        random_pq_nodes = [bus_id for bus_id in selected_nodes if bus_id not in pv_nodes_set]
        random_pv_nodes.sort()
        random_pq_nodes.sort()

        all_pq_nodes_with_load = []
        for bus_id in pq_nodes:
            i_idx = bus_id_map[bus_id]
            if P_inj[i_idx] !=0 or Q_inj[i_idx] != 0:
                all_pq_nodes_with_load.append(bus_id)
        if numberOfLoadConsumptionCurves > len(all_pq_nodes_with_load): 
            print(f"\nError: Number Of Load Consumption Curves ({numberOfLoadConsumptionCurves}) is larger than the number of available loads ({len(all_pq_nodes_with_load)}).")
            sys.exit(1)
        selected_pq_nodes = random.sample(all_pq_nodes_with_load, int(numberOfLoadConsumptionCurves))
        selected_pq_nodes.sort()

        if numberOfGenConsumptionCurves > len(pv_nodes): 
            print(f"\nError: Number Of Generator Consumption Curves ({numberOfGenConsumptionCurves}) is larger than the number of available loads ({len(pv_nodes)}).")
            sys.exit(1)
        if numberOfGenConsumptionCurves < 0 or numberOfLoadConsumptionCurves < 0: 
            print(f"\nError: Number of curves must be positive.")
            sys.exit(1)
        selected_pv_nodes = random.sample(pv_nodes, int(numberOfGenConsumptionCurves))
        selected_pv_nodes.sort()

        # Write admittance matrix entries as parameters 
        for i_idx in range(n):
            for j_idx in range(n):
                bus_i = index_to_bus_id[i_idx]
                bus_j = index_to_bus_id[j_idx]
                magnitude = abs(Y[i_idx, j_idx])
                angle_rad = np.angle(Y[i_idx, j_idx])

                if magnitude != 0: # write only if not zero
                    if converter_type == "polar":
                        file.write(f"\t{Y_mag}_{bus_i}_{bus_j} = {magnitude}; ")
                        if angle_rad != 0:
                            file.write(f"{Y_angle}_{bus_i}_{bus_j} = {angle_rad} ")
                    elif converter_type == "rectangular":
                        G_val = G[i_idx, j_idx]
                        B_val = B[i_idx, j_idx]
                        file.write(f"\t{G_var}_{bus_i}_{bus_j} = {0 if abs(G_val) < eps else G_val}; ")
                        file.write(f"{B_var}_{bus_i}_{bus_j} = {0 if abs(B_val) < eps else B_val} ")
                    elif converter_type == "complex":
                        file.write(f"\t{Y_cplx}_{bus_i}_{bus_j} = ")
                        if Y[i_idx, j_idx].real != 0:
                            file.write(f"{Y[i_idx, j_idx].real} ")
                            if Y[i_idx, j_idx].real != 0 and Y[i_idx, j_idx].imag > 0: 
                                file.write(f"+ ")
                        if Y[i_idx, j_idx].imag != 0:
                            file.write(f"{Y[i_idx, j_idx].imag}i ")
                    # Add comments to lines and transformers
                    if comment_params:
                        if bus_i != bus_j:
                            if (i_idx, j_idx) in transformer_set:
                                file.write(f"// transformer {bus_i}-{bus_j}\n")
                            else:
                                file.write(f"// line {bus_i}-{bus_j}\n")
                        else:
                            file.write(f"\n")
                    else:
                        file.write(f"\n")

        # Write active and reactive injections (P_inj, Q_inj) to file
        for i_idx in range(n):
            bus_id = index_to_bus_id[i_idx]
            if converter_type == "complex":
                # Write S for PQ nodes
                if bus_id in pq_nodes:
                    if P_inj[i_idx] != 0:
                        file.write(f"\tS{bus_id}_inj = {P_inj[i_idx]}")
                        if Q_inj[i_idx] > 0:
                            file.write(f" + {Q_inj[i_idx]}i")
                        elif Q_inj[i_idx] < 0:
                            file.write(f" {Q_inj[i_idx]}i")
                    if P_inj[i_idx] == 0 and Q_inj[i_idx] != 0:
                        file.write(f"\tS{bus_id}_inj = {Q_inj[i_idx]}i\n")
                    elif P_inj[i_idx] != 0:
                        file.write(f" \n")
                # Write P and Q for PV nodes
                if bus_id in pv_nodes:
                    if P_inj[i_idx] != 0:
                        file.write(f"\tP{bus_id}_inj = {P_inj[i_idx]}\n")
                    if not calcQOfPVGensInEachIteration:
                        file.write(f"\tQ{bus_id}_inj [out = true]\n")
                    else: 
                        file.write(f"\tQ{bus_id}_inj = {Q_inj[i_idx]} [out = true]\n")      
            elif converter_type == "polar" or converter_type == "rectangular":
                if P_inj[i_idx] != 0:
                    file.write(f"\tP{bus_id}_inj = {P_inj[i_idx]}\n")
                if Q_inj[i_idx] != 0 and bus_id not in pv_nodes:
                    file.write(f"\tQ{bus_id}_inj = {Q_inj[i_idx]}\n")
                if bus_id in pv_nodes:
                    if not calcQOfPVGensInEachIteration:
                        file.write(f"\tQ{bus_id}_inj [out = true]\n")
                    else: 
                        file.write(f"\tQ{bus_id}_inj = {Q_inj[i_idx]} [out = true]\n")

        # If enabled, write PV node controls (voltage magnitude setpoint, limits) to file
        for pv_bus in pv_nodes:
            if pv_bus in gen_by_bus:
                g = gen_by_bus[pv_bus][0]
                if include_limits:
                    file.write(f"\tcGen{pv_bus}Reg=true\n")
                    file.write(f"\tQ{pv_bus}_inj_min = {g[4]/baseMVA}\n")
                    file.write(f"\tQ{pv_bus}_inj_max = {g[3]/baseMVA}\n")
                file.write(f"\tV_{pv_bus}_sp = {g[5]}\n")

        # Adding random branches
        for bus_id in random_branches:
            from_bus=bus_id[0]
            to_bus=bus_id[1]
            if converter_type == "complex":
                file.write(f"\tS_{int(from_bus)}_{int(to_bus)}_meas [out = true]\n")
                file.write(f"\tS_{int(from_bus)}_{int(to_bus)}_est [out = true]\n")
            else:
                file.write(f"\tP_{int(from_bus)}_{int(to_bus)}_meas [out = true]; ")
                file.write(f"P_{int(from_bus)}_{int(to_bus)}_est [out = true]\n")
                file.write(f"\tQ_{int(from_bus)}_{int(to_bus)}_meas [out = true]; ")
                file.write(f"Q_{int(from_bus)}_{int(to_bus)}_est [out = true]\n")
        
        # Adding measuring params
        for bus_id in range(1, n + 1):
            real_bus_id = index_to_bus_id[bus_id - 1]
            #if real_bus_id not in slack:
            row = bus[bus_id_map[real_bus_id]]
            if converter_type == "complex":
                file.write(f"\t{v_cplx}_{real_bus_id}_meas [type=real out=true]\n")
            else:
                file.write(f"\t{V_mag}_{real_bus_id}_meas [out=true]\n")
        if converter_type == "complex":
            file.write(f"\tw_inj = {inj_weight} [type=real]\n")
            file.write(f"\tw_br = {branch_weight} [type=real]\n")
            file.write(f"\tw_v = {voltage_weight} [type=real]\n")
            file.write(f"\tw_zi = {zero_inj_weight} [type=real]\n")
        else:
            file.write(f"\tw_inj = {inj_weight}\n")
            file.write(f"\tw_br = {branch_weight}\n")
            file.write(f"\tw_v = {voltage_weight}\n")
            file.write(f"\tw_zi = {zero_inj_weight}\n")
        for i_idx in range(n):
            bus_id = index_to_bus_id[i_idx]
            if converter_type == "complex":
                # Write S for PQ nodes
                if bus_id in pq_nodes:
                    if P_inj[i_idx] != 0 or Q_inj[i_idx] != 0:
                        file.write(f"\tS{bus_id}_meas [out = true]\n\tS{bus_id}_est [out = true]\n")
            elif converter_type == "polar" or converter_type == "rectangular":
                if bus_id in pq_nodes:
                    if P_inj[i_idx] != 0 or Q_inj[i_idx] != 0:
                        file.write(f"\tP{bus_id}_meas [out = true]; P{bus_id}_est [out = true]\n\tQ{bus_id}_meas [out = true]; Q{bus_id}_est [out = true]\n")
            # Write P and Q for PV nodes
            if bus_id in pv_nodes:
                if P_inj[i_idx] != 0:
                    file.write(f"\tP{bus_id}_meas [out = true]; P{bus_id}_est [out = true]\n\tQ{bus_id}_est [out = true]\n")
        if includeConsumptionCurves:
            for bus_id in selected_pq_nodes:
                i_idx = bus_id_map[bus_id]
                if converter_type == "complex":
                    file.write(f"\tk{bus_id}_load = 1 [type=real]\n")
                elif converter_type == "polar" or converter_type == "rectangular":    
                    file.write(f"\tk{bus_id}_load = 1\n")
            for bus_id in selected_pv_nodes:
                i_idx = bus_id_map[bus_id]
                if converter_type == "complex":
                    file.write(f"\tk{bus_id}_gen = 1 [type=real]\n")
                elif converter_type == "polar" or converter_type == "rectangular":    
                    file.write(f"\tk{bus_id}_gen = 1\n")
        # SUBMODEL ----------------------------------------------> Writing submodel in model 
        if converter_type=="complex":
            file.write("SubModel [type=NL alwaysOn=true eps=1e-4 maxIter=200 domain=cmplx copyPars=3000 alwaysOn=false name=\"Solving power flow\"]:\n")
            file.write("\tVars [conj=true out=true]:\n")
        else:
            file.write("SubModel [type=NL alwaysOn=true eps=1e-4 maxIter=200 domain=real copyPars=3000 alwaysOn=false name=\"Solving power flow\"]:\n")
            file.write("\tVars [out=true]:\n")
        for bus_id in range(1, n + 1):
            real_bus_id = index_to_bus_id[bus_id - 1]
            if real_bus_id not in slack:
                for real_bus_id1 in slack:
                    if converter_type == "polar":
                        file.write(f"\t\t{V_angle}_{real_bus_id} = {V_angle}_{real_bus_id1}_sl; ")
                        file.write(f"{V_mag}_{real_bus_id} = {V_mag}_{real_bus_id1}_sl\n")
                    elif converter_type == "rectangular":
                        file.write(f"\t\t{e_var}_{real_bus_id} = {e_var}_{real_bus_id1}_sl; ")
                        file.write(f"{f_var}_{real_bus_id} = {f_var}_{real_bus_id1}_sl\n")
                    elif converter_type == "complex":
                        file.write(f"\t\t{v_cplx}_{real_bus_id} = {v_cplx}_{real_bus_id1}_sl\n")
        file.write("\tParams:\n")
        # Declare slack bus variables 
        for real_bus_id in slack:
            row = bus[bus_id_map[real_bus_id]]
            v_slack = gen_by_bus[real_bus_id][0][5] if real_bus_id in gen_by_bus else row[7]
            Va_deg = row[8]
            v_angle = np.deg2rad(Va_deg)
            for real_bus_id1 in slack:
                if converter_type == "polar":
                    file.write(f"\t\t{V_angle}_{real_bus_id} = {v_angle} [out=true]; ")
                    file.write(f"\t{V_mag}_{real_bus_id} = {v_slack} [out=true]\n")
                elif converter_type == "rectangular":
                    file.write(f"\t\t{e_var}_{real_bus_id} = {v_slack*np.cos(v_angle)} [out=true]; ")
                    file.write(f"\t{f_var}_{real_bus_id} = {v_slack*np.sin(v_angle)} [out=true]\n")
                elif converter_type == "complex":
                    file.write(f"\t\t{v_cplx}_{real_bus_id} = {v_cplx}_{real_bus_id1}_sl\n")
        file.write("\tDistribs:\n")
        file.write(f"\t\tg_inj [type=Gauss mean=0 dev={inj_deviation}]\n")
        file.write(f"\t\tg_br [type=Gauss mean=0 dev={branch_deviation}]\n")
        file.write(f"\t\tg_v [type=Gauss mean=0 dev={voltage_deviation}]\n")

        # Begin writing the nonlinear equations (NLEs) section
        file.write("\tNLEs:\n")
        S_mag_by_bus = {}

        # Loop over all PQ nodes to write power balance equations
        for bus_id in pq_nodes:
            if comment_equations:
                file.write(f"\t\t// node {bus_id} - PQ\n")
            i_idx = bus_id_map[bus_id]

            # Calculate real and reactive power in system units (MW, MVar)
            p_val = P_inj[i_idx]*baseMVA
            q_val = Q_inj[i_idx]*baseMVA
            s_magnitude = np.sqrt(p_val**2 + q_val**2)  # Apparent power magnitude

            # Write real power balance equation for PQ node
            if converter_type == "polar":
                file.write("\t\t") # Start the line             
                if useSumOfCurrentsForZI and P_inj[i_idx]==0 and Q_inj[i_idx]==0:
                    # This formulation comes from P_i = Re(V_i * I_i_conj), where I_i is the sum of currents
                    # P_i = V_mag_i * (cos(V_angle_i) * Real(I_i) + sin(V_angle_i) * Imag(I_i))
                    # Real(I_i) = Sum_j [Y_mag_ij * V_mag_j * cos(V_angle_j + Y_angle_ij)]
                    # Imag(I_i) = Sum_j [Y_mag_ij * V_mag_j * sin(V_angle_j + Y_angle_ij)]
                    real_current_terms = []
                    for j_idx in range(n):
                        j_bus_id = index_to_bus_id[j_idx]
                        aY = abs(Y[i_idx][j_idx])
                        if aY == 0:
                            continue
                        theta = np.angle(Y[i_idx][j_idx])
                        theta_term = f" + {Y_angle}_{bus_id}_{j_bus_id}" if theta != 0 else ""
                        real_term = f"{Y_mag}_{bus_id}_{j_bus_id} * {V_mag}_{j_bus_id} * cos({V_angle}_{j_bus_id}{theta_term})"
                        real_current_terms.append(real_term)
                    real_sum = " + ".join(real_current_terms)
                    file.write(f"{real_sum} ")
                else:
                    # Sum of Powers formulation
                    file.write(f"{V_mag}_{bus_id} * (")
                    terms = []
                    for j_idx in range(n):
                        j_bus_id = index_to_bus_id[j_idx]
                        aY = abs(Y[i_idx][j_idx])
                        if aY == 0:
                            continue
                        theta = np.angle(Y[i_idx][j_idx])
                        theta_term = f" - {Y_angle}_{bus_id}_{j_bus_id}" if theta != 0 else ""
                        term = f"{Y_mag}_{bus_id}_{j_bus_id} * {V_mag}_{j_bus_id} * cos({V_angle}_{bus_id}{theta_term} - {V_angle}_{j_bus_id})"
                        terms.append(term)
                    file.write(" + ".join(terms))
                    file.write(") ")
            elif converter_type == "rectangular":
                if useSumOfCurrentsForZI and P_inj[i_idx]==0 and Q_inj[i_idx]==0: # sum of currents
                    file.write(f"\t\t")
                    real_terms = []
                    for j_idx in range(n):
                        j_bus_id = index_to_bus_id[j_idx]
                        if abs(G[i_idx][j_idx]) < eps and abs(B[i_idx][j_idx]) < eps:
                            continue
                        term = f"{G_var}_{bus_id}_{j_bus_id} * {e_var}_{j_bus_id} - {B_var}_{bus_id}_{j_bus_id} * {f_var}_{j_bus_id}"
                        real_terms.append(term)
                    file.write(" + ".join(real_terms))
                else: # powers
                    file.write(f"\t\t")  
                    terms = []
                    for j_idx in range(n):
                        j_bus_id = index_to_bus_id[j_idx]
                        if abs(G[i_idx][j_idx]) < eps and abs(B[i_idx][j_idx]) < eps:
                            continue
                        term = f"{e_var}_{bus_id} * ({G_var}_{bus_id}_{j_bus_id} * {e_var}_{j_bus_id} - {B_var}_{bus_id}_{j_bus_id} * {f_var}_{j_bus_id}) + {f_var}_{bus_id} * ({B_var}_{bus_id}_{j_bus_id} * {e_var}_{j_bus_id} + {G_var}_{bus_id}_{j_bus_id} * {f_var}_{j_bus_id})"
                        terms.append(term)
                    file.write(" + ".join(terms))
            elif converter_type == "complex": 
                if useSumOfCurrentsForZI and P_inj[i_idx]==0 and Q_inj[i_idx]==0: #sum of currents
                    terms = []
                    for j_idx in range(n):
                        j_bus_id = index_to_bus_id[j_idx]
                        if abs(Y[i_idx][j_idx]) < eps:
                            continue
                        terms.append(f"{Y_cplx}_{bus_id}_{j_bus_id} * {v_cplx}_{j_bus_id}")
                    current_summation = " + ".join(terms)
                    file.write(f"\t\tconj({current_summation}")
                else: # power
                    file.write(f"\t\t{v_cplx}_{bus_id} * conj(")
                    terms = []
                    for j_idx in range(n):
                        j_bus_id = index_to_bus_id[j_idx]
                        if abs(Y[i_idx][j_idx]) < eps:
                            continue
                        terms.append(f"{Y_cplx}_{bus_id}_{j_bus_id} * {v_cplx}_{j_bus_id}")
                    file.write(" - ".join(terms) if len(terms) == 1 else " + ".join(terms))
                if P_inj[i_idx] == 0 and Q_inj[i_idx] == 0:
                    file.write(") = 0\n")
                else:
                    # Writing complex ZIP if included
                    if zip_coeff:
                        for group_name, data in zip_limits_data.items():
                                if s_magnitude < data['max']:
                                    Vm = np.array([row[7] for row in bus])
                                    if Vm[i_idx] != 1:
                                        file.write(f") = S{bus_id}_inj*(Kz_{group_name}*({v_cplx}_{bus_id} * conj({v_cplx}_{bus_id})/{Vm[i_idx]}^2) + Ki_{group_name}*(sqrt({v_cplx}_{bus_id} * conj({v_cplx}_{bus_id}))/{Vm[i_idx]}) + Kp_{group_name}) \n")
                                    else:
                                        file.write(f") = S{bus_id}_inj*(Kz_{group_name}*({v_cplx}_{bus_id} * conj({v_cplx}_{bus_id})) + Ki_{group_name}*sqrt({v_cplx}_{bus_id} * conj({v_cplx}_{bus_id})) + Kp_{group_name}) \n")
                                    break
                    else:
                        if includeConsumptionCurves and bus_id in selected_pq_nodes:
                            file.write(f") = S{bus_id}_inj * k{bus_id}_load\n")
                        else:
                            file.write(f") = S{bus_id}_inj\n")
            # Handle optional ZIP load model or default power injection form
            if converter_type != "complex":
                if P_inj[i_idx] == 0:
                    file.write(" = 0\n")
                else:
                    if zip_coeff:
                        if converter_type == "polar":
                            for group_name, data in zip_limits_data.items():
                                if s_magnitude < data['max']:
                                    Vm = np.array([row[7] for row in bus])
                                    if Vm[i_idx] != 1:
                                        file.write(f" = P{bus_id}_inj*(Kz_{group_name}*({V_mag}_{bus_id}/{Vm[i_idx]})^2 + Ki_{group_name}*({V_mag}_{bus_id}/{Vm[i_idx]}) + Kp_{group_name}) \n")
                                    else:
                                        file.write(f" = P{bus_id}_inj*(Kz_{group_name}*({V_mag}_{bus_id})^2 + Ki_{group_name}*{V_mag}_{bus_id} + Kp_{group_name}) \n")
                                    break
                        elif converter_type == "rectangular":
                            for group_name, data in zip_limits_data.items():
                                if s_magnitude < data['max']:
                                    Vm = np.array([row[7] for row in bus])
                                    if Vm[i_idx] != 1:
                                        file.write(f" = P{bus_id}_inj*(Kz_{group_name}*(({e_var}_{bus_id}^2+{f_var}_{bus_id}^2)/{Vm[i_idx]}^2) + Ki_{group_name}*(sqrt({e_var}_{bus_id}^2+{f_var}_{bus_id}^2)/{Vm[i_idx]}) + Kp_{group_name}) \n")
                                    else:
                                        file.write(f" = P{bus_id}_inj*(Kz_{group_name}*({e_var}_{bus_id}^2+{f_var}_{bus_id}^2) + Ki_{group_name}*sqrt({e_var}_{bus_id}^2+{f_var}_{bus_id}^2) + Kp_{group_name}) \n")
                                    break
                    else:
                        if includeConsumptionCurves and bus_id in selected_pq_nodes:
                            file.write(f" = P{bus_id}_inj * k{bus_id}_load\n")
                        else:
                            file.write(f" = P{bus_id}_inj\n")

            # Write reactive power balance equation for PQ node or KL1 if injected power is zero
            if converter_type == "polar":
                if useSumOfCurrentsForZI and P_inj[i_idx]==0 and Q_inj[i_idx]==0:
                    # This formulation comes from Q_i = Im(V_i * I_i_conj)
                    # Q_i = V_mag_i * (sin(V_angle_i) * Real(I_i) - cos(V_angle_i) * Imag(I_i))                    
                    imag_current_terms = []
                    for j_idx in range(n):
                        j_bus_id = index_to_bus_id[j_idx]
                        if abs(Y[i_idx][j_idx]) == 0:
                            continue
                        theta = np.angle(Y[i_idx][j_idx])
                        theta_term = f" + {Y_angle}_{bus_id}_{j_bus_id}" if theta != 0 else ""
                        imag_term = f"{Y_mag}_{bus_id}_{j_bus_id} * {V_mag}_{j_bus_id} * sin({V_angle}_{j_bus_id}{theta_term})"
                        imag_current_terms.append(imag_term)
                    imag_sum = " + ".join(imag_current_terms)
                    file.write(f"\t\t{imag_sum} ")
                else:
                    file.write(f"\t\t{V_mag}_{bus_id} * (")
                    terms = []
                    for j_idx in range(n):
                        j_bus_id = index_to_bus_id[j_idx]
                        aY = abs(Y[i_idx][j_idx])
                        if aY == 0:
                            continue
                        theta = np.angle(Y[i_idx][j_idx])
                        theta_term = f" - {Y_angle}_{bus_id}_{j_bus_id}" if theta != 0 else ""
                        term = f"{Y_mag}_{bus_id}_{j_bus_id} * {V_mag}_{j_bus_id} * sin({V_angle}_{bus_id}{theta_term} - {V_angle}_{j_bus_id})"
                        terms.append(term)
                    file.write(" + ".join(terms))
                    file.write(") ")
            elif converter_type == "rectangular":
                if useSumOfCurrentsForZI and P_inj[i_idx]==0 and Q_inj[i_idx]==0:                   
                    file.write(f"\t\t")
                    imag_terms = []
                    for j_idx in range(n):
                        j_bus_id = index_to_bus_id[j_idx]
                        if abs(G[i_idx][j_idx]) < eps and abs(B[i_idx][j_idx]) < eps:
                            continue
                        term = f"{B_var}_{bus_id}_{j_bus_id} * {e_var}_{j_bus_id} + {G_var}_{bus_id}_{j_bus_id} * {f_var}_{j_bus_id}"
                        imag_terms.append(term)
                    file.write(" + ".join(imag_terms))
                else:
                    file.write(f"\t\t") 
                    terms = []
                    for j_idx in range(n):
                        j_bus_id = index_to_bus_id[j_idx]
                        if abs(G[i_idx][j_idx]) < eps and abs(B[i_idx][j_idx]) < eps:
                            continue
                        term = (f"{f_var}_{bus_id} * ({G_var}_{bus_id}_{j_bus_id} * {e_var}_{j_bus_id} - {B_var}_{bus_id}_{j_bus_id} * {f_var}_{j_bus_id}) - {e_var}_{bus_id} * ({B_var}_{bus_id}_{j_bus_id} * {e_var}_{j_bus_id} + {G_var}_{bus_id}_{j_bus_id} * {f_var}_{j_bus_id})")
                        terms.append(term)
                    file.write(" + ".join(terms))
            elif converter_type == "complex": 
                if useSumOfCurrentsForZI and P_inj[i_idx]==0 and Q_inj[i_idx]==0:
                    terms = []
                    for j_idx in range(n):
                        j_bus_id = index_to_bus_id[j_idx]
                        if abs(Y[i_idx][j_idx]) < eps:
                            continue
                        terms.append(f"{Y_cplx}_{bus_id}_{j_bus_id} * {v_cplx}_{j_bus_id}")
                    current_summation = " + ".join(terms)
                    file.write(f"\t\t({current_summation}")
                else:                 
                    v_i = f"{v_cplx}_{bus_id}"
                    conj_vi = f"conj({v_i})"
                    sum_terms = []
                    for j_idx in range(n):
                        j_bus_id = index_to_bus_id[j_idx]
                        if abs(Y[i_idx][j_idx]) < eps:
                            continue
                        v_j = f"{v_cplx}_{j_bus_id}"
                        Y_sym = f"{Y_cplx}_{bus_id}_{j_bus_id}"
                        term = f"{Y_sym} * {v_j}"
                        sum_terms.append(term)

                    rhs = f"conj(S{bus_id}_inj)"
                    file.write(f"\t\t{conj_vi} * (")
                    file.write(" + ".join(sum_terms))
                if P_inj[i_idx] == 0 and Q_inj[i_idx] == 0:
                    rhs = f"0"
                else:
                    if zip_coeff:
                        for group_name, data in zip_limits_data.items():
                                if s_magnitude < data['max']:
                                    Vm = np.array([row[7] for row in bus])
                                    if Vm[i_idx] != 1:
                                        rhs = f" conj(S{bus_id}_inj)*(Kz_{group_name}*({v_cplx}_{bus_id} * conj({v_cplx}_{bus_id})/{Vm[i_idx]}^2) + Ki_{group_name}*(sqrt({v_cplx}_{bus_id} * conj({v_cplx}_{bus_id}))/{Vm[i_idx]}) + Kp_{group_name}) \n"
                                    else:
                                        rhs = f" conj(S{bus_id}_inj)*(Kz_{group_name}*({v_cplx}_{bus_id} * conj({v_cplx}_{bus_id})) + Ki_{group_name}*sqrt({v_cplx}_{bus_id} * conj({v_cplx}_{bus_id})) + Kp_{group_name}) \n"
                                    break
                    else:
                        if includeConsumptionCurves and bus_id in selected_pq_nodes:
                            rhs = f"conj(S{bus_id}_inj * k{bus_id}_load)"
                        else:
                            rhs = f"conj(S{bus_id}_inj)"
                file.write(f") = {rhs}\n")
            # Add Q if not complex domain
            if converter_type != "complex":
                if Q_inj[i_idx] == 0:
                    file.write(" = 0\n")
                else:
                    if zip_coeff:
                        if converter_type == "polar":
                            for group_name, data in zip_limits_data.items():
                                if s_magnitude < data['max']:
                                    Vm = np.array([row[7] for row in bus])
                                    if Vm[i_idx] != 1:
                                        file.write(f" = Q{bus_id}_inj*(Kz_{group_name}*({V_mag}_{bus_id}/{Vm[i_idx]})^2 + Ki_{group_name}*({V_mag}_{bus_id}/{Vm[i_idx]}) + Kp_{group_name}) \n")
                                    else:
                                        file.write(f" = Q{bus_id}_inj*(Kz_{group_name}*({V_mag}_{bus_id})^2 + Ki_{group_name}*{V_mag}_{bus_id} + Kp_{group_name}) \n")
                                    break
                        elif converter_type == "rectangular":
                            for group_name, data in zip_limits_data.items():
                                if s_magnitude < data['max']:
                                    Vm = np.array([row[7] for row in bus])
                                    if Vm[i_idx] != 1:
                                        file.write(f" = Q{bus_id}_inj*(Kz_{group_name}*(({e_var}_{bus_id}^2+{f_var}_{bus_id}^2)/{Vm[i_idx]}^2) + Ki_{group_name}*(sqrt({e_var}_{bus_id}^2+{f_var}_{bus_id}^2)/{Vm[i_idx]}) + Kp_{group_name}) \n")
                                    else:
                                        file.write(f" = Q{bus_id}_inj*(Kz_{group_name}*({e_var}_{bus_id}^2+{f_var}_{bus_id}^2) + Ki_{group_name}*sqrt({e_var}_{bus_id}^2+{f_var}_{bus_id}^2) + Kp_{group_name}) \n")
                                    break
                    else:
                        if includeConsumptionCurves and bus_id in selected_pq_nodes:
                            file.write(f" = Q{bus_id}_inj * k{bus_id}_load\n")
                        else:
                            file.write(f" = Q{bus_id}_inj\n")

        # Loop over all PV nodes to write real power balance and voltage control
        for i in pv_nodes:
            if comment_equations:
                file.write(f"\t\t// node {i} - PV\n\t\t")
            i_idx = bus_id_map[i]

            # Write real power balance equation for PV node
            if converter_type == "polar":
                file.write(f"{V_mag}_{i} * (")
                terms = []
                for j_idx in range(n):
                    j = index_to_bus_id[j_idx]
                    aY = abs(Y[i_idx][j_idx])
                    if aY == 0:
                        continue
                    theta = np.angle(Y[i_idx][j_idx])
                    theta_term = f" - {Y_angle}_{i}_{j}" if theta != 0 else ""
                    term = f"{Y_mag}_{i}_{j} * {V_mag}_{j} * cos({V_angle}_{i}{theta_term} - {V_angle}_{j})"
                    terms.append(term)
                file.write(" + ".join(terms))
                file.write(f") ")
            elif converter_type == "rectangular":
                terms = []
                for j_idx in range(n):
                    j = index_to_bus_id[j_idx]
                    if abs(G[i_idx][j_idx]) < eps and abs(B[i_idx][j_idx]) < eps:
                        continue
                    term = f"{e_var}_{i}*({G_var}_{i}_{j} * {e_var}_{j} - {B_var}_{i}_{j} * {f_var}_{j}) + {f_var}_{i} * ({G_var}_{i}_{j} * {f_var}_{j} + {B_var}_{i}_{j} * {e_var}_{j})"
                    terms.append(term)
                file.write(" + ".join(terms))
            elif converter_type == "complex":
                expr = []
                for j_idx in range(n):
                    j_bus_id = index_to_bus_id[j_idx]
                    if abs(Y[i_idx, j_idx]) > eps:  
                        expr.append(f"{Y_cplx}_{i}_{j_bus_id} * {v_cplx}_{j_bus_id}")
                current_expr = " + ".join(expr)
                file.write(f"{v_cplx}_{i} * conj({current_expr}) + conj({v_cplx}_{i}) * ({current_expr}) ")
                if P_inj[i_idx] == 0:
                    file.write(" = 0\n")
                else:
                    if includeConsumptionCurves and i in selected_pv_nodes:
                        file.write(f" = 2*P{i}_inj * k{i}_gen\n")
                    else:
                        file.write(f" = 2*P{i}_inj \n")
            if converter_type != "complex":
                if P_inj[i_idx] == 0:
                    file.write(" = 0\n")
                else:
                    if includeConsumptionCurves and i in selected_pv_nodes:
                        file.write(f" = P{i}_inj * k{i}_gen\n")
                    else:
                        file.write(f" = P{i}_inj \n")

            # Voltage control for PV node (setpoint enforcement or reactive control under limits)
            if include_limits:
                file.write(f"\tif cGen{i}Reg:\n\t")

            # Voltage equation
            if converter_type == "polar":
                file.write(f"\t\t{V_mag}_{i} = {V_mag}_{i}_sp \n")
            elif converter_type == "rectangular":
                file.write(f"\t\t{e_var}_{i}^2 + {f_var}_{i}^2 = V_{i}_sp^2 \n")
            elif converter_type == "complex":
                file.write(f"\t\t{v_cplx}_{i} * conj({v_cplx}_{i}) = V_{i}_sp^2 \n")

            # Q equation
            if include_limits:
                file.write(f"\telse:\n")
                terms = []
                if converter_type == "polar":
                    file.write(f"\t\t{V_mag}_{i} * (")
                    for j_idx in range(n):
                        j = index_to_bus_id[j_idx]
                        aY = abs(Y[i_idx][j_idx])
                        if aY == 0:
                            continue
                        theta = np.angle(Y[i_idx][j_idx])
                        theta_term = f" - {Y_angle}_{i}_{j}" if theta != 0 else ""
                        term = f"{Y_mag}_{i}_{j} * {V_mag}_{j} * sin({V_angle}_{i}{theta_term} - {V_angle}_{j})"
                        terms.append(term)
                    file.write(" + ".join(terms))
                elif converter_type == "rectangular":
                    file.write(f"\t\t(")
                    terms = []
                    for j_idx in range(n):
                        j_bus_id = index_to_bus_id[j_idx]
                        if abs(G[i_idx][j_idx]) < eps and abs(B[i_idx][j_idx]) < eps:
                            continue
                        term = (f"{f_var}_{i}*({G_var}_{i}_{j_bus_id}*{e_var}_{j_bus_id} - {B_var}_{i}_{j_bus_id}*{f_var}_{j_bus_id}) - "
                                f"{e_var}_{i}*({B_var}_{i}_{j_bus_id}*{e_var}_{j_bus_id} + {G_var}_{i}_{j_bus_id}*{f_var}_{j_bus_id})")
                        terms.append(term)
                    file.write(" + ".join(terms))
                elif converter_type == "complex":
                    expr_inner = []
                    for j_idx in range(n):
                        j_bus_id = index_to_bus_id[j_idx]
                        y_ij = Y[i_idx, j_idx]
                        # Sign always plus, minus accounted for in Y matrix
                        if abs(y_ij) > eps:
                            if j_bus_id == i:
                                expr_inner.append(f"+ {Y_cplx}_{i}_{j_bus_id} * {v_cplx}_{j_bus_id}")
                            else:
                                expr_inner.append(f"+ {Y_cplx}_{i}_{j_bus_id} * {v_cplx}_{j_bus_id}")
                    inner_expr = " ".join(expr_inner)
                    file.write(f"\t\t{v_cplx}_{i} * conj({inner_expr}) - conj({v_cplx}_{i}) * ({inner_expr}) ")
                    if P_inj[i_idx] == 0:
                        file.write(" = 0\n\tend\n")
                    else:
                        file.write(f" = 2i*Q{i}_inj \n\tend\n")
                if converter_type != "complex":
                    file.write(f") = Q{i}_inj \n")
                    file.write("\tend\n")
                
        # Compute apparent power magnitude for each PV bus by injected
        '''
        S_mag_by_bus = {}
        for bus_id in pv_nodes:
            Pg = P_inj[bus_id_map[bus_id]] * baseMVA
            Qg = Q_inj[bus_id_map[bus_id]] * baseMVA
            S_mag_by_bus[bus_id] = np.sqrt(Pg**2 + Qg**2)
            print("S", bus_id, " = ", S_mag_by_bus[bus_id], "\n")

        # Group buses by power limits
        group_to_buses = {name: [] for name in power_limits.keys()}
        group_names_sorted = sorted(power_limits.items(), key=lambda x: x[1])

        for bus_id, S in S_mag_by_bus.items():
            for name, max_val in group_names_sorted:
                if S <= max_val:
                    group_to_buses[name].append(bus_id)
                    break
        '''

        # Compute apparent power magnitude for each PV bus bu Qmax and Pmax in MATPOWER
        pmax_by_bus = {}
        for g in gen:
            bus_id = int(g[0])
            pmax = g[8]  # Column 9 (index 8) is Pmax in MW
            qmax=g[3]
            if bus_id in pmax_by_bus:
                pmax_by_bus[bus_id] += (np.sqrt(pmax**2+qmax**2))
            else:
                pmax_by_bus[bus_id] = (np.sqrt(pmax**2+qmax**2))

        # Group buses by their total Pmax and Qmax
        group_to_buses = {name: [] for name in power_limits.keys()}
        group_names_sorted = sorted(power_limits.items(), key=lambda x: x[1])

        # Only iterate over PV nodes that have generation attached
        for bus_id in pv_nodes:
            if bus_id in pmax_by_bus:
                pmax_val = pmax_by_bus[bus_id]
                
                # Find the correct group for this bus based on its Pmax
                for name, max_val_config in group_names_sorted:
                    if pmax_val <= max_val_config:
                        group_to_buses[name].append(bus_id)
                        break

        # Voltage/reactive power limits for PV buses (if enabled)
        if include_limits: 
            if calcQOfPVGensInEachIteration:
                # IterPostP for default reactive power calculation with limits
                file.write(f"IterPostP:\n")
                for group_name, bus_list in group_to_buses.items():
                    if not bus_list:
                        continue
                    for i in bus_list:
                        i_idx = bus_id_map[i]
                        file.write(f"\tQ{i}_inj = ")
                        if converter_type == "polar":
                            file.write(f"{V_mag}_{i} * (")
                            terms = []
                            for j_idx in range(n):
                                j = index_to_bus_id[j_idx]
                                aY = abs(Y[i_idx][j_idx])
                                if aY == 0:
                                    continue
                                theta = np.angle(Y[i_idx][j_idx])
                                theta_term = f" - {Y_angle}_{i}_{j}" if theta != 0 else ""
                                term = f"{Y_mag}_{i}_{j} * {V_mag}_{j} * sin({V_angle}_{i}{theta_term} - {V_angle}_{j})"
                                terms.append(term)
                            file.write(" + ".join(terms))
                            file.write(")\n")
                        elif converter_type == "rectangular":
                            terms = []
                            for j_idx in range(n):
                                j_bus_id = index_to_bus_id[j_idx]
                                if abs(G[i_idx][j_idx]) < eps and abs(B[i_idx][j_idx]) < eps:
                                    continue
                                term = (f"{f_var}_{i}*({G_var}_{i}_{j_bus_id}*{e_var}_{j_bus_id} - {B_var}_{i}_{j_bus_id}*{f_var}_{j_bus_id}) - "
                                        f"{e_var}_{i}*({B_var}_{i}_{j_bus_id}*{e_var}_{j_bus_id} + {G_var}_{i}_{j_bus_id}*{f_var}_{j_bus_id})")
                                terms.append(term)
                            file.write(" + ".join(terms))
                            file.write("\n")
                        elif converter_type == "complex":
                            terms = []
                            for j_idx in range(n):
                                j_bus_id = index_to_bus_id[j_idx]
                                y_ij = Y[i_idx, j_idx]
                                if abs(y_ij) > eps:
                                    term = f"{Y_cplx}_{i}_{j_bus_id} * {v_cplx}_{j_bus_id}"
                                    terms.append(term)
                            inner_expr = " + ".join(terms)
                            file.write(f"imag({v_cplx}_{i} * conj({inner_expr}))\n")

            # Write constraints for limits in the model file
            file.write("Limits:\n")
            for group_name, bus_list in group_to_buses.items():
                if not bus_list:
                    continue

                file.write(f"\tgroup [name=\"{group_name}\" enabled=true]:\n")
                for i in bus_list:
                    file.write(f"\t\tif cGen{i}Reg:\n")

                    # Write reactive power in limits, not the default setting
                    if not calcQOfPVGensInEachIteration:
                        i_idx = bus_id_map[i]
                        file.write(f"\t\t\tQ{i}_inj = ")
                        if converter_type == "polar":
                            file.write(f"{V_mag}_{i} * (")
                            terms = []
                            for j_idx in range(n):
                                j = index_to_bus_id[j_idx]
                                aY = abs(Y[i_idx][j_idx])
                                if aY == 0:
                                    continue
                                theta = np.angle(Y[i_idx][j_idx])
                                theta_term = f" - {Y_angle}_{i}_{j}" if theta != 0 else ""
                                term = f"{Y_mag}_{i}_{j} * {V_mag}_{j} * sin({V_angle}_{i}{theta_term} - {V_angle}_{j})"
                                terms.append(term)
                            file.write(" + ".join(terms))
                            file.write(")\n")
                        elif converter_type == "rectangular":
                            terms = []
                            for j_idx in range(n):
                                j_bus_id = index_to_bus_id[j_idx]
                                if abs(G[i_idx][j_idx]) < eps and abs(B[i_idx][j_idx]) < eps:
                                    continue
                                term = (f"{f_var}_{i}*({G_var}_{i}_{j_bus_id}*{e_var}_{j_bus_id} - {B_var}_{i}_{j_bus_id}*{f_var}_{j_bus_id}) - "
                                        f"{e_var}_{i}*({B_var}_{i}_{j_bus_id}*{e_var}_{j_bus_id} + {G_var}_{i}_{j_bus_id}*{f_var}_{j_bus_id})")
                                terms.append(term)
                            file.write(" + ".join(terms))
                            file.write("\n")
                        elif converter_type == "complex":
                            terms = []
                            for j_idx in range(n):
                                j_bus_id = index_to_bus_id[j_idx]
                                y_ij = Y[i_idx, j_idx]
                                if abs(y_ij) > eps:
                                    term = f"{Y_cplx}_{i}_{j_bus_id} * {v_cplx}_{j_bus_id}"
                                    terms.append(term)                                
                            inner_expr = " + ".join(terms)
                            file.write(f"imag({v_cplx}_{i} * conj({inner_expr}))\n")
                    # Check and enforce lower limit
                    file.write(f"\t\t\tif Q{i}_inj<=Q{i}_inj_min [signal=TooLow]:\n")
                    file.write(f"\t\t\t\t cGen{i}Reg=false\n")
                    file.write(f"\t\t\t\t Q{i}_inj=Q{i}_inj_min\n")

                    # Check and enforce upper limit
                    file.write("\t\t\telse:\n")
                    file.write(f"\t\t\t\tif Q{i}_inj>=Q{i}_inj_max [signal=TooHigh]:\n")
                    file.write(f"\t\t\t\t\t cGen{i}Reg=false\n")
                    file.write(f"\t\t\t\t\t Q{i}_inj=Q{i}_inj_max\n")

                    file.write("\t\t\t\tend\n")
                    file.write("\t\t\tend\n")
                    file.write(f"\t\tend\n")
                file.write(f"\tend\n")

        # Add PostProc if complex domain to calculate Qinj
        file.write("\tPostProc:\n")
        if converter_type == "complex":
            for group_name, bus_list in group_to_buses.items():
                if not bus_list:
                    continue
                for i in bus_list:
                    i_idx = bus_id_map[i]
                    terms = []
                    for j_idx in range(n):
                        j_bus_id = index_to_bus_id[j_idx]
                        y_ij = Y[i_idx, j_idx]
                        if abs(y_ij) > eps:
                            term = f"{Y_cplx}_{i}_{j_bus_id} * {v_cplx}_{j_bus_id}"
                            terms.append(term)                                
                    inner_expr = " + ".join(terms)
                    file.write(f"\t\tQ{i}_inj = imag({v_cplx}_{i} * conj({inner_expr}))\n")
            for bus_id in range(1, n + 1):
                real_bus_id = index_to_bus_id[bus_id - 1]
                # Format the string exactly as requested
                line = f"\t\t@main.{v_cplx}_{real_bus_id}_meas = abs({v_cplx}_{real_bus_id}) + real(rnd(g_v))\n"
                file.write(line)

            for bus_id in pq_nodes:
                i_idx = bus_id_map[bus_id]
                if P_inj[i_idx]!=0 or Q_inj[i_idx]!=0:
                    terms = []
                    for j_idx in range(n):
                        j_bus_id = index_to_bus_id[j_idx]
                        if abs(Y[i_idx, j_idx]) > eps:
                            terms.append(f"{Y_cplx}_{bus_id}_{j_bus_id} * {v_cplx}_{j_bus_id}")
                    current_sum_expr = " + ".join(terms)
                    power_expression = f"{v_cplx}_{bus_id} * conj({current_sum_expr})"
                    file.write(f"\t\t@main.S{bus_id}_meas = {power_expression} + rnd(g_inj)\n")

            for bus_id in pv_nodes:
                i_idx = bus_id_map[bus_id]
                if P_inj[i_idx]!=0 or Q_inj[i_idx]!=0:
                    terms = []
                    for j_idx in range(n):
                        j_bus_id = index_to_bus_id[j_idx]
                        if abs(Y[i_idx, j_idx]) > eps:
                            terms.append(f"{Y_cplx}_{bus_id}_{j_bus_id} * {v_cplx}_{j_bus_id}")
                    current_sum_expr = " + ".join(terms)
                    power_expression = f"{v_cplx}_{bus_id} * conj({current_sum_expr})"
                    file.write(f"\t\t@main.P{bus_id}_meas = real({power_expression}) + real(rnd(g_inj))\n")

            for bus in random_branches:
                file.write(f"\t\t@main.S_{bus[0]}_{bus[1]}_meas = {v_cplx}_{bus[0]} * conj({v_cplx}_{bus[0]} - {v_cplx}_{bus[1]}) * conj(-{Y_cplx}_{bus[0]}_{bus[1]}) + rnd(g_br)\n")
            #file.write("\n") # Add a newline for spacing
        elif converter_type == "polar":
            for bus_id in range(1, n + 1):
                    real_bus_id = index_to_bus_id[bus_id - 1]
                    # Format the string exactly as requested
                    line = f"\t\t@main.{V_mag}_{real_bus_id}_meas = {V_mag}_{real_bus_id} + rnd(g_v)\n"
                    file.write(line)
            for i in pq_nodes:
                i_idx = bus_id_map[i]
                if P_inj[i_idx]!=0 or Q_inj[i_idx]!=0:
                    terms = []
                    terms1 = []
                    for j_idx in range(n):
                        j = index_to_bus_id[j_idx]
                        aY = abs(Y[i_idx][j_idx])
                        if aY == 0:
                            continue
                        theta = np.angle(Y[i_idx][j_idx])
                        theta_term = f" - {Y_angle}_{i}_{j}" if theta != 0 else ""
                        term = f"{Y_mag}_{i}_{j} * {V_mag}_{j} * sin({V_angle}_{i}{theta_term} - {V_angle}_{j})"
                        term1 = f"{Y_mag}_{i}_{j} * {V_mag}_{j} * cos({V_angle}_{i}{theta_term} - {V_angle}_{j})"
                        terms.append(term)
                        terms1.append(term1)
                    sum_expression = " + ".join(terms)
                    sum_expression1 = " + ".join(terms1)
                    file.write(f"\t\t@main.Q{i}_meas = {V_mag}_{i} * ({sum_expression}) + rnd(g_inj)\n")
                    file.write(f"\t\t@main.P{i}_meas = {V_mag}_{i} * ({sum_expression1}) + rnd(g_inj)\n")
            for i in pv_nodes:
                i_idx = bus_id_map[i]
                if P_inj[i_idx]!=0 or Q_inj[i_idx]!=0:
                    terms = []
                    terms1 = []
                    for j_idx in range(n):
                        j = index_to_bus_id[j_idx]
                        aY = abs(Y[i_idx][j_idx])
                        if aY == 0:
                            continue
                        theta = np.angle(Y[i_idx][j_idx])
                        theta_term = f" - {Y_angle}_{i}_{j}" if theta != 0 else ""
                        term = f"{Y_mag}_{i}_{j} * {V_mag}_{j} * sin({V_angle}_{i}{theta_term} - {V_angle}_{j})"
                        term1 = f"{Y_mag}_{i}_{j} * {V_mag}_{j} * cos({V_angle}_{i}{theta_term} - {V_angle}_{j})"
                        terms.append(term)
                        terms1.append(term1)
                    sum_expression = " + ".join(terms)
                    sum_expression1 = " + ".join(terms1)
                    #file.write(f"\t\t@main.Q{i}_meas = {V_mag}_{i} * ({sum_expression}) + rnd(g_inj)\n")
                    file.write(f"\t\t@main.P{i}_meas = {V_mag}_{i} * ({sum_expression1}) + rnd(g_inj)\n")
            for bus in random_branches:
                p_formula = (f"{V_mag}_{bus[0]}^2 * {Y_mag}_{bus[0]}_{bus[0]} * cos ({Y_angle}_{bus[0]}_{bus[0]}) - {V_mag}_{bus[0]} * {Y_mag}_{bus[0]}_{bus[1]} * {V_mag}_{bus[1]} * cos ({V_angle}_{bus[0]} - {Y_angle}_{bus[0]}_{bus[1]} - {V_angle}_{bus[1]})")
                q_formula = (f"- {V_mag}_{bus[0]}^2 * {Y_mag}_{bus[0]}_{bus[0]} * sin ({Y_angle}_{bus[0]}_{bus[0]}) - {V_mag}_{bus[0]} * {Y_mag}_{bus[0]}_{bus[1]} * {V_mag}_{bus[1]} * sin ({V_angle}_{bus[0]} - {Y_angle}_{bus[0]}_{bus[1]} - {V_angle}_{bus[1]})")
                file.write(f"\t\t@main.P_{bus[0]}_{bus[1]}_meas = {p_formula} + rnd(g_br)\n")
                file.write(f"\t\t@main.Q_{bus[0]}_{bus[1]}_meas = {q_formula} + rnd(g_br)\n")
        elif converter_type == "rectangular":
            for bus_id in range(1, n + 1):
                    real_bus_id = index_to_bus_id[bus_id - 1]
                    # Format the string exactly as requested
                    line = f"\t\t@main.{V_mag}_{real_bus_id}_meas = sqrt({e_var}_{real_bus_id}^2 + {f_var}_{real_bus_id}^2) + rnd(g_v)\n"
                    file.write(line)
            for bus_id in pq_nodes:
                i_idx = bus_id_map[bus_id]
                if P_inj[i_idx] == 0 and Q_inj[i_idx] == 0:
                    continue
                p_terms = []
                q_terms = []
                for j_idx in range(n):
                    j_bus_id = index_to_bus_id[j_idx]
                    if abs(G[i_idx][j_idx]) < eps and abs(B[i_idx][j_idx]) < eps:
                        continue
                    p_term = f"{e_var}_{bus_id} * ({G_var}_{bus_id}_{j_bus_id} * {e_var}_{j_bus_id} - {B_var}_{bus_id}_{j_bus_id} * {f_var}_{j_bus_id}) + {f_var}_{bus_id} * ({B_var}_{bus_id}_{j_bus_id} * {e_var}_{j_bus_id} + {G_var}_{bus_id}_{j_bus_id} * {f_var}_{j_bus_id})"
                    p_terms.append(p_term)
                    q_term = f"{f_var}_{bus_id} * ({G_var}_{bus_id}_{j_bus_id} * {e_var}_{j_bus_id} - {B_var}_{bus_id}_{j_bus_id} * {f_var}_{j_bus_id}) - {e_var}_{bus_id} * ({B_var}_{bus_id}_{j_bus_id} * {e_var}_{j_bus_id} + {G_var}_{bus_id}_{j_bus_id} * {f_var}_{j_bus_id})"
                    q_terms.append(q_term)
                p_sum_expression = " + ".join(p_terms)
                file.write(f"\t\t@main.P{bus_id}_meas = {p_sum_expression} + rnd(g_inj)\n")
                q_sum_expression = " + ".join(q_terms)
                file.write(f"\t\t@main.Q{bus_id}_meas = {q_sum_expression} + rnd(g_inj)\n")
                # Loop over the PV nodes to write P and Q equations in rectangular coorinates
            for i in pv_nodes:
                i_idx = bus_id_map[i]
                p_terms = []
                q_terms = []
                for j_idx in range(n):
                    j_bus_id = index_to_bus_id[j_idx]
                    if abs(G[i_idx][j_idx]) < eps and abs(B[i_idx][j_idx]) < eps:
                        continue
                    p_term = f"{e_var}_{i}*({G_var}_{i}_{j_bus_id} * {e_var}_{j_bus_id} - {B_var}_{i}_{j_bus_id} * {f_var}_{j_bus_id}) + {f_var}_{i}*({B_var}_{i}_{j_bus_id} * {e_var}_{j_bus_id} + {G_var}_{i}_{j_bus_id} * {f_var}_{j_bus_id})"
                    p_terms.append(p_term)
                    q_term = f"{f_var}_{i}*({G_var}_{i}_{j_bus_id}*{e_var}_{j_bus_id} - {B_var}_{i}_{j_bus_id}*{f_var}_{j_bus_id}) - {e_var}_{i}*({B_var}_{i}_{j_bus_id}*{e_var}_{j_bus_id} + {G_var}_{i}_{j_bus_id}*{f_var}_{j_bus_id})"
                    q_terms.append(q_term)
                p_sum_expression = " + ".join(p_terms)
                file.write(f"\t\t@main.P{i}_meas = {p_sum_expression} + rnd(g_inj)\n")
                q_sum_expression = " + ".join(q_terms)
                #file.write(f"\t\t@main.Q{i}_meas = {q_sum_expression} + real(rnd(g_inj))\n")
            for bus in random_branches:
                p_formula = (f"({e_var}_{bus[0]}^2 + {f_var}_{bus[0]}^2) * {G_var}_{bus[0]}_{bus[0]} - "
                    f"({e_var}_{bus[0]}*({G_var}_{bus[0]}_{bus[1]}*{e_var}_{bus[1]} + {B_var}_{bus[0]}_{bus[1]}*{f_var}_{bus[1]}) + "
                    f"{f_var}_{bus[0]}*({G_var}_{bus[0]}_{bus[1]}*{f_var}_{bus[1]} - {B_var}_{bus[0]}_{bus[1]}*{e_var}_{bus[1]}))")
                q_formula = ( f"-({e_var}_{bus[0]}^2 + {f_var}_{bus[0]}^2) * {B_var}_{bus[0]}_{bus[0]} - "
                    f"({f_var}_{bus[0]}*({G_var}_{bus[0]}_{bus[1]}*{e_var}_{bus[1]} + {B_var}_{bus[0]}_{bus[1]}*{f_var}_{bus[1]}) - "
                    f"{e_var}_{bus[0]}*({G_var}_{bus[0]}_{bus[1]}*{f_var}_{bus[1]} - {B_var}_{bus[0]}_{bus[1]}*{e_var}_{bus[1]}))")
                file.write(f"\t\t@main.P_{bus[0]}_{bus[1]}_meas = {p_formula} + rnd(g_br)\n")
                file.write(f"\t\t@main.Q_{bus[0]}_{bus[1]}_meas = {q_formula} + rnd(g_br)\n")
        file.write("end\n")
        # ---------------------------------------------> END OF SUBMODEL
        file.write("WLSEs:\n")
        if converter_type == "complex":
            # Writing wls eqs
            for bus_id in slack:
                file.write(f"\t[w=w_v] {v_cplx}_{bus_id} = {v_cplx}_{bus_id}_sl\n")       
                file.write(f"\t[w=w_v] conj({v_cplx}_{bus_id}) = conj({v_cplx}_{bus_id}_sl)\n") 
            for bus_id in range(1, n + 1):
                    real_bus_id = index_to_bus_id[bus_id - 1]
                    # Format the string exactly as requested
                    if real_bus_id not in slack:
                        line = f"\t[w=w_v] {v_cplx}_{real_bus_id}*conj({v_cplx}_{real_bus_id}) = {v_cplx}_{real_bus_id}_meas^2\n"
                        file.write(line)
            for bus_id in pv_nodes:
                    i_idx = bus_id_map[bus_id]
                    if P_inj[i_idx]!=0 or Q_inj[i_idx]!=0:
                        terms = []
                        for j_idx in range(n):
                            j_bus_id = index_to_bus_id[j_idx]
                            if abs(Y[i_idx, j_idx]) > eps:
                                terms.append(f"{Y_cplx}_{bus_id}_{j_bus_id} * {v_cplx}_{j_bus_id}")
                        current_sum_expr = " + ".join(terms)
                        power_expression = f"{v_cplx}_{bus_id} * conj({current_sum_expr})"
                        power_expression1 = f"conj({v_cplx}_{bus_id}) * ({current_sum_expr})"
                        file.write(f"\t[w=w_inj] {power_expression} + {power_expression1} = 2 * P{bus_id}_meas\n")
            for bus_id in pq_nodes:
                    i_idx = bus_id_map[bus_id]
                    if P_inj[i_idx]!=0 or Q_inj[i_idx]!=0:
                        terms = []
                        for j_idx in range(n):
                            j_bus_id = index_to_bus_id[j_idx]
                            if abs(Y[i_idx, j_idx]) > eps:
                                terms.append(f"{Y_cplx}_{bus_id}_{j_bus_id} * {v_cplx}_{j_bus_id}")
                        current_sum_expr = " + ".join(terms)
                        power_expression = f"{v_cplx}_{bus_id} * conj({current_sum_expr})"
                        file.write(f"\t[w=w_inj] {power_expression} = S{bus_id}_meas\n")
                        power_expression1 = f"conj({v_cplx}_{bus_id}) * ({current_sum_expr})"
                        file.write(f"\t[w=w_inj] {power_expression1} = conj(S{bus_id}_meas)\n")
            for bus in random_branches:
                file.write(f"\t[w=w_br] {v_cplx}_{bus[0]} * conj({v_cplx}_{bus[0]} - {v_cplx}_{bus[1]}) * conj(-{Y_cplx}_{bus[0]}_{bus[1]}) = S_{bus[0]}_{bus[1]}_meas\n")
                file.write(f"\t[w=w_br] conj({v_cplx}_{bus[0]}) * ({v_cplx}_{bus[0]} - {v_cplx}_{bus[1]}) * (-{Y_cplx}_{bus[0]}_{bus[1]}) = conj(S_{bus[0]}_{bus[1]}_meas)\n")
            if estimation_method == "NE":
                for bus_id in pq_nodes:
                    i_idx = bus_id_map[bus_id]
                    if P_inj[i_idx]==0 and Q_inj[i_idx]==0: #sum of currents
                            terms = []
                            for j_idx in range(n):
                                j_bus_id = index_to_bus_id[j_idx]
                                if abs(Y[i_idx][j_idx]) < eps:
                                    continue
                                terms.append(f"{Y_cplx}_{bus_id}_{j_bus_id} * {v_cplx}_{j_bus_id}")
                            current_summation = " + ".join(terms)
                            file.write(f"\t[w=w_zi] {current_summation} = 0\n")
                            file.write(f"\t[w=w_zi] conj({current_summation}) = 0\n")
        elif converter_type == "polar":
            for bus_id in slack:
                file.write(f"\t[w=w_v] {V_mag}_{bus_id} = {V_mag}_{bus_id}_sl\n")       
                file.write(f"\t[w=w_v] {V_angle}_{bus_id} = {V_angle}_{bus_id}_sl\n") 
            for bus_id in range(1, n + 1):
                    real_bus_id = index_to_bus_id[bus_id - 1]
                    if real_bus_id not in slack:
                        line = f"\t[w=w_v] {V_mag}_{real_bus_id} = {V_mag}_{real_bus_id}_meas\n"
                        file.write(line)
            for i in pv_nodes:
                i_idx = bus_id_map[i]
                if P_inj[i_idx]!=0 or Q_inj[i_idx]!=0:
                    terms = []
                    terms1 = []
                    for j_idx in range(n):
                        j = index_to_bus_id[j_idx]
                        aY = abs(Y[i_idx][j_idx])
                        if aY == 0:
                            continue
                        theta = np.angle(Y[i_idx][j_idx])
                        theta_term = f" - {Y_angle}_{i}_{j}" if theta != 0 else ""
                        term = f"{Y_mag}_{i}_{j} * {V_mag}_{j} * sin({V_angle}_{i}{theta_term} - {V_angle}_{j})"
                        term1 = f"{Y_mag}_{i}_{j} * {V_mag}_{j} * cos({V_angle}_{i}{theta_term} - {V_angle}_{j})"
                        terms.append(term)
                        terms1.append(term1)
                    sum_expression = " + ".join(terms)
                    sum_expression1 = " + ".join(terms1)
                    #file.write(f"\t\t@main.Q{i}_meas = {V_mag}_{i} * ({sum_expression}) + rnd(g_inj)\n")
                    file.write(f"\t[w=w_inj] {V_mag}_{i} * ({sum_expression1}) = P{i}_meas\n")
            for i in pq_nodes:
                i_idx = bus_id_map[i]
                if P_inj[i_idx]!=0 or Q_inj[i_idx]!=0:
                    terms = []
                    terms1 = []
                    for j_idx in range(n):
                        j = index_to_bus_id[j_idx]
                        aY = abs(Y[i_idx][j_idx])
                        if aY == 0:
                            continue
                        theta = np.angle(Y[i_idx][j_idx])
                        theta_term = f" - {Y_angle}_{i}_{j}" if theta != 0 else ""
                        term = f"{Y_mag}_{i}_{j} * {V_mag}_{j} * sin({V_angle}_{i}{theta_term} - {V_angle}_{j})"
                        term1 = f"{Y_mag}_{i}_{j} * {V_mag}_{j} * cos({V_angle}_{i}{theta_term} - {V_angle}_{j})"
                        terms.append(term)
                        terms1.append(term1)
                    sum_expression = " + ".join(terms)
                    sum_expression1 = " + ".join(terms1)
                    file.write(f"\t[w=w_inj] {V_mag}_{i} * ({sum_expression}) = Q{i}_meas\n")
                    file.write(f"\t[w=w_inj] {V_mag}_{i} * ({sum_expression1}) = P{i}_meas\n")
            for bus in random_branches:
                p_formula = (f"{V_mag}_{bus[0]}^2 * {Y_mag}_{bus[0]}_{bus[0]} * cos ({Y_angle}_{bus[0]}_{bus[0]}) - {V_mag}_{bus[0]} * {Y_mag}_{bus[0]}_{bus[1]} * {V_mag}_{bus[1]} * cos ({V_angle}_{bus[0]} - {Y_angle}_{bus[0]}_{bus[1]} -{V_angle}_{bus[1]})")
                q_formula = (f"- {V_mag}_{bus[0]}^2 * {Y_mag}_{bus[0]}_{bus[0]} * sin ({Y_angle}_{bus[0]}_{bus[0]}) - {V_mag}_{bus[0]} * {Y_mag}_{bus[0]}_{bus[1]} * {V_mag}_{bus[1]} * sin ({V_angle}_{bus[0]} - {Y_angle}_{bus[0]}_{bus[1]} -{V_angle}_{bus[1]})")
                file.write(f"\t[w=w_br] {p_formula} = P_{bus[0]}_{bus[1]}_meas\n")
                file.write(f"\t[w=w_br] {q_formula} = Q_{bus[0]}_{bus[1]}_meas\n")
            if estimation_method == "NE":
                for bus_id in pq_nodes:
                    i_idx = bus_id_map[bus_id]
                    if P_inj[i_idx]==0 and Q_inj[i_idx]==0: #sum of currents
                        real_current_terms = []
                        imag_current_terms = []
                        for j_idx in range(n):
                            j_bus_id = index_to_bus_id[j_idx]
                            aY = abs(Y[i_idx][j_idx])
                            if aY == 0:
                                continue
                            theta = np.angle(Y[i_idx][j_idx])
                            theta_term = f" + {Y_angle}_{bus_id}_{j_bus_id}" if theta != 0 else ""
                            real_term = f"{Y_mag}_{bus_id}_{j_bus_id} * {V_mag}_{j_bus_id} * cos({V_angle}_{j_bus_id}{theta_term})"
                            real_current_terms.append(real_term)
                            imag_term = f"{Y_mag}_{bus_id}_{j_bus_id} * {V_mag}_{j_bus_id} * sin({V_angle}_{j_bus_id}{theta_term})"
                            imag_current_terms.append(imag_term)
                        real_sum = " + ".join(real_current_terms)
                        file.write(f"\t[w=w_zi] {real_sum} = 0\n")
                        imag_sum = " + ".join(imag_current_terms)
                        file.write(f"\t[w=w_zi] {imag_sum} = 0\n")
        elif converter_type == "rectangular":
            for bus_id in slack:
                file.write(f"\t[w=w_v] {e_var}_{bus_id} = {e_var}_{bus_id}_sl\n")       
                file.write(f"\t[w=w_v] {f_var}_{bus_id} = {f_var}_{bus_id}_sl\n") 
            for bus_id in range(1, n + 1):
                    real_bus_id = index_to_bus_id[bus_id - 1]
                    if real_bus_id not in slack:
                        line = f"\t[w=w_v] {e_var}_{real_bus_id}^2 + {f_var}_{real_bus_id}^2 = {V_mag}_{real_bus_id}_meas^2\n"
                        file.write(line)
            for i in pv_nodes:
                i_idx = bus_id_map[i]
                p_terms = []
                q_terms = []
                for j_idx in range(n):
                    j_bus_id = index_to_bus_id[j_idx]
                    if abs(G[i_idx][j_idx]) < eps and abs(B[i_idx][j_idx]) < eps:
                        continue
                    p_term = f"{e_var}_{i}*({G_var}_{i}_{j_bus_id} * {e_var}_{j_bus_id} - {B_var}_{i}_{j_bus_id} * {f_var}_{j_bus_id}) + {f_var}_{i}*({B_var}_{i}_{j_bus_id} * {e_var}_{j_bus_id} + {G_var}_{i}_{j_bus_id} * {f_var}_{j_bus_id})"
                    p_terms.append(p_term)
                    q_term = f"{f_var}_{i}*({G_var}_{i}_{j_bus_id}*{e_var}_{j_bus_id} - {B_var}_{i}_{j_bus_id}*{f_var}_{j_bus_id}) - {e_var}_{i}*({B_var}_{i}_{j_bus_id}*{e_var}_{j_bus_id} + {G_var}_{i}_{j_bus_id}*{f_var}_{j_bus_id})"
                    q_terms.append(q_term)
                p_sum_expression = " + ".join(p_terms)
                file.write(f"\t[w=w_inj] {p_sum_expression} = P{i}_meas\n")
                q_sum_expression = " + ".join(q_terms)
                #file.write(f"\t\t@main.Q{i}_meas = {q_sum_expression} + real(rnd(g_inj))\n")
            for bus_id in pq_nodes:
                i_idx = bus_id_map[bus_id]
                if P_inj[i_idx] == 0 and Q_inj[i_idx] == 0:
                    continue
                p_terms = []
                q_terms = []
                for j_idx in range(n):
                    j_bus_id = index_to_bus_id[j_idx]
                    if abs(G[i_idx][j_idx]) < eps and abs(B[i_idx][j_idx]) < eps:
                        continue
                    p_term = f"{e_var}_{bus_id} * ({G_var}_{bus_id}_{j_bus_id} * {e_var}_{j_bus_id} - {B_var}_{bus_id}_{j_bus_id} * {f_var}_{j_bus_id}) + {f_var}_{bus_id} * ({B_var}_{bus_id}_{j_bus_id} * {e_var}_{j_bus_id} + {G_var}_{bus_id}_{j_bus_id} * {f_var}_{j_bus_id})"
                    p_terms.append(p_term)
                    q_term = f"{f_var}_{bus_id} * ({G_var}_{bus_id}_{j_bus_id} * {e_var}_{j_bus_id} - {B_var}_{bus_id}_{j_bus_id} * {f_var}_{j_bus_id}) - {e_var}_{bus_id} * ({B_var}_{bus_id}_{j_bus_id} * {e_var}_{j_bus_id} + {G_var}_{bus_id}_{j_bus_id} * {f_var}_{j_bus_id})"
                    q_terms.append(q_term)
                p_sum_expression = " + ".join(p_terms)
                file.write(f"\t[w=w_inj] {p_sum_expression} = P{bus_id}_meas\n")
                q_sum_expression = " + ".join(q_terms)
                file.write(f"\t[w=w_inj] {q_sum_expression} = Q{bus_id}_meas\n")
            for bus in random_branches:
                p_formula = (f"({e_var}_{bus[0]}^2 + {f_var}_{bus[0]}^2) * {G_var}_{bus[0]}_{bus[0]} - "
                    f"({e_var}_{bus[0]}*({G_var}_{bus[0]}_{bus[1]}*{e_var}_{bus[1]} + {B_var}_{bus[0]}_{bus[1]}*{f_var}_{bus[1]}) + "
                    f"{f_var}_{bus[0]}*({G_var}_{bus[0]}_{bus[1]}*{f_var}_{bus[1]} - {B_var}_{bus[0]}_{bus[1]}*{e_var}_{bus[1]}))")
                q_formula = ( f"-({e_var}_{bus[0]}^2 + {f_var}_{bus[0]}^2) * {B_var}_{bus[0]}_{bus[0]} - "
                    f"({f_var}_{bus[0]}*({G_var}_{bus[0]}_{bus[1]}*{e_var}_{bus[1]} + {B_var}_{bus[0]}_{bus[1]}*{f_var}_{bus[1]}) - "
                    f"{e_var}_{bus[0]}*({G_var}_{bus[0]}_{bus[1]}*{f_var}_{bus[1]} - {B_var}_{bus[0]}_{bus[1]}*{e_var}_{bus[1]}))")
                file.write(f"\t[w=w_br] {p_formula} = P_{bus[0]}_{bus[1]}_meas\n")
                file.write(f"\t[w=w_br] {q_formula} = Q_{bus[0]}_{bus[1]}_meas\n")
            if estimation_method == "NE":
                for bus_id in pq_nodes:
                    i_idx = bus_id_map[bus_id]
                    if P_inj[i_idx]==0 and Q_inj[i_idx]==0: #sum of currents
                        real_terms = []
                        imag_terms = []
                        for j_idx in range(n):
                            j_bus_id = index_to_bus_id[j_idx]
                            if abs(G[i_idx][j_idx]) < eps and abs(B[i_idx][j_idx]) < eps:
                                continue
                            real_term = f"{G_var}_{bus_id}_{j_bus_id} * {e_var}_{j_bus_id} - {B_var}_{bus_id}_{j_bus_id} * {f_var}_{j_bus_id}"
                            real_terms.append(real_term)
                            imag_term = f"{B_var}_{bus_id}_{j_bus_id} * {e_var}_{j_bus_id} + {G_var}_{bus_id}_{j_bus_id} * {f_var}_{j_bus_id}"
                            imag_terms.append(imag_term)
                        real_sum_expression = " + ".join(real_terms)
                        file.write(f"\t[w=w_zi] {real_sum_expression} = 0 \n")
                        imag_sum_expression = " + ".join(imag_terms)
                        file.write( f"\t[w=w_zi] {imag_sum_expression} = 0 \n")
        if estimation_method == "EC":
            file.write("ECs:\n")
            if converter_type == "complex":
                for bus_id in pq_nodes:
                    i_idx = bus_id_map[bus_id]
                    if P_inj[i_idx]==0 and Q_inj[i_idx]==0: #sum of currents
                            terms = []
                            for j_idx in range(n):
                                j_bus_id = index_to_bus_id[j_idx]
                                if abs(Y[i_idx][j_idx]) < eps:
                                    continue
                                terms.append(f"{Y_cplx}_{bus_id}_{j_bus_id} * {v_cplx}_{j_bus_id}")
                            current_summation = " + ".join(terms)
                            file.write(f"\t{current_summation} = 0\n")
                            file.write(f"\tconj({current_summation}) = 0\n")
            elif converter_type == "polar":
                for bus_id in pq_nodes:
                    i_idx = bus_id_map[bus_id]
                    if P_inj[i_idx]==0 and Q_inj[i_idx]==0: #sum of currents
                        real_current_terms = []
                        imag_current_terms = []
                        for j_idx in range(n):
                            j_bus_id = index_to_bus_id[j_idx]
                            aY = abs(Y[i_idx][j_idx])
                            if aY == 0:
                                continue
                            theta = np.angle(Y[i_idx][j_idx])
                            theta_term = f" + {Y_angle}_{bus_id}_{j_bus_id}" if theta != 0 else ""
                            real_term = f"{Y_mag}_{bus_id}_{j_bus_id} * {V_mag}_{j_bus_id} * cos({V_angle}_{j_bus_id}{theta_term})"
                            real_current_terms.append(real_term)
                            imag_term = f"{Y_mag}_{bus_id}_{j_bus_id} * {V_mag}_{j_bus_id} * sin({V_angle}_{j_bus_id}{theta_term})"
                            imag_current_terms.append(imag_term)
                        real_sum = " + ".join(real_current_terms)
                        file.write(f"\t{real_sum} = 0\n")
                        imag_sum = " + ".join(imag_current_terms)
                        file.write(f"\t{imag_sum} = 0\n")
            elif converter_type == "rectangular":
                for bus_id in pq_nodes:
                    i_idx = bus_id_map[bus_id]
                    if P_inj[i_idx]==0 and Q_inj[i_idx]==0: #sum of currents
                        real_terms = []
                        imag_terms = []
                        for j_idx in range(n):
                            j_bus_id = index_to_bus_id[j_idx]
                            if abs(G[i_idx][j_idx]) < eps and abs(B[i_idx][j_idx]) < eps:
                                continue
                            real_term = f"{G_var}_{bus_id}_{j_bus_id} * {e_var}_{j_bus_id} - {B_var}_{bus_id}_{j_bus_id} * {f_var}_{j_bus_id}"
                            real_terms.append(real_term)
                            imag_term = f"{B_var}_{bus_id}_{j_bus_id} * {e_var}_{j_bus_id} + {G_var}_{bus_id}_{j_bus_id} * {f_var}_{j_bus_id}"
                            imag_terms.append(imag_term)
                        real_sum_expression = " + ".join(real_terms)
                        file.write(f"\t{real_sum_expression} = 0 \n")
                        imag_sum_expression = " + ".join(imag_terms)
                        file.write( f"\t{imag_sum_expression} = 0 \n")

        file.write("PostProc:\n")
        if converter_type == "complex":
            for bus_id in pq_nodes:
                i_idx = bus_id_map[bus_id]
                if P_inj[i_idx]!=0 or Q_inj[i_idx]!=0:
                    terms = []
                    for j_idx in range(n):
                        j_bus_id = index_to_bus_id[j_idx]
                        if abs(Y[i_idx, j_idx]) > eps:
                            terms.append(f"{Y_cplx}_{bus_id}_{j_bus_id} * {v_cplx}_{j_bus_id}")
                    current_sum_expr = " + ".join(terms)
                    power_expression = f"{v_cplx}_{bus_id} * conj({current_sum_expr})"
                    file.write(f"\tS{bus_id}_est = {power_expression}\n")
            for bus_id in pv_nodes:
                i_idx = bus_id_map[bus_id]
                if P_inj[i_idx]!=0 or Q_inj[i_idx]!=0:
                    terms = []
                    for j_idx in range(n):
                        j_bus_id = index_to_bus_id[j_idx]
                        if abs(Y[i_idx, j_idx]) > eps:
                            terms.append(f"{Y_cplx}_{bus_id}_{j_bus_id} * {v_cplx}_{j_bus_id}")
                    current_sum_expr = " + ".join(terms)
                    power_expression = f"{v_cplx}_{bus_id} * conj({current_sum_expr})"
                    file.write(f"\tP{bus_id}_est = real({power_expression})\n")
                    file.write(f"\tQ{bus_id}_est = imag({power_expression})\n")
            for bus in random_branches:
                file.write(f"\tS_{bus[0]}_{bus[1]}_est = {v_cplx}_{bus[0]} * conj({v_cplx}_{bus[0]} - {v_cplx}_{bus[1]}) * conj(-{Y_cplx}_{bus[0]}_{bus[1]})\n")
        elif converter_type == "polar":
            for i in pv_nodes:
                i_idx = bus_id_map[i]
                if P_inj[i_idx]!=0 or Q_inj[i_idx]!=0:
                    terms = []
                    terms1 = []
                    for j_idx in range(n):
                        j = index_to_bus_id[j_idx]
                        aY = abs(Y[i_idx][j_idx])
                        if aY == 0:
                            continue
                        theta = np.angle(Y[i_idx][j_idx])
                        theta_term = f" - {Y_angle}_{i}_{j}" if theta != 0 else ""
                        term = f"{Y_mag}_{i}_{j} * {V_mag}_{j} * sin({V_angle}_{i}{theta_term} - {V_angle}_{j})"
                        term1 = f"{Y_mag}_{i}_{j} * {V_mag}_{j} * cos({V_angle}_{i}{theta_term} - {V_angle}_{j})"
                        terms.append(term)
                        terms1.append(term1)
                    sum_expression = " + ".join(terms)
                    sum_expression1 = " + ".join(terms1)
                    file.write(f"\tQ{i}_est = {V_mag}_{i} * ({sum_expression})\n")
                    file.write(f"\tP{i}_est = {V_mag}_{i} * ({sum_expression1})\n")
            for i in pq_nodes:
                i_idx = bus_id_map[i]
                if P_inj[i_idx]!=0 or Q_inj[i_idx]!=0:
                    terms = []
                    terms1 = []
                    for j_idx in range(n):
                        j = index_to_bus_id[j_idx]
                        aY = abs(Y[i_idx][j_idx])
                        if aY == 0:
                            continue
                        theta = np.angle(Y[i_idx][j_idx])
                        theta_term = f" - {Y_angle}_{i}_{j}" if theta != 0 else ""
                        term = f"{Y_mag}_{i}_{j} * {V_mag}_{j} * sin({V_angle}_{i}{theta_term} - {V_angle}_{j})"
                        term1 = f"{Y_mag}_{i}_{j} * {V_mag}_{j} * cos({V_angle}_{i}{theta_term} - {V_angle}_{j})"
                        terms.append(term)
                        terms1.append(term1)
                    sum_expression = " + ".join(terms)
                    sum_expression1 = " + ".join(terms1)
                    file.write(f"\tQ{i}_est = {V_mag}_{i} * ({sum_expression})\n")
                    file.write(f"\tP{i}_est = {V_mag}_{i} * ({sum_expression1})\n")
            for bus in random_branches:
                p_formula = (f"{V_mag}_{bus[0]}^2 * {Y_mag}_{bus[0]}_{bus[0]} * cos ({Y_angle}_{bus[0]}_{bus[0]}) - {V_mag}_{bus[0]} * {Y_mag}_{bus[0]}_{bus[1]} * {V_mag}_{bus[1]} * cos ({V_angle}_{bus[0]} - {Y_angle}_{bus[0]}_{bus[1]} -{V_angle}_{bus[1]})")
                q_formula = (f"- {V_mag}_{bus[0]}^2 * {Y_mag}_{bus[0]}_{bus[0]} * sin ({Y_angle}_{bus[0]}_{bus[0]}) - {V_mag}_{bus[0]} * {Y_mag}_{bus[0]}_{bus[1]} * {V_mag}_{bus[1]} * sin ({V_angle}_{bus[0]} - {Y_angle}_{bus[0]}_{bus[1]} -{V_angle}_{bus[1]})")
                file.write(f"\tP_{bus[0]}_{bus[1]}_est = {p_formula}\n")
                file.write(f"\tQ_{bus[0]}_{bus[1]}_est = {q_formula}\n")       
        elif converter_type == "rectangular":
            for i in pv_nodes:
                i_idx = bus_id_map[i]
                p_terms = []
                q_terms = []
                for j_idx in range(n):
                    j_bus_id = index_to_bus_id[j_idx]
                    if abs(G[i_idx][j_idx]) < eps and abs(B[i_idx][j_idx]) < eps:
                        continue
                    p_term = f"{e_var}_{i}*({G_var}_{i}_{j_bus_id} * {e_var}_{j_bus_id} - {B_var}_{i}_{j_bus_id} * {f_var}_{j_bus_id}) + {f_var}_{i}*({B_var}_{i}_{j_bus_id} * {e_var}_{j_bus_id} + {G_var}_{i}_{j_bus_id} * {f_var}_{j_bus_id})"
                    p_terms.append(p_term)
                    q_term = f"{f_var}_{i}*({G_var}_{i}_{j_bus_id}*{e_var}_{j_bus_id} - {B_var}_{i}_{j_bus_id}*{f_var}_{j_bus_id}) - {e_var}_{i}*({B_var}_{i}_{j_bus_id}*{e_var}_{j_bus_id} + {G_var}_{i}_{j_bus_id}*{f_var}_{j_bus_id})"
                    q_terms.append(q_term)
                p_sum_expression = " + ".join(p_terms)
                file.write(f"\tP{i}_est = {p_sum_expression}\n")
                q_sum_expression = " + ".join(q_terms)
                file.write(f"\tQ{i}_est = {q_sum_expression}\n")
            for bus_id in pq_nodes:
                i_idx = bus_id_map[bus_id]
                if P_inj[i_idx] == 0 and Q_inj[i_idx] == 0:
                    continue
                p_terms = []
                q_terms = []
                for j_idx in range(n):
                    j_bus_id = index_to_bus_id[j_idx]
                    if abs(G[i_idx][j_idx]) < eps and abs(B[i_idx][j_idx]) < eps:
                        continue
                    p_term = f"{e_var}_{bus_id} * ({G_var}_{bus_id}_{j_bus_id} * {e_var}_{j_bus_id} - {B_var}_{bus_id}_{j_bus_id} * {f_var}_{j_bus_id}) + {f_var}_{bus_id} * ({B_var}_{bus_id}_{j_bus_id} * {e_var}_{j_bus_id} + {G_var}_{bus_id}_{j_bus_id} * {f_var}_{j_bus_id})"
                    p_terms.append(p_term)
                    q_term = f"{f_var}_{bus_id} * ({G_var}_{bus_id}_{j_bus_id} * {e_var}_{j_bus_id} - {B_var}_{bus_id}_{j_bus_id} * {f_var}_{j_bus_id}) - {e_var}_{bus_id} * ({B_var}_{bus_id}_{j_bus_id} * {e_var}_{j_bus_id} + {G_var}_{bus_id}_{j_bus_id} * {f_var}_{j_bus_id})"
                    q_terms.append(q_term)
                p_sum_expression = " + ".join(p_terms)
                file.write(f"\tP{bus_id}_est = {p_sum_expression}\n")
                q_sum_expression = " + ".join(q_terms)
                file.write(f"\tQ{bus_id}_est = {q_sum_expression}\n")
            for bus in random_branches:
                p_formula = (f"({e_var}_{bus[0]}^2 + {f_var}_{bus[0]}^2) * {G_var}_{bus[0]}_{bus[0]} - "
                    f"({e_var}_{bus[0]}*({G_var}_{bus[0]}_{bus[1]}*{e_var}_{bus[1]} + {B_var}_{bus[0]}_{bus[1]}*{f_var}_{bus[1]}) + "
                    f"{f_var}_{bus[0]}*({G_var}_{bus[0]}_{bus[1]}*{f_var}_{bus[1]} - {B_var}_{bus[0]}_{bus[1]}*{e_var}_{bus[1]}))")
                q_formula = ( f"-({e_var}_{bus[0]}^2 + {f_var}_{bus[0]}^2) * {B_var}_{bus[0]}_{bus[0]} - "
                    f"({f_var}_{bus[0]}*({G_var}_{bus[0]}_{bus[1]}*{e_var}_{bus[1]} + {B_var}_{bus[0]}_{bus[1]}*{f_var}_{bus[1]}) - "
                    f"{e_var}_{bus[0]}*({G_var}_{bus[0]}_{bus[1]}*{f_var}_{bus[1]} - {B_var}_{bus[0]}_{bus[1]}*{e_var}_{bus[1]}))")
                file.write(f"\tP_{bus[0]}_{bus[1]}_est = {p_formula}\n")
                file.write(f"\tQ_{bus[0]}_{bus[1]}_est = {q_formula}\n")
        file.write("end\n")

    print("\nConversion successful!")
    print(f"Output written to: {dmodl_output_path}")

# Standard entry point for a Python script
if __name__ == "__main__":
    main()