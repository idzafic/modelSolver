import re
import numpy as np
import xml.etree.ElementTree as ET
import json
import argparse 
import os      
import sys      
import random 
from typing import Dict, Optional

def is_debugging():
    """Check if the script is currently being debugged."""
    # sys.gettrace() returns the current tracing function set by the debugger.
    # If a debugger is active, it will return a non-None value.
    return sys.gettrace() is not None

import math
import re

def evaluate_matpower_expr(expr: str) -> float:
    """
    Safely evaluates a MATPOWER-style expression from a .m file.
    Examples: "50/3", "-50/3", "12/sqrt(3)", "sqrt(2)/2", "1.05", "2^3", "(1+sqrt(5))/2"
    """
    if not expr or not expr.strip():
        return 0.0

    expr = expr.strip()

    # Matlab uses ^ for power → convert to Python **
    expr = expr.replace('^', '**')
    expr = expr.replace('Inf', '1e+100')

    # Safe evaluation namespace (only math functions + basic builtins)
    safe_dict = {
        "__builtins__": {},
        "sqrt": math.sqrt,
        "exp": math.exp,
        "log": math.log,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "pi": math.pi,
        "abs": abs,
        "pow": pow,
    }

    try:
        result = eval(expr, safe_dict)
        return float(result)
    except Exception as e:
        print(f"Warning: Could not evaluate MATPOWER expression '{expr}': {e}")
        # Last-resort fallback: strip everything except valid number characters
        try:
            clean = re.sub(r'[^0-9.eE+-]', '', expr)
            if clean:
                return float(clean)
        except:
            pass
        return 0.0

def addToYij(i: int, j: int, Y_ij: Dict[int, complex], value: complex) -> None:
    """
    Append a complex value to the Y_ij (offdiagonal) map using combined key from i and j.
    
    Args:
        i: First index
        j: Second index  
        Y_ij: Dictionary mapping combined index to complex values
        value: Complex value to add or append
    """
    key = i * 1000000 + j   #create single index (combine i and j)
    if key in Y_ij:
        Y_ij[key] += value  # Update existing value
    else:
        Y_ij[key] = value   # Insert new value


def getFromYij(i: int, j: int, Y_ij: Dict[int, complex]) -> Optional[complex]:
    """
    Retrieve a complex value from the Y_ij map using combined key from i and j.
    
    Args:
        i: First index
        j: Second index
        Y_ij: Dictionary mapping combined index to complex values
        
    Returns:
        Complex value if found, None otherwise
    """
    key = i * 1000000 + j
    if key in Y_ij:
        return Y_ij[key]  # Return existing value
    return None  # Return None if key doesn't exist

def addToSet(busIDFrom : int, busIDTo : int, toSet : set) -> None:
    key = 100000000 * busIDFrom+ busIDTo
    toSet.add(key)

def isInSet(busIDFrom : int, busIDTo : int, inSet : set) -> bool:
    key = 100000000 * busIDFrom + busIDTo
    if key in inSet:  
        return True
    return False

def getXFMRComment(busIDFrom : int, busIDTo : int, tapChangers : set, phaseShifers : set) -> Optional[str]:
    strToRet = None
    if isInSet(busIDFrom, busIDTo, tapChangers):
        strToRet = "\t//TapCh "

    if isInSet(busIDFrom, busIDTo, phaseShifers):
        if strToRet:
            strToRet += " + PhShift"
        else:
            strToRet = "\t//PhShift "

    return strToRet

def writeYParam(file, y, busFrom : int, busTo : int, lblMag : str, lblAngle : str, tapChangers:set, phaseShifters: set, converterType : int) -> None:
    if converterType == 0:  #polar
        magnitude = abs(y)
        angleRad = np.angle(y)
        file.write(f"\t{lblMag}_{busFrom}_{busTo} = {magnitude}; ")
        if angleRad != 0:
            file.write(f"{lblAngle}_{busFrom}_{busTo} = {angleRad} ")
        if busFrom != busTo:
            strComment = getXFMRComment(busFrom, busTo, tapChangers, phaseShifters)
            if strComment:
                file.write(strComment)
        file.write("\n")

def writeYParamCmplx(file, y, busFrom : int, busTo : int, lblY : str, tapChangers:set, phaseShifters: set, converterType : int) -> None:
    if converterType == 2:  #complex
        if y.real == 0 and y.imag == 0:
            val_str = "0"
        elif y.real == 0:
            val_str = f"{y.imag}i"
        elif y.imag == 0:
            val_str = f"{y.real}"
        else:
            val_str = f"{y.real}{'+' if y.imag > 0 else ''}{y.imag}i"
        file.write(f"\t{lblY}_{busFrom}_{busTo} = {val_str}; ")
        if busFrom != busTo:
            strComment = getXFMRComment(busFrom, busTo, tapChangers, phaseShifters)
            if strComment:
                file.write(strComment)
        file.write("\n")

def writeDiagPowerTermInPolar(realTerms, busID : int, lblV, lblYMag, lblYAngle, bZeroInjection : bool) -> None:
    if bZeroInjection:
        realTerms.append(f"\t{lblYMag}_{busID}_{busID}*{lblV}_{busID}*cos({lblYAngle}_{busID}_{busID})")
    else:
        realTerms.append(f"\t{lblYMag}_{busID}_{busID}*{lblV}_{busID}^2*cos({lblYAngle}_{busID}_{busID})")

def writeDiagReactivePowerTermInPolar(imagTerms, busID : int, lblV, lblYMag, lblYAngle, bZeroInjection : bool) -> None:
    if bZeroInjection:
        imagTerms.append(f"\t-{lblYMag}_{busID}_{busID}*{lblV}_{busID}*sin({lblYAngle}_{busID}_{busID})")
    else:
        imagTerms.append(f"\t-{lblYMag}_{busID}_{busID}*{lblV}_{busID}^2*sin({lblYAngle}_{busID}_{busID})")

def writeOffDiagPowerTermInPolar(realTerms, busI, busJ, lblV, lblVangle, lblYMag, lblYAngle, bFirst : bool):
    if not bFirst:
        realTerms.append(f" + {lblYMag}_{busI}_{busJ}*{lblV}_{busJ}*cos({lblVangle}_{busI}-{lblYAngle}_{busI}_{busJ}-{lblVangle}_{busJ})")
    else:
        realTerms.append(f"{lblYMag}_{busI}_{busJ}*{lblV}_{busJ}*cos({lblVangle}_{busI}-{lblYAngle}_{busI}_{busJ}-{lblVangle}_{busJ})")

def writeOffDiagReactivePowerTermInPolar(imagTerms, busI, busJ, lblV, lblVangle, lblYMag, lblYAngle, bFirst : bool):
    if not bFirst:
        imagTerms.append(f" + {lblYMag}_{busI}_{busJ}*{lblV}_{busJ}*sin({lblVangle}_{busI}-{lblYAngle}_{busI}_{busJ}-{lblVangle}_{busJ})")
    else:
        imagTerms.append(f"{lblYMag}_{busI}_{busJ}*{lblV}_{busJ}*sin({lblVangle}_{busI}-{lblYAngle}_{busI}_{busJ}-{lblVangle}_{busJ})")

def writeZIRealTermPolar(realTerms, busIDFrom : int, busIDTo : int, lblV, lblVangle, lblYMag, lblYAngle, firstTerm : bool):
    if firstTerm:
        realTerms.append(f"\t{lblYMag}_{busIDFrom}_{busIDTo}*{lblV}_{busIDTo}*cos({lblYAngle}_{busIDFrom}_{busIDTo}+{lblVangle}_{busIDTo})")
    else:
        realTerms.append(f" + {lblYMag}_{busIDFrom}_{busIDTo}*{lblV}_{busIDTo}*cos({lblYAngle}_{busIDFrom}_{busIDTo}+{lblVangle}_{busIDTo})")

def writeZIImagTermPolar(imagTerms, busIDFrom : int, busIDTo : int, lblV, lblVangle, lblYMag, lblYAngle, firstTerm : bool):
    if firstTerm:
        imagTerms.append(f"\t{lblYMag}_{busIDFrom}_{busIDTo}*{lblV}_{busIDTo}*sin({lblYAngle}_{busIDFrom}_{busIDTo}+{lblVangle}_{busIDTo})")
    else:
        imagTerms.append(f" + {lblYMag}_{busIDFrom}_{busIDTo}*{lblV}_{busIDTo}*sin({lblYAngle}_{busIDFrom}_{busIDTo}+{lblVangle}_{busIDTo})")

def writeDiagComplexSumTerm(terms, busID, lblV, lblY):
    terms.append(f"{lblY}_{busID}_{busID} * {lblV}_{busID}")

def writeOffDiagComplexSumTerm(terms, busI, busJ, lblV, lblY, bFirst):
    pref = " + " if not bFirst else ""
    terms.append(f"{pref}{lblY}_{busI}_{busJ} * {lblV}_{busJ}")

def loadPrecomputedY(filename: str, Y_ii: np.ndarray, Y_ij: Dict[int, complex]) -> bool:
    """
    Loads precomputed system admittance matrix Y from a text file in rectangular form.
    
    Expected format (one entry per line, tab/space separated):
      row  col  Real(G)  Imaginary(B)    # bus numbers are 1-based
    
    - Diagonal elements (row == col) → stored in Y_ii[bus-1]
    - Off-diagonal elements       → stored in Y_ij using key = i * 1000000 + j
    
    Returns:
        True  if loading was successful and at least one entry was loaded
        False otherwise → caller should fall back to internal Y computation
    """
    Y_ii.fill(0j)           # Clear any previous content
    Y_ij.clear()

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"  -> Precomputed Y file not found: {filename}")
        return False
    except Exception as e:
        print(f"  -> Error opening Y file {filename}: {e}")
        return False

    if not lines:
        print(f"  -> File is empty: {filename}")
        return False

    loaded = 0
    errors = 0
    skipped = 0

    for lineno, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('%') or line.startswith('#'):
            skipped += 1
            continue

        parts = line.split()
        if len(parts) < 4:
            skipped += 1
            continue

        try:
            i   = int(parts[0])          # 1-based row / from-bus
            j   = int(parts[1])          # 1-based col / to-bus
            g   = float(parts[2])        # real part = conductance G
            b   = float(parts[3])        # imaginary part = susceptance B

            y = complex(g, b)

            if i == j:
                # Diagonal element
                idx = i - 1
                if 0 <= idx < len(Y_ii):
                    Y_ii[idx] = y
                    loaded += 1
                else:
                    print(f"  -> Invalid diagonal bus index {i} at line {lineno}")
                    errors += 1
            else:
                # Off-diagonal
                # Your code stores both directions → we do the same if file contains both
                addToYij(i-1, j-1, Y_ij, y)
                loaded += 1

        except (ValueError, IndexError) as e:
            print(f"  -> Parse error at line {lineno}: {line}")
            errors += 1
            continue

    print(f"  Loaded {loaded} nonzero Y entries from {os.path.basename(filename)}")
    if skipped > 0:
        print(f"  -> Skipped {skipped} lines (header/comments/empty)")
    if errors > 0:
        print(f"  ->{errors} lines had parsing or index errors")

    success = loaded > 0

    if not success:
        print("  -> No valid entries loaded -> will compute Y internally")
        Y_ii.fill(0j)
        Y_ij.clear()

    return success

def main():
    #  If you want to bypass input arguments (uncomment the line below)
    #debug_mode = True #is_debugging() 

    # ────────────────────────────────────────────────
    #  Define all important input/config values
    #  These will be set either manually (debug) or from args
    # ────────────────────────────────────────────────
    matpower_file = None       # required
    output_path   = None       # optional
    res_path      = "res"       # optional (output folder)

    config_file_path = "config.xml"
    greek_symbols_path = "greek_symbols.json"
    
    #  DEBUG MODE: hardcode values
    if debug_mode:
        print("DEBUG MODE DETECTED — using hardcoded values")

        # ── Manually set case ID here ──
        # case3375wp - Polish system
        # case6468rte - French system
        matpower_file = "case1197.m" # "case_ACTIVSg25k.m" "case118.m" #"case3375wp.m" #"case_SyntheticUSA.m" #"case6495rte.m" #"case6468rte.m" #"case300.m"
        #output_path   = "res\\custom_output.dmodl"           # uncomment if you want
        # res_path      = "results"                        # uncomment if you want
    else:
        #  NORMAL MODE: parse from command line
        parser = argparse.ArgumentParser(
            description="Converts a MATPOWER .m case file to a dTwin .dmodl model file.",
            formatter_class=argparse.RawTextHelpFormatter
        )

        parser.add_argument(
            "matpower_file",
            help="Path to the input MATPOWER .m case file."
        )

        parser.add_argument(
            "-o", "--output",
            help="Path for the output .dmodl file.\n"
                 "If not specified, created in same dir with same name (.dmodl)."
        )

        parser.add_argument(
            "-r", "--resPath",
            help="Output folder for the .dmodl file.\n"
                 "If not specified, same directory as input file."
        )

        args = parser.parse_args()

        matpower_file = args.matpower_file
        output_path   = args.output
        res_path      = args.resPath

    #  Now use the same variables in both modes
    if not matpower_file:
        print("Error: matpower_file is required")
        sys.exit(1)

    # You can keep your original paths logic if you want relative path handling
    matpower_input_path = f"cases/{matpower_file}"

    # Determine output path (same logic as before, but now using variables)
    base_name = os.path.basename(matpower_file)
    file_name_without_ext = os.path.splitext(base_name)[0]
    if output_path:
        dmodl_output_path = output_path
    else:
        if res_path:
            dmodl_output_path = f"{res_path}/{file_name_without_ext}"
        else:
            dmodl_output_path = file_name_without_ext

    print(f"Starting conversion...")
    print(f"  > Input MATPOWER file: {matpower_input_path}")

    if debug_mode:
        print(f"  > Debug mode active — hardcoded file used")

    #   Error handling for file not found
    try:
        # Parsing configuration file for user options and variable naming
        tree = ET.parse(config_file_path)
        rootNode = tree.getroot()
        root = rootNode.find('powerFlow')
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
    pfprecision_element = options.find('pfprecision')
    pfprecision = pfprecision_element.text.strip().lower() if pfprecision_element is not None else '1e-6'
    reportType_element = options.find('reportType')
    reportType = reportType_element.text.strip() if reportType_element is not None else 'All'

    ignorePhaseShifters = options.find('ignorePhaseShifters').text.strip().lower() == 'true'

    converter_type_element = options.find('converter_type')
    converter_type = converter_type_element.text.strip().lower() if converter_type_element is not None else 'polar'  # default 'polar' ako nije navedeno

    iConverterType = 0  #default polar (faster comparison with numbers)
    if converter_type == "polar":
        iConverterType = 0
    elif converter_type == "rectangular":
        iConverterType = 1
    elif converter_type == "complex":
        iConverterType = 2

    include_limits = options.find('include_limits').text.strip().lower() == 'true'
    comment_equations = options.find('comment_equations').text.strip().lower() == 'true'
    #comment_params = options.find('comment_params').text.strip().lower() == 'true'
    zero_loads = options.find('zero_loads').text.strip().lower() == 'true'
    zip_coeff = options.find('zip_coeff').text.strip().lower() == 'true'
    zip_Kpone = options.find('zip_Kpone').text.strip().lower() == 'true'
    calcQOfPVGensInEachIteration = options.find('calcQOfPVGensInEachIteration').text.strip().lower() == 'true'
    useSumOfCurrentsForZI = options.find('useSumOfCurrentsForZI').text.strip().lower() == 'true'
    convertLoadsToImpedance = options.find('convertLoadsToImpedance').text.strip().lower() == 'true'
    loadScaling = options.find('loadScaling').text.strip().lower() == 'true'
    genScaling = options.find('genScaling').text.strip().lower() == 'true'
    loadTransfer = options.find('loadTransfer').text.strip().lower() == 'true'
    genLimits = options.find('genLimits').text.strip().lower() == 'true'
    useCurrentOnRHS = options.find('useCurrentOnRHS').text.strip().lower() == 'true'
    usePrecalculatedY = options.find('usePrecalculatedY').text.strip().lower() == 'true' #System admittance matrix Y is calculated in MATPOWER and exported in polar form in folder Y
    flatStart = options.find('flatStart').text.strip().lower() == 'true'    #if true: initial values will be set to slack values (typically 1 with angle 0)
    
    if usePrecalculatedY:
        strYFileName = "Y/"+file_name_without_ext + ".txt"
        if not os.path.isfile(strYFileName):
            usePrecalculatedY = False
            print(f"Warning! Cannot find precalculate system admittance matrix Y in file {strYFileName}. Will be internaly computed.")

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

    includeConsumptionCurves = rootCommon.find('includeConsumptionCurves').text.strip().lower() == 'true'
    numberOfLoadConsumptionCurves = int(rootCommon.find('numberOfLoadConsumptionCurves').text)
    numberOfGenConsumptionCurves = int(rootCommon.find('numberOfGenConsumptionCurves').text)

    nBuses = 0
    nGens = 0
    nBranches = 0

    eps=1e-14
    number_pattern = re.compile(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?')

    BUS_TYPE_POS = 1 
    PQ = 1
    PV = 2
    SLACK = 3

    def parse_matrix_block(lines, start_index, elementType):
        nonlocal nBuses, nGens, nBranches
        LARGE = 1e40
        SMALL = -1e40

        # Rough pre-allocation
        if elementType == 0:        # bus
            estimated = (len(lines) - start_index) // 2
        elif elementType == 1:      # gen
            estimated = (len(lines) - start_index) // 2
        else:                       # branch
            estimated = len(lines) - nBuses - nGens

        if estimated <= 0:
            estimated = 100

        data = [None] * estimated
        iPos = 0
        i = start_index + 1                     # skip header line
        detectedOffStatus = False
        while i < len(lines):
            line = lines[i].strip()
            i += 1

            if not line or line.startswith('%'):
                continue
            if '];' in line:
                break

            # Remove comment and trailing semicolon
            row = line.split('%')[0].strip()
            if row.endswith(';'):
                row = row[:-1].strip()

            # Split on any whitespace → each token is either a number or an expression
            tokens = re.split(r'\s+', row)

            row_data = []
            for token in tokens:
                if token:                                   # skip empty strings
                    value = evaluate_matpower_expr(token)
                    row_data.append(value)

            if row_data:
                if elementType == 0: #bus
                    if int(row_data[BUS_TYPE_POS]) == PV:    #if PV
                        row_data[BUS_TYPE_POS] = 99      # set it to 99 (later we will go through gens and set it to PV if there is any active gen on the bus)
                elif elementType == 1:  #gen
                    #check the status of the gen
                    if row_data[7] < 0.1:  
                        if not detectedOffStatus: 
                            detectedOffStatus = True
                            print(f"Detected off-service generator! First time at bus {int(row_data[0])}")
                        continue    #ignore gens with status == 0
                else:
                    if row_data[10] < 0.1:
                        if not detectedOffStatus: 
                            detectedOffStatus = True
                            print(f"Detected off-service line! First time at branch {int(row_data[0])}-{int(row_data[1])}")
                        continue    #ignore lines with status =0
                    if row_data[2] < 0 or row_data[3] < 0:
                        print(f"WARNING!! Negative branch impedances on line {int(row_data[0])}-{int(row_data[1])}. r={row_data[2]}, x={row_data[3]}")
        
                if iPos < len(data):
                    data[iPos] = row_data
                else:
                    data.append(row_data)
                iPos += 1

        # Update counters
        if elementType == 0: #bus
            nBuses = iPos
        elif elementType == 1:  #gen
            nGens = iPos
        else:
            nBranches = iPos    #branch

        return data[:iPos], i

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
            content = line.split('%')[0].strip()
            if '=' not in content:
                i += 1
                continue
            
            expr = content.split('=', 1)[1].strip()
            if expr.endswith(';'):
                expr = expr[:-1].strip()
            
            baseMVA = evaluate_matpower_expr(expr)
            
            if baseMVA == 0.0 and expr.strip():
                print(f"ERROR: baseMVA evaluated to 0 from: {expr!r}")
                exit()
            
            i += 1
 
        # Parse bus, gen, and branch matrices
        elif line.startswith('mpc.bus'):
            bus, i = parse_matrix_block(lines, i, 0)
        elif line.startswith('mpc.gen ='):
            gen, i = parse_matrix_block(lines, i, 1)
        elif line.startswith('mpc.branch'):
            branch, i = parse_matrix_block(lines, i, 2)
        elif bus and gen and branch:
            break
        else:
            i += 1
    
    if not baseMVA:
        print("ERROR! Base power not defined!")
        exit()

    print(f"Detected {nBuses} buses, {nGens} generators, and {nBranches} branches.")
    # Map bus indices for easier access
    n = len(bus)
    bus_id_map = {int(bus[i][0]): i for i in range(n)}
    index_to_bus_id = {v: k for k, v in bus_id_map.items()}
    pq_nodes, pv_nodes, slack = [], [], []

    # Organizing generators and branches by bus_id, set PV flags at buses for in-service generators
    gen_by_bus = {}
    for g in gen:
        busID = int(g[0])
        if busID not in gen_by_bus:
            gen_by_bus[busID] = []
        gen_by_bus[busID].append(g)
        idx = bus_id_map[busID]
        if int(bus[idx][BUS_TYPE_POS]) == 99:
            bus[idx][BUS_TYPE_POS] = PV

    # Assign node type
    nSlacks = 0
    pvNodeWithoutGensDetected = False
    for i in range(n):
        bus_type = int(bus[i][BUS_TYPE_POS])
        bus_id = int(bus[i][0])
        if bus_type == 99:
            if not pvNodeWithoutGensDetected:
                pvNodeWithoutGensDetected = True
                print(f"Detected previously declared PV node wihtout any in-service generator. First time on bus {bus_id}")
            bus_type = PQ
            bus[i][BUS_TYPE_POS] = PQ

        if bus_type == PQ:
            pq_nodes.append(bus_id)
        elif bus_type == PV:
            pv_nodes.append(bus_id)
        elif bus_type == SLACK:
            slack.append(bus_id)
            nSlacks += 1
        else:
            print(f"ERROR! Unknown bus type!!! BusID={bus_id}, type = {bus_type}")
            exit()

    if nSlacks == 0:
        print(f"There is no slack node in the system!!")
        exit
    elif nSlacks > 1:
        print(f"Warning!! Detect {nSlacks} slacks in the system!")

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
    print("Creating empty Y sparse matrix structure ...")
    #Y = np.zeros((n, n), dtype=complex)

    Y_ii = np.zeros((n), dtype=complex)

    # Initial declaration of the map
    Y_ij: Dict[int, complex] = {}

    phaseShifters = set()
    tapChangers = set()

    nodeBranches = [set() for _ in range(n)]    #allocate topology info

    YfromMATPOWER = set()

    if usePrecalculatedY:
        # TODO: Load precalculated matrix Y from strYFileName and store it to Y_ii and Y_ij
        #       Use methods addToYij and Y_ii
        print("Loading precomputed sparse matrix Y...")
        usePrecalculatedY = loadPrecomputedY(strYFileName, Y_ii, Y_ij)

    # Create a set to track transformer connections
    transformer_set = set()
    if not usePrecalculatedY:
        print("Computing sparse matrix Y...")
    # Loop over each row in the branch data to construct the admittance matrix
    tapChangerDetected = False
    phaseShifterDetected = False
    nTapChangers = 0
    nPhaseShifters = 0
    nNotInService = 0 
    for row in branch:
        f_bus = int(row[0])  # From bus
        t_bus = int(row[1])  # To bus
        status = int(row[10])

        if status != 1:
            if nNotInService == 0:
                print(f"Detected out of service branch! First on {f_bus}-{t_bus}")
            nNotInService += 1
            continue

        r, x, b = row[2], row[3], row[4]  # Line resistance, reactance, and shunt susceptance
        tapVal = row[8]
        if tapVal == 0:
            tapVal = 1.0
        if tapVal != 1.0:
            if not tapChangerDetected:
                print(f"Tap changer detected! First on branch {f_bus}-{t_bus}")
                tapChangerDetected = True
            nTapChangers += 1
            addToSet(f_bus, t_bus, tapChangers)

        if not usePrecalculatedY:
            if tapVal < 0.5 or tapVal > 1.6:
                print(f"Warning! Detected tap ratio with value {tapVal} on branch {f_bus}-{t_bus}. Setting to 1.0")
                tapVal = 1

        ratio = tapVal  # Tap ratio; default to 1 if zero
        angle_deg = row[9]  # Phase shift angle in degrees
        if angle_deg != 0:
            if not phaseShifterDetected:
                print(f"Phase shifter detected! First on branch {f_bus}-{t_bus}")
                phaseShifterDetected = True
            nPhaseShifters += 1

            addToSet(f_bus, t_bus, phaseShifters)
            if ignorePhaseShifters:
                angle_deg = 0      #RESET phase shift
        if not usePrecalculatedY:
            if angle_deg < -60 or angle_deg > 60:
                print(f"Warning! Detected phase shift with value {angle_deg} on branch {f_bus}-{t_bus}. Setting to 0.0")
                angle_deg = 0

        angle_rad = np.deg2rad(angle_deg)
        a = ratio * np.exp(1j * angle_rad)  # Complex tap ratio for transformer

        # Calculate series admittance and shunt admittance
        y = 1 / complex(r, x) if r != 0 or x != 0 else 0
        b_shunt = complex(0, b / 2)

        # Map from bus IDs to matrix indices
        f_idx = bus_id_map[f_bus]
        t_idx = bus_id_map[t_bus]

        nodeBranches[f_idx].add(t_idx)   #update list of branches for a node
        nodeBranches[t_idx].add(f_idx)   #update list of branches for a node

        # Identify transformer branches (with tap or phase shift)
        if ratio != 0 and ratio != 1.0 or angle_deg != 0.0:
            transformer_set.add((f_idx, t_idx))
            transformer_set.add((t_idx, f_idx))
        
        if not usePrecalculatedY:
            # Update admittance matrix using transformer model
            Y_ii[f_idx] += (y + b_shunt) / (a * np.conj(a))
            Y_ii[t_idx] += y + b_shunt
            addToYij(f_idx, t_idx, Y_ij, -y / np.conj(a))
            addToYij(t_idx, f_idx, Y_ij, -y / a)

    if nTapChangers > 0:
        print(f"Detected {nTapChangers} tap changers!")

    if nPhaseShifters > 0:
        print(f"Detected {nPhaseShifters} phase shifters!")

    if nNotInService > 0:
        print(f"Detected {nNotInService} out of service branches!")

    # Add shunt conductance (gs) and susceptance (bs) from the bus data to diagonal elements of Y
    detectedShunts = False
    for row in bus:
        busID = int(row[0])
        i_idx = bus_id_map[busID]
        gs = row[4] / baseMVA #<--- shunt admittances aren't in per unit
        bs = row[5] / baseMVA #<--- shunnt admittances aren't in per unit
        if gs != 0 or bs != 0:
            if not detectedShunts:
                print(f"Detected shunts in the system. First on bus {busID}.")
                detectedShunts = True
        #Y[i_idx][i_idx] += complex(gs, bs)
        Y_ii[i_idx] += complex(gs, bs)

    #G = Y.real  # Conductance matrix
    #B = Y.imag  # Susceptance matrix

    # Set slack bus voltage magnitude and angle from generator data or default to bus data
    for real_bus_id in slack:
        row = bus[bus_id_map[real_bus_id]]
        v_slack = gen_by_bus[real_bus_id][0][5] if real_bus_id in gen_by_bus else row[7]
        Va_deg = row[8]
        v_angle = np.deg2rad(Va_deg)

    # Load Transfer File Parsing
    current_dir = os.getcwd()
    case_name = os.path.splitext(os.path.basename(matpower_file))[0]
    loadTransferSelect = False
    if loadTransfer:
        lt_xml_name = f"aux/{case_name}_lt.xml"
        lt_xml_path = os.path.join(current_dir, lt_xml_name)
        loadTransferSelect = os.path.isfile(lt_xml_path)
        if not loadTransferSelect:
            print(f"Case _lt file not found.")

    genLimitsSelect = False
    if genLimits:
        # Gen Limits File Parsing
        pv_xml_name = f"aux/{case_name}_pv.xml"
        pv_xml_path = os.path.join(current_dir, pv_xml_name)
        genLimitsSelect = genLimits and os.path.isfile(pv_xml_path)
        if not genLimitsSelect:
            print(f"Case _pv file not found.")  
        pv_overrides = {}
    
    if loadTransferSelect:
        resFileName = dmodl_output_path + "_lt.dmodl"
        lt_params = {'dk': 0.0, 'k': 0.0, 'dg': 0.0, 'kg': 0.0}
        active_pq_set = set()
        active_pv_set = set()
        lt_root = ET.parse(lt_xml_path).getroot()
        s = lt_root.find('settings')
        if s is not None:
            lt_params['dk'] = float(s.get('delta_k', 0))
            lt_params['k']  = float(s.get('k', 0))
            lt_params['dg'] = float(s.get('delta_gen', 0))
            lt_params['kg'] = float(s.get('k_gen', 0))

        # Iterating all buses in XML file
        for b in lt_root.findall('.//bus'):
            bid = int(b.get('id'))
            if bid in bus_id_map:
                if bid in pq_nodes:
                    active_pq_set.add(bid)
                elif bid in pv_nodes:
                    active_pv_set.add(bid)
                else:
                    print(f"Bus {bid} is Slack, Load Transfer skipped.")
            else:
                print(f"Error: Bus {bid} does not exist in case!")
        
        print(f"LT data loades: {len(active_pq_set)} PQ and {len(active_pv_set)} PV buses.")
    else:
        resFileName = dmodl_output_path + ".dmodl"

    
    if genLimitsSelect:
        resFileName = dmodl_output_path + "_pv.dmodl"
        print(f"  > Loading Gen Limits file: {pv_xml_name}")
        pv_root = ET.parse(pv_xml_path).getroot()
        
        for b in pv_root.findall('.//bus'):
            bid = int(b.get('id'))
            if bid in bus_id_map and (bid in pv_nodes or bid in slack):
                pv_overrides[bid] = {
                    'min': float(b.get('qmin')),
                    'max': float(b.get('qmax'))
                }
            else:
                print(f"Bus {bid} from PV file is not a generator bus.")
        
        print(f"Loaded overrides for {len(pv_overrides)} generators.")

    # Begin writing the dTwin .dmodl file
    with open(resFileName, "w", encoding='utf-8') as file:
        # Write model header
        file.write(f"Header:\n\tmaxIter=100\n\treport={reportType}\t//Solved - only final solved solution, All - shows solved and nonSolved with iterations, AllDetails - All + debug information\n\tmaxReps = -1\n\toutToTxt = false\nend\n")
        file.write("//Generated by MATPOWER to dmodl coverter\n")
        if converter_type=="complex":
            file.write(f"Model [type=NL domain=cmplx eps={pfprecision} name=\"PF in {converter_type} coordinates\"]:\n")
        else:
            file.write(f"Model [type=NL domain=real eps={pfprecision} name=\"PF in {converter_type} coordinates\"]:\n")

        # Declare variables for voltage angles and magnitudes (except slack bus)
        file.write("Vars [out=true]:\n")
        print("Writing Vars...")
        for iBus in range(n):
            # IDz need fix for bus ID
            real_bus_id = index_to_bus_id[iBus]
            if real_bus_id not in slack:
                #iBus = bus_id_map[real_bus_id]
                row = bus[iBus]
                Vm = row[7]
                Va_deg = row[8]

                # Setting voltage angles and magnitudes according to slack values
                for real_bus_id1 in slack:
                    if converter_type == "polar":
                        if flatStart:
                            file.write(f"\t{V_angle}_{real_bus_id} = {V_angle}_{real_bus_id1}; ")
                            file.write(f"{V_mag}_{real_bus_id} = {V_mag}_{real_bus_id1}\n")
                        else:
                            Va_rad = np.deg2rad(Va_deg)
                            file.write(f"\t{V_angle}_{real_bus_id} = {Va_rad}; ")
                            file.write(f"{V_mag}_{real_bus_id} = {Vm}\n")

                    elif converter_type == "rectangular":
                        if flatStart:
                            file.write(f"\t{e_var}_{real_bus_id} = {e_var}_{real_bus_id1}; ")
                            file.write(f"{f_var}_{real_bus_id} = {f_var}_{real_bus_id1}\n")
                        else:
                            Va_rad = np.deg2rad(Va_deg)
                            Va_real = Vm * np.cos(Va_rad)
                            Va_imag = Vm * np.sin(Va_rad)
                            file.write(f"\t{V_angle}_{real_bus_id} = {Va_real}; ")
                            file.write(f"{V_mag}_{real_bus_id} = {Va_imag}\n")
                    elif converter_type == "complex":
                        if flatStart:
                            file.write(f"\t{v_cplx}_{real_bus_id} = {v_cplx}_{real_bus_id1}\n")
                        else:
                            Va_rad = np.deg2rad(Va_deg)
                            Va_real = Vm * np.cos(Va_rad)
                            Va_imag = Vm * np.sin(Va_rad)
                            V_cmplx = complex(Va_real, Va_imag)
                            file.write(f"\t{v_cplx}_{real_bus_id} = {V_cmplx}\n")

        # Write ZIP model coefficients as parameters if enabled
        file.write("Params:\n")
        print("Writing Params...")
        if loadTransferSelect:
            file.write(f"\tdelta_k = {lt_params['dk']}; k = {lt_params['k']}\n")
            file.write(f"\tdelta_k_gen = {lt_params['dg']}; k_gen = {lt_params['kg']}\n")
        elif loadScaling:
            file.write("\tk = 1.0 // Global load scaling factor\n")
        if genScaling and not loadTransferSelect:
            file.write("\tk_gen = 1.0 // Global gen scaling factor\n")
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
                file.write(f"\t{V_angle}_{real_bus_id} = {v_angle} [out=true]; ")
                file.write(f"\t{V_mag}_{real_bus_id} = {v_slack} [out=true]\n")
            elif converter_type == "rectangular":
                file.write(f"\t{e_var}_{real_bus_id} = {v_slack*np.cos(v_angle)} [out=true]; ")
                file.write(f"\t{f_var}_{real_bus_id} = {v_slack*np.sin(v_angle)} [out=true]\n")
            elif converter_type == "complex":
                file.write(f"\t{v_cplx}_{real_bus_id} = {v_slack}*e^(1i*{v_angle}) [out=true]\n")
        
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
                    #shunt_admittance = (Pd[i_idx] - 1j * Qd[i_idx]) / (Vm[i_idx]**2)
                    shunt_admittance = (Pd[i_idx] - 1j * Qd[i_idx]) / baseMVA   #This is correct
                    Y_ii[i_idx] += shunt_admittance
                    # Update admittance matrix using transformer model
        else: 
            # Subtract the power demands from the generator injections to get net power
            for i_idx in range(n):
                P_inj[i_idx] -= Pd[i_idx]
                Q_inj[i_idx] -= Qd[i_idx]

        all_pq_nodes_with_load = []
        for bus_id in pq_nodes:
            i_idx = bus_id_map[bus_id]
            if P_inj[i_idx] !=0 or Q_inj[i_idx] != 0:
                all_pq_nodes_with_load.append(bus_id)

        if includeConsumptionCurves:
            if numberOfLoadConsumptionCurves > len(all_pq_nodes_with_load): 
                print(f"\nError: Number Of Load Consumption Curves ({numberOfLoadConsumptionCurves}) is larger than the number of available loads ({len(all_pq_nodes_with_load)}).")
                sys.exit(1)
            selected_pq_nodes = random.sample(all_pq_nodes_with_load, int(numberOfLoadConsumptionCurves))
            selected_pq_nodes.sort()

            if pv_nodes and numberOfGenConsumptionCurves > 0:
                k = min(int(numberOfGenConsumptionCurves), len(pv_nodes))
                selected_pv_nodes = random.sample(pv_nodes, k)
                selected_pv_nodes.sort()
            else:
                selected_pv_nodes = []
        
        #print("... split admittance real/imaginary parts ...")
        #G = Y.real  # Conductance matrix
        #B = Y.imag  # Susceptance matrix

        # Write admittance matrix entries as parameters 
        print("... admittance values ...")

        for i_idx in range(n):
            bus_i = index_to_bus_id[i_idx]
            y_ii = Y_ii[i_idx]
            angle_rad = np.angle(y_ii)
            if iConverterType ==0:
                writeYParam(file, y_ii, bus_i, bus_i, Y_mag, Y_angle, tapChangers, phaseShifters, iConverterType)
            elif iConverterType == 2:
                writeYParamCmplx(file, y_ii, bus_i, bus_i, Y_cplx, tapChangers, phaseShifters, iConverterType)
            for j_idx in nodeBranches[i_idx]:
                bus_j = index_to_bus_id[j_idx]
                y_ij = getFromYij(i_idx, j_idx, Y_ij)
                if not y_ij:
                    assert(False)
                if iConverterType ==0:
                    writeYParam(file, y_ij, bus_i, bus_j, Y_mag, Y_angle, tapChangers, phaseShifters, iConverterType)
                elif iConverterType == 2:
                    writeYParamCmplx(file, y_ij, bus_i, bus_j, Y_cplx, tapChangers, phaseShifters, iConverterType)

        # Write active and reactive injections (P_inj, Q_inj) to file
        print("... active and reactive params...")
        for i_idx in range(n):
            bus_id = index_to_bus_id[i_idx]
            if iConverterType == 2:
                # Write S for PQ nodes
                if bus_id in pq_nodes:
                    if P_inj[i_idx] != 0:
                        file.write(f"\tS_{bus_id} = {P_inj[i_idx]}")
                        if Q_inj[i_idx] > 0:
                            file.write(f" + {Q_inj[i_idx]}i")
                        elif Q_inj[i_idx] < 0:
                            file.write(f" {Q_inj[i_idx]}i")
                    if P_inj[i_idx] == 0 and Q_inj[i_idx] != 0:
                        file.write(f"\tS_{bus_id} = {Q_inj[i_idx]}i\n")
                    elif P_inj[i_idx] != 0:
                        file.write(f" \n")
                # Write P and Q for PV nodes
                if bus_id in pv_nodes:
                    file.write(f"\tP_{bus_id}_g = {P_inj[i_idx]}\n")
                    #if calcQOfPVGensInEachIteration:
                    file.write(f"\tQ_{bus_id}_g = {Q_inj[i_idx]} [out = true]\n")      
            elif iConverterType == 0:
                if bus_id in pv_nodes:
                    file.write(f"\tP_{bus_id}_g = {P_inj[i_idx]}\n")
                    if calcQOfPVGensInEachIteration or include_limits or genLimitsSelect:
                        file.write(f"\tQ_{bus_id}_g = {Q_inj[i_idx]}\t[out=true]\n")
                else:
                    if P_inj[i_idx] != 0 or Q_inj[i_idx] != 0:     #Omit only if both zero
                        file.write(f"\tP_{bus_id} = {P_inj[i_idx]}\n")
                        file.write(f"\tQ_{bus_id} = {Q_inj[i_idx]}\n")

        # If enabled, write PV node controls (voltage magnitude setpoint, limits) to file
        print(".... limits, regs ...")
        for pv_bus in pv_nodes:
            if pv_bus in gen_by_bus:
                g = gen_by_bus[pv_bus][0]
                q_min = g[4] / baseMVA
                q_max = g[3] / baseMVA
                if genLimitsSelect and pv_bus in pv_overrides:
                    q_min = pv_overrides[pv_bus]['min']
                    q_max = pv_overrides[pv_bus]['max']
                    file.write(f"\tcGen{pv_bus}Reg=true\n")
                    file.write(f"\tQ_{pv_bus}_g_min = {q_min}\n")
                    file.write(f"\tQ_{pv_bus}_g_max = {q_max}\n")
                elif include_limits and not genLimitsSelect:
                    file.write(f"\tcGen{pv_bus}Reg=true\n")
                    file.write(f"\tQ_{pv_bus}_g_min = {q_min}\n")
                    file.write(f"\tQ_{pv_bus}_g_max = {q_max}\n")
                file.write(f"\tV_{pv_bus}_sp = {g[5]}\n")
        
        print(".... load curves ...")
        if includeConsumptionCurves:
            for bus_id in selected_pq_nodes:
                i_idx = bus_id_map[bus_id]
                if converter_type == "complex":
                    file.write(f"\tk{bus_id}_load = 1 [type=real]\n")
                elif iConverterType == 0:    
                    file.write(f"\tk{bus_id}_load = 1\n")
            for bus_id in selected_pv_nodes:
                i_idx = bus_id_map[bus_id]
                if converter_type == "complex":
                    file.write(f"\tk{bus_id}_g = 1 [type=real]\n")
                elif iConverterType == 0:  
                    file.write(f"\tk{bus_id}_g = 1\n")

        # Begin writing the nonlinear equations (NLEs) section
        file.write("NLEs:\n")
        print("Writing NLEs .....")
        S_mag_by_bus = {}

        iPos = 0
        iTotal = len(pq_nodes)
        deltaInfo = iTotal // 10
        # Loop over all PQ nodes to write power balance equations
        # OVO nije dobro. Jednadzbe se moraju pisati po redoslijedu cvorova
        # petlja mora biti po list bus-ova a onda unutra pitati je li pq ili pv odnosno slack
        # na ovaj nacin se razbija topoloska simetrija i onda je tesko faktorizirati matricu
        # ================> FIXED <===============

        realTerms = []
        imagTerms = []

        for i_idx in range(n):
            bus_id = index_to_bus_id[i_idx]

            if bus_id in slack:
                if comment_equations:
                    file.write(f"\t// node {bus_id} - SLACK\n")
                continue

            adjacentNodes = nodeBranches[i_idx]
            nAdjucent = len(adjacentNodes)
            if nAdjucent == 0:
                print(f"ERROR! Bus {bus_id} is not connected to any other node")
                exit(-1)
            if genLimitsSelect:
                apply_limits_to_this_bus = (bus_id in pv_overrides)
            else:
                apply_limits_to_this_bus = include_limits

            if bus_id in pq_nodes:
                i_idx = bus_id_map[bus_id]
                
                iPos += 1
                if deltaInfo > 0:
                    if iPos % deltaInfo == 0:
                        percent = 10 * iPos // deltaInfo
                        print(f"{percent}%", end=" ")
                    
                # Calculate real and reactive power in system units (MW, MVar)
                p_val = P_inj[i_idx]*baseMVA
                q_val = Q_inj[i_idx]*baseMVA
                s_magnitude = np.sqrt(p_val**2 + q_val**2)  # Apparent power magnitude

                # Write real power balance equation for PQ node
                # NEW PQ (polar)
                if iConverterType == 0: # converter_type == "polar":
                    realTerms = []
                    imagTerms = []          
                    if useSumOfCurrentsForZI and P_inj[i_idx]==0 and Q_inj[i_idx]==0:
                        if comment_equations:
                            file.write(f"\t// node {bus_id} - PQ (ZI)\n")
                        #writeZIRealTermPolar(realTerms, bus_id, bus_id, V_mag, V_angle, Y_mag, Y_angle, True)
                        #writeZIImagTermPolar(imagTerms, bus_id, bus_id, V_mag, V_angle, Y_mag, Y_angle, True)
                        writeDiagPowerTermInPolar(realTerms, bus_id, V_mag, Y_mag, Y_angle, True)
                        writeDiagReactivePowerTermInPolar(imagTerms, bus_id, V_mag, Y_mag, Y_angle, True)

                        realTerms.append(" + ")
                        imagTerms.append(" + ")

                        bFirst = True
                        for j_idx in adjacentNodes:
                            j_bus_id = index_to_bus_id[j_idx]
                            #writeZIRealTermPolar(realTerms, bus_id, j_bus_id, V_mag, V_angle, Y_mag, Y_angle, False)
                            #writeZIImagTermPolar(imagTerms, bus_id, j_bus_id, V_mag, V_angle, Y_mag, Y_angle, False)
                            writeOffDiagPowerTermInPolar(realTerms, bus_id, j_bus_id, V_mag, V_angle, Y_mag, Y_angle, bFirst)
                            writeOffDiagReactivePowerTermInPolar(imagTerms, bus_id, j_bus_id, V_mag, V_angle, Y_mag, Y_angle, bFirst)
                            bFirst = False
                        assert(not bFirst)

                        file.write("".join(realTerms))
                        file.write(" = 0\n")

                        file.write("".join(imagTerms))
                        file.write(" = 0\n")
                    else:
                        if comment_equations:
                            file.write(f"\t// node {bus_id} - PQ\n")
                        # Sum of Powers formulation
                        if useCurrentOnRHS:
                            writeDiagPowerTermInPolar(realTerms, bus_id, V_mag, Y_mag, Y_angle, True)
                            writeDiagReactivePowerTermInPolar(imagTerms, bus_id, V_mag, Y_mag, Y_angle, True)
                            realTerms.append(" + ")
                            imagTerms.append(" + ")
                        else:
                            writeDiagPowerTermInPolar(realTerms, bus_id, V_mag, Y_mag, Y_angle, False)
                            writeDiagReactivePowerTermInPolar(imagTerms, bus_id, V_mag, Y_mag, Y_angle, False)
                            realTerms.append(f" + {V_mag}_{bus_id} * (")
                            imagTerms.append(f" + {V_mag}_{bus_id} * (")
                        bFirst = True
                        for j_idx in adjacentNodes:
                            j_bus_id = index_to_bus_id[j_idx]
                            writeOffDiagPowerTermInPolar(realTerms, bus_id, j_bus_id, V_mag, V_angle, Y_mag, Y_angle, bFirst)
                            writeOffDiagReactivePowerTermInPolar(imagTerms, bus_id, j_bus_id, V_mag, V_angle, Y_mag, Y_angle, bFirst)
                            bFirst = False
                        assert(not bFirst)
                        file.write("".join(realTerms))

                        # Writing RHS
                        if P_inj[i_idx] == 0 and Q_inj[i_idx] == 0:
                            file.write(") = 0\n")
                        else:
                            file.write(f"{'' if useCurrentOnRHS else ')'} = P_{bus_id}")
                            
                            if loadTransferSelect and bus_id in active_pq_set:
                                file.write(f" * k")
                            elif loadScaling and not loadTransferSelect:
                                file.write(f" * k")
                            if zip_coeff:
                                for group_name, data in zip_limits_data.items():
                                    if s_magnitude < data['max']:
                                        Vm = np.array([row[7] for row in bus])
                                        if Vm[i_idx] != 1:
                                            file.write(f" * (Kz_{group_name}*({V_mag}_{bus_id}/{Vm[i_idx]})^2 + Ki_{group_name}*({V_mag}_{bus_id}/{Vm[i_idx]}) + Kp_{group_name}) ")
                                        else:
                                            file.write(f" * (Kz_{group_name}*({V_mag}_{bus_id})^2 + Ki_{group_name}*{V_mag}_{bus_id} + Kp_{group_name}) ")
                                        break
                            if includeConsumptionCurves and bus_id in selected_pq_nodes:
                                file.write(f" * k{bus_id}_load")
                            if useCurrentOnRHS:
                                file.write(f"/{V_mag}_{bus_id}")
                            file.write(f"\n")
                        file.write("".join(imagTerms))
                        if P_inj[i_idx] == 0 and Q_inj[i_idx] == 0:
                            file.write(") = 0\n")
                        else:
                            file.write(f"{'' if useCurrentOnRHS else ')'} = Q_{bus_id}")
                            if loadTransferSelect and bus_id in active_pq_set:
                                file.write(f" * k")
                            elif loadScaling and not loadTransferSelect:
                                file.write(f" * k")
                            if zip_coeff:
                                for group_name, data in zip_limits_data.items():
                                        if s_magnitude < data['max']:
                                            Vm = np.array([row[7] for row in bus])
                                            if Vm[i_idx] != 1:
                                                file.write(f" * (Kz_{group_name}*({V_mag}_{bus_id}/{Vm[i_idx]})^2 + Ki_{group_name}*({V_mag}_{bus_id}/{Vm[i_idx]}) + Kp_{group_name}) ")
                                            else:
                                                file.write(f" * (Kz_{group_name}*({V_mag}_{bus_id})^2 + Ki_{group_name}*{V_mag}_{bus_id} + Kp_{group_name}) ")
                                            break
                            if includeConsumptionCurves and bus_id in selected_pq_nodes:
                                file.write(f" * k{bus_id}_load")
                            if useCurrentOnRHS:
                                file.write(f"/{V_mag}_{bus_id}")
                            file.write(f"\n")
                elif iConverterType == 2: # complex
                    if useSumOfCurrentsForZI and P_inj[i_idx]==0 and Q_inj[i_idx]==0:
                        if comment_equations:
                            file.write(f"\t// node {bus_id} - PQ (ZI)\n")
                        terms = []
                        writeDiagComplexSumTerm(terms, bus_id, v_cplx, Y_cplx)
                        bFirst = False
                        for j_idx in adjacentNodes:
                            j_bus_id = index_to_bus_id[j_idx]
                            writeOffDiagComplexSumTerm(terms, bus_id, j_bus_id, v_cplx, Y_cplx, bFirst)
                        inner_sum = "".join(terms)
                        file.write(f"\tconj({inner_sum})")
                        file.write(" = 0\n")
                        file.write(f"\t{inner_sum}")
                        file.write(" = 0\n")                       
                    else:
                        if comment_equations:
                            file.write(f"\t// node {bus_id} - PQ\n")

                        terms = []
                        writeDiagComplexSumTerm(terms, bus_id, v_cplx, Y_cplx)
                        bFirst = False
                        for j_idx in adjacentNodes:
                            j_bus_id = index_to_bus_id[j_idx]
                            writeOffDiagComplexSumTerm(terms, bus_id, j_bus_id, v_cplx, Y_cplx, bFirst)
                        inner_sum = "".join(terms)
                        if useCurrentOnRHS:
                            file.write(f"\tconj({inner_sum})")
                        else:
                            file.write(f"\t{v_cplx}_{bus_id} * conj({inner_sum})")
                        

                        if P_inj[i_idx] == 0 and Q_inj[i_idx] == 0:
                                file.write(" = 0\n")
                        else:
                            file.write(f" = S_{bus_id}")
                            if loadTransferSelect and bus_id in active_pq_set:
                                file.write(f" * k")
                            elif loadScaling and not loadTransferSelect:
                                file.write(f" * k")
                            if zip_coeff:
                                for group_name, data in zip_limits_data.items():
                                    if s_magnitude < data['max']:
                                        Vm = np.array([row[7] for row in bus])
                                        if Vm[i_idx] != 1:
                                            file.write(f" * (Kz_{group_name}*({v_cplx}_{bus_id} * conj({v_cplx}_{bus_id})/{Vm[i_idx]}^2) + Ki_{group_name}*(sqrt({v_cplx}_{bus_id} * conj({v_cplx}_{bus_id}))/{Vm[i_idx]}) + Kp_{group_name})")
                                        else:
                                            file.write(f" * (Kz_{group_name}*({v_cplx}_{bus_id} * conj({v_cplx}_{bus_id})) + Ki_{group_name}*sqrt({v_cplx}_{bus_id} * conj({v_cplx}_{bus_id})) + Kp_{group_name})")                                       
                                        break
                            if includeConsumptionCurves and bus_id in selected_pq_nodes:
                                file.write(f" * k{bus_id}_load")
                            if useCurrentOnRHS:
                                file.write(f"/{v_cplx}_{bus_id}")
                            file.write(f"\n")

                        if useCurrentOnRHS:
                            file.write(f"\t{inner_sum}")
                        else:
                            file.write(f"\tconj({v_cplx}_{bus_id}) * ({inner_sum})")
                        if P_inj[i_idx] == 0 and Q_inj[i_idx] == 0:
                                file.write(" = 0\n")
                        else:
                            file.write(f" = conj(S_{bus_id})")
                            if loadTransferSelect and bus_id in active_pq_set:
                                file.write(f" * k")
                            elif loadScaling and not loadTransferSelect:
                                file.write(f" * k")
                            if zip_coeff:
                                for group_name, data in zip_limits_data.items():
                                    if s_magnitude < data['max']:
                                        Vm = np.array([row[7] for row in bus])
                                        if Vm[i_idx] != 1:
                                            file.write(f" * (Kz_{group_name}*({v_cplx}_{bus_id} * conj({v_cplx}_{bus_id})/{Vm[i_idx]}^2) + Ki_{group_name}*(sqrt({v_cplx}_{bus_id} * conj({v_cplx}_{bus_id}))/{Vm[i_idx]}) + Kp_{group_name})")
                                        else:
                                            file.write(f" * (Kz_{group_name}*({v_cplx}_{bus_id} * conj({v_cplx}_{bus_id})) + Ki_{group_name}*sqrt({v_cplx}_{bus_id} * conj({v_cplx}_{bus_id})) + Kp_{group_name})")
                                        break
                            if includeConsumptionCurves and bus_id in selected_pq_nodes:
                                file.write(f" * k{bus_id}_load")
                            if useCurrentOnRHS:
                                file.write(f"/conj({v_cplx}_{bus_id})")
                            file.write(f"\n")
            # Loop over all PV nodes to write real power balance and voltage control
            else: #bus_id in pv_nodes: must be PV if it's neither Alack nor PQ
                i = bus_id
                if comment_equations:
                    file.write(f"\t// node {i} - PV\n")
                i_idx = bus_id_map[i]

                # Write real power balance equation for PV node
                # NEW PV (polar)
                if iConverterType == 0: #converter_type == "polar":
                    # Sum of Powers formulation
                    realTerms = []
                    writeDiagPowerTermInPolar(realTerms, bus_id, V_mag, Y_mag, Y_angle, False)
                    realTerms.append(f" + {V_mag}_{bus_id} * (")
                    bFirst = True
                    for j_idx in adjacentNodes:
                        j_bus_id = index_to_bus_id[j_idx]
                        writeOffDiagPowerTermInPolar(realTerms, bus_id, j_bus_id, V_mag, V_angle, Y_mag, Y_angle, bFirst)
                        bFirst = False
                    assert(not bFirst)
                    file.write("".join(realTerms))
                    if P_inj[i_idx] == 0:
                        file.write(") = 0\n")
                    else:
                        file.write(f") = P_{bus_id}_g")
                        if loadTransferSelect and bus_id in active_pv_set:
                            file.write(f" * k_gen")
                        elif genScaling and not loadTransferSelect:
                             file.write(f" * k_gen")
                        if includeConsumptionCurves and i in selected_pv_nodes:
                            file.write(f" * k{bus_id}_g")
                        file.write(f"\n")

                    # Including regulation
                    if apply_limits_to_this_bus:
                        file.write(f"\tif cGen{bus_id}Reg:\n")
                        file.write(f"\t\t{V_mag}_{bus_id}={V_mag}_{bus_id}_sp\n")
                        file.write("\telse:\n")
                        imagTerms = []
                        imagTerms.append(f"\t")
                        writeDiagReactivePowerTermInPolar(imagTerms, bus_id, V_mag, Y_mag, Y_angle, False)
                        imagTerms.append(f" + {V_mag}_{bus_id} * (")
                        bFirst = True
                        for j_idx in adjacentNodes:
                            j_bus_id = index_to_bus_id[j_idx]
                            writeOffDiagReactivePowerTermInPolar(imagTerms, bus_id, j_bus_id, V_mag, V_angle, Y_mag, Y_angle, bFirst)
                            bFirst = False
                        assert(not bFirst)
                        
                        file.write("".join(imagTerms))
                        if Q_inj[i_idx] == 0:
                            file.write(") = 0\n")
                        else:
                            ''' # treba li scaling na Q kada je PV
                            if loadTransferSelect and bus_id in active_pv_set:
                                file.write(f") = Q_{bus_id}_g * k_gen\n")
                            else:
                            '''
                            file.write(f") = Q_{bus_id}_g\n")
                        
                    if apply_limits_to_this_bus:
                        file.write("\tend\n")
                    else:
                        file.write(f"\t{V_mag}_{bus_id}={V_mag}_{bus_id}_sp\n")
                elif iConverterType == 2:
                    terms = []
                    writeDiagComplexSumTerm(terms, bus_id, v_cplx, Y_cplx)
                    bFirst = False
                    for j_idx in adjacentNodes:
                        j_bus_id = index_to_bus_id[j_idx]
                        writeOffDiagComplexSumTerm(terms, bus_id, j_bus_id, v_cplx, Y_cplx, bFirst)
                    inner_sum = "".join(terms)
                    file.write(f"\t{v_cplx}_{bus_id} * conj({inner_sum}) + conj({v_cplx}_{bus_id}) * ({inner_sum})")
                    if P_inj[i_idx] == 0:
                            file.write(" = 0\n")
                    else:                       
                        file.write(f" = 2 * P_{bus_id}_g")
                        if loadTransferSelect and bus_id in active_pv_set:
                            file.write(f" * k_gen")
                        elif genScaling and not loadTransferSelect:
                             file.write(f" * k_gen")
                        if includeConsumptionCurves and i in selected_pv_nodes:
                            file.write(f" * k{bus_id}_g")
                        file.write(f"\n")

                    if apply_limits_to_this_bus:
                        file.write(f"\tif cGen{bus_id}Reg:\n")
                        file.write(f"\t\t{v_cplx}_{bus_id} * conj({v_cplx}_{bus_id}) = {V_mag}_{bus_id}_sp^2\n")
                        file.write("\telse:\n")
                        terms = []
                        writeDiagComplexSumTerm(terms, bus_id, v_cplx, Y_cplx)
                        bFirst = False
                        for j_idx in adjacentNodes:
                            j_bus_id = index_to_bus_id[j_idx]
                            writeOffDiagComplexSumTerm(terms, bus_id, j_bus_id, v_cplx, Y_cplx, bFirst)
                        inner_sum = "".join(terms)
                        file.write(f"\t\t{v_cplx}_{bus_id} * conj({inner_sum}) - conj({v_cplx}_{bus_id}) * ({inner_sum})")                       
                        if Q_inj[i_idx] == 0:
                            file.write(" = 0\n")
                        else:
                            file.write(f" = 2i * Q_{bus_id}_g\n")


                    if apply_limits_to_this_bus:
                        file.write("\tend\n")
                    else:
                        file.write(f"\t{v_cplx}_{bus_id} * conj({v_cplx}_{bus_id}) = {V_mag}_{bus_id}_sp^2\n")
                    
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
        target_buses = pv_nodes

        # Only iterate over PV nodes that have generation attached
        for bus_id in target_buses:
            if genLimitsSelect and bus_id not in pv_overrides:
                continue
            if bus_id in pmax_by_bus:
                pmax_val = pmax_by_bus[bus_id]
                
                # Find the correct group for this bus based on its Pmax
                for name, max_val_config in group_names_sorted:
                    if pmax_val <= max_val_config:
                        group_to_buses[name].append(bus_id)
                        break

        # Voltage/reactive power limits for PV buses (if enabled)
        if include_limits or genLimitsSelect: 
            
            if calcQOfPVGensInEachIteration:
                # IterPostP for default reactive power calculation with limits
                file.write(f"IterPostP:\n")
                for group_name, bus_list in group_to_buses.items():
                    if not bus_list:
                        continue
                    for bus_id in bus_list:
                        i_idx = bus_id_map[bus_id]
                        adjacentNodes = nodeBranches[i_idx] 
                        file.write(f"\tQ_{bus_id}_g = ")
                        if iConverterType == 0:
                            imagTerms = []
                            writeDiagReactivePowerTermInPolar(imagTerms, bus_id, V_mag, Y_mag, Y_angle, False)
                            imagTerms.append(f" + {V_mag}_{bus_id} * (")
                            bFirst = True
                            for j_idx in adjacentNodes:
                                j_bus_id = index_to_bus_id[j_idx]
                                writeOffDiagReactivePowerTermInPolar(imagTerms, bus_id, j_bus_id, V_mag, V_angle, Y_mag, Y_angle, bFirst)
                                bFirst = False
                            assert(not bFirst)
                            file.write("".join(imagTerms))
                            file.write(")\n")
                        if iConverterType == 2:
                            terms = []
                            writeDiagComplexSumTerm(terms, bus_id, v_cplx, Y_cplx)
                            bFirst = False
                            for j_idx in adjacentNodes:
                                j_bus_id = index_to_bus_id[j_idx]
                                writeOffDiagComplexSumTerm(terms, bus_id, j_bus_id, v_cplx, Y_cplx, bFirst)
                            inner_sum = "".join(terms)
                            file.write(f"\timag({v_cplx}_{bus_id} * conj({inner_sum}))\n")
            
            # Write constraints for limits in the model file
            file.write("Limits:\n")
            for group_name, bus_list in group_to_buses.items():
                if not bus_list:
                    continue

                file.write(f"\tgroup [name=\"{group_name}\" enabled=true]:\n")
                for bus_id in bus_list:
                    file.write(f"\t\tif cGen{bus_id}Reg:\n")
                    i_idx = bus_id_map[bus_id]
                    adjacentNodes = nodeBranches[i_idx] 
                    # Write reactive power in limits, not the default setting
                    if not calcQOfPVGensInEachIteration:
                        if iConverterType == 0:
                            i_idx = bus_id_map[bus_id]
                            file.write(f"\t\t\tQ_{bus_id}_g = ")
                            imagTerms = []
                            writeDiagReactivePowerTermInPolar(imagTerms, bus_id, V_mag, Y_mag, Y_angle, False)
                            imagTerms.append(f" + {V_mag}_{bus_id} * (")
                            bFirst = True
                            for j_idx in adjacentNodes:
                                j_bus_id = index_to_bus_id[j_idx]
                                writeOffDiagReactivePowerTermInPolar(imagTerms, bus_id, j_bus_id, V_mag, V_angle, Y_mag, Y_angle, bFirst)
                                bFirst = False
                            assert(not bFirst)
                            file.write("".join(imagTerms))
                            file.write(")\n")
                        if iConverterType == 2:
                            terms = []
                            writeDiagComplexSumTerm(terms, bus_id, v_cplx, Y_cplx)
                            bFirst = False
                            for j_idx in adjacentNodes:
                                j_bus_id = index_to_bus_id[j_idx]
                                writeOffDiagComplexSumTerm(terms, bus_id, j_bus_id, v_cplx, Y_cplx, bFirst)
                            inner_sum = "".join(terms)
                            file.write(f"\t\t\tQ_{bus_id}_g = imag({v_cplx}_{bus_id} * conj({inner_sum}))\n")

                    # Check and enforce lower limit
                    file.write(f"\t\t\tif Q_{bus_id}_g<=Q_{bus_id}_g_min [signal=TooLow]:\n")
                    file.write(f"\t\t\t\t cGen{bus_id}Reg=false\n")
                    file.write(f"\t\t\t\t Q_{bus_id}_g=Q_{bus_id}_g_min\n")

                    # Check and enforce upper limit
                    file.write("\t\t\telse:\n")
                    file.write(f"\t\t\t\tif Q_{bus_id}_g>=Q_{bus_id}_g_max [signal=TooHigh]:\n")
                    file.write(f"\t\t\t\t\t cGen{bus_id}Reg=false\n")
                    file.write(f"\t\t\t\t\t Q_{bus_id}_g=Q_{bus_id}_g_max\n")

                    file.write("\t\t\t\tend\n")
                    file.write("\t\t\tend\n")
                    file.write(f"\t\tend\n")
                file.write(f"\tend\n")
        
        # Add PostProc if complex domain to calculate Qinj
        if iConverterType == 2:
            file.write("PostProc:\n")
            for group_name, bus_list in group_to_buses.items():
                if not bus_list:
                    continue
                
                for bus_id in bus_list:
                    i_idx = bus_id_map[bus_id]
                    adjacentNodes = nodeBranches[i_idx]
                    nAdjucent = len(adjacentNodes)
                    if nAdjucent == 0:
                        print(f"ERROR! Bus {bus_id} is not connected to any other node")
                        exit(-1)
                    terms = []
                    writeDiagComplexSumTerm(terms, bus_id, v_cplx, Y_cplx)
                    bFirst = False
                    for j_idx in adjacentNodes:
                        j_bus_id = index_to_bus_id[j_idx]
                        writeOffDiagComplexSumTerm(terms, bus_id, j_bus_id, v_cplx, Y_cplx, bFirst)
                    inner_sum = "".join(terms)
                    file.write(f"\tQ_{bus_id}_g = imag({v_cplx}_{bus_id} * conj({inner_sum}))\n")

        if loadTransferSelect:
            file.write("Repeats:\n")
            file.write("\tk += delta_k\n")
            file.write("\tk_gen += delta_k_gen\n")
            file.write("\trepeat\n")

        file.write("end\n")

    print("\nConversion successful!")
    print(f"Output written to: {resFileName}")

# Standard entry point for a Python script
if __name__ == "__main__":
    main()