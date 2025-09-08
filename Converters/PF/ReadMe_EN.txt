This Python script converts MATPOWER .m case files into a dTwin .dmodl format.
The configuration and behavior of the script is controlled via an XML config.xml file.
Make sure that both config.xml and greek_symbols.json are placed in the same directory as the Python script.

Python 3.7 or higher is recommended for full compatibility.
Required Libraries:
	- numpy (pip install numpy if not installed)
	- xml.etree.ElementTree (part of standard library)
	- json (part of standard library)
	- re (part of standard library)
	- argparse (part of standard library)
    - os (part of standard library)
    - sys (part of standard library)

Usage:
	-Prepare the XML configuration file specifying your MATPOWER .m input file and script options.
		1.) File Section:
			In this section users can specify th path (full or relative) to the original MATPOWER input file (.m file) and the base name
			for the output file that the script will generate in the current folder.
		2.) Options Section:
			Ability to include generator reactive power limits.
			Adding comments to equations and parameters for better readability.
			Enabling or disabling ZIP load modeling.
			Optionally setting all loads to zero for testing purposes.
			Using constant ZIP coefficients (with Kp=1) across all categories.
		3.) Variables Section:
			This section allows the user to define variable names and choose their formatting styles for the generated code,
			including voltage magnitudes, voltage angles, and line admittance parameters. Users can specify whether to use 
			symbolic names (symbol) or plain text identifiers (name). The list of available symbols is defined 
			in the greek_symbols.json file.
		4.) Limits Section:
			This section defines power categories for classifying generators by their maximum power. Categories should be 
			ordered by increasing maximum value, with the last category typically set to unlimited (max="inf").
		5.) ZIP Limits Section:
			Defines categories for ZIP load models including coefficients Kz, Ki, Kp parts of the load. These coefficients
			must sum to 1 for each category, and categories must be sorted by increasing maximum load size, with the last
			category typically set to unlimited (max="inf").
	-Run the Python script, which reads the XML config, parses the MATPOWER file, and generates the output accordingly.
	
To run the script from the terminal, type the following command where "X" is the number of the case file, where case file is in cases folder:
    >>> python matp2modl.py caseX.m

To run the script with a custom path, type the following command where "X" is the number of the case file:
    >>> python matp2modl.py full/path/to/caseX.m

You can also specify a custom name for the output file:
    >>> python matp2modl.py caseX.m -o nameOfOutputFile

If you would like just to specify output folder (while reusing input file name with changed extension):
    >>> python matp2modl.py caseX.m -resPath=./res
or
    >>> python matp2modl.py caseX.m --r=./res
In this case, you can provide any available folder as the resulting output folder.

For a full explanation of all available options, use the help flag:
    >>> python matp2modl.py --help


