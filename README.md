
# Model Solver Framework



The Model Solver is a powerful tool designed to make it easier for domain experts to create and deploy Digital Twins without needing extensive software development knowledge. It allows users to define complex models using human-readable mathematical formulations, and then automatically generates all necessary components for application deployment. This tool is versatile and can handle any system described mathematically, producing customized data structures for AI, ML, and DL applications.

By utilizing advanced numerical methods like Newton-Raphson solvers with symbolic differentiation, the framework ensures high precision and stability in simulations. It also includes an implicit solver based on Butcher’s tables for solving differential-algebraic equations (DAEs), providing excellent stability and accuracy. The framework supports various modeling approaches, including nonlinear state-space and transfer function models, making it ideal for dynamic systems in fields like automation and control.

With Model-Driven Development (MDD), domain experts can focus on refining their models, while the tool handles the complexities of code generation, including loops, conditions, and pointer management. The Model Solver is implemented in C++, supports large-scale systems with sparse matrices, and leverages SIMD on Windows, Linux, and macOS (including Intel and Apple Silicon).

Note: Version 2.0 introduces new modeling paradigm. XML modeling is available in previous versions. See [Releases](https://github.com/idzafic/modelSolver/releases) section.

## Install

1. Download the repository from the main branch:
   - The `bin` folder is reserved for binaries and needs to be populated with the appropriate release files depending on your operating system.
   - The `models` folder contains example models in XML format.
   - The `plotSol.py` script can be used to plot results for DAE and ODE simulations.

2. Download the binaries for your operating system from the [Releases](https://github.com/idzafic/modelSolver/releases) section.

3. Unpack the binaries into the corresponding subfolder within the `bin` folder.
   
**Note:** On some Windows systems, the Model Solver may not work due to missing C++ runtime dependencies. 

To resolve this issue, you need to install the Microsoft Visual Studio C++ Runtime Package for 64-bit systems. It can be downloaded from the following links:
- [Direct download link](https://aka.ms/vs/17/release/vc_redist.x64.exe)
- [Microsoft documentation](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#visual-studio-2015-2017-2019-and-2022)



## Usage 

A) (XML models - ver 2.x.x and above):

```bash
modelSolver inputFileName
```
where:
- `inputFileName`: is the input file name in XML format.

All model details must be specified in the header. Txt output (results) will be generated in the same folder with inputFileName and will txt extension.

New model details can be found [here] (https://www.arxiv.org/abs/2508.17882) with all examples provided in foler models/PowerSystem/PaperExamples. It includes MATPOWER to dmodl conversion for Power Flow and State Estimation.


B) (XML models - ver 1.x.x):

```bash
modelSolver modelType domain inputFileName outputFileName [t0] [dT] [t1]
```
where:

- `modelType` can be:
    - `ODE` for solving Ordinary Differential Equations
    - `DAE` for solving Differential Algebraic Equations
    - `NR` for solving a set of nonlinear equations using the Newton-Raphson method
    - `WLS` for solving Weighted Least Squares (EC) state estimation problems
    - `DIFF` for testing differentiation.
  
- `domain` can be:
    - `real` for problems defined in the real domain (R)
    - `cmplx` for problems defined in the complex domain (C)
  
- `inputFileName`: is the input file name in XML format.

- `outputFileName`: is the output (result) file name in TXT format.

- `t0`: is required for ODE and DAE. Ignored by other methods. Represents the starting time of the simulation.
  
- `dT`: is required for ODE and DAE. Ignored by other methods. Represents the time increment of the simulation.
  
- `t1`: is required for ODE and DAE. Ignored by other methods. Represents the final time of the simulation.

## Example (DAE problem, real domain, Windows x64):

Replace 'PATH_TO_ModelSolver' with your Model Solver repository location and 'PATH_TO_OUTPUT_FOLDER' with your desired output directory: Then:

```bash
PATH_TO_ModelSolver/bin/modelSolver DAE real PATH_TO_ModelSolver/models/DAE/ACGenWith1LoadMechLimitAndInitialProblem.xml PATH_TO_OUTPUT_FOLDER/ACGen1.txt 0 0.01 20
```
This command generates results in ACGen1.txt within your specified output folder.
To visualize the results, use the provided plotting script:
```bash
python PATH_TO_ModelSolver/plotSol.py PATH_TO_OUTPUT_FOLDER/ACGen1.txt.
```
This would generate the following image:

![plot](https://github.com/user-attachments/assets/4d306138-986a-4cae-9f32-2d9fab9d32cc)

### Note: plotSol.py requires Python packages 'numpy' and 'matplotlib'. These two packages can be installed usin pip:
```bash
pip3 install numpy
pip3 install matplotlib
```
