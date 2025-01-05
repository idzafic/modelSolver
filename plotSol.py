#!/usr/bin/env python3
#(c) Izudin Dzafic, 2023
#add chmod +x plotSol.py
import sys
import numpy as np
import matplotlib.pyplot as plt

args = len(sys.argv)-1
if args != 1:
    print('Missing input file name!')
    sys.exit()
    
# set dark background
plt.style.use('dark_background')
# Read the file and extract the relevant lines
lines = []
header = []
iRows = 0
with open(sys.argv[1], 'r') as file:
    foundSolutionData = False
    headerLoaded = False
    firstHeaderLine = False
    secondHeaderLine = False
    for line in file:
        if foundSolutionData and firstHeaderLine and not headerLoaded:
            header = line.strip().split()
            headerLoaded = True
            print('Header is loaded')
        elif foundSolutionData:
            if not firstHeaderLine:
                if line.startswith('------'):
                    firstHeaderLine = True
                    continue
                else:
                    print('File is not formated properly! First line (--------) is not available!')
                    sys.exit()
            if not secondHeaderLine:       
                if line.startswith('------'):
                    secondHeaderLine = True
                    continue
                else:
                    print('File is not formated properly! Second line (--------) is not available!')
                    sys.exit()             
            if line.startswith('----'):
                print('Detected end of SOLUTION_DATA')
                break
            lines.append(line)
            iRows += 1
        elif line.strip() == 'SOLUTION_DATA':
            foundSolutionData = True
            print('Detected SOLUTION_DATA')

if not headerLoaded:
    print('ERROR! Could not detect header')
    sys.exit()

if iRows == 0:
    print('ERROR! Could not load any usable line with data')
    sys.exit()
    
# Parse the data
data = np.loadtxt(lines, dtype=float)

# Check if data is empty
if data.size == 0:
    print("No data found in the file.")
else:
    # Extract the columns
    t = data[:, 0]
    columns = data[:, 1:].T
    
    # Plot the data
    for i, column in enumerate(columns):
        plt.plot(t, column, label=header[i+1])

    # Add labels and legend
    if len(header) <= 11:
        xLbl = ''
        yLbl = ''
        k=0
        for lbl in header:
            if k>1:
                yLbl += ', ' + lbl.strip()
            elif k>0:
                yLbl = lbl.strip()
            else:
                xLbl = lbl.strip()
            k=k+1
        plt.xlabel(xLbl)
        plt.ylabel(yLbl)     
    else:
        plt.xlabel('Time (t) [s]')
        plt.ylabel('Multiple Values')
        
    plt.legend()
    
    # show grid
    plt.grid(linestyle='dotted', linewidth=1)
    
    # Display the plot
    plt.show()
