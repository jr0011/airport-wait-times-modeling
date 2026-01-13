# Airport Security Screening Simulation

A discrete-event simulation (DES) model built with SimPy to analyze passenger wait times at airport security checkpoints.

## Overview

This project simulates airport security screening queues using real passenger data from `airport.csv`. The model validates against theoretical queueing theory (M/G/1) and compares different operational scenarios.

## Features

- **Validation**: Compares simulation results against Pollaczek-Khintchine (P-K) formula
- **Scenario Analysis**: Tests three configurations:
  - **Baseline**: 1 server, σ = 0.25 min
  - **Option A**: 2 servers, σ = 0.25 min  
  - **Option B**: 1 server, σ = 0.10 min (reduced variability)
- **Statistical Testing**: Welch's t-test for scenario comparisons
- **Visualization**: Comprehensive plots including histograms, confidence intervals, violin plots, and Gantt charts

## Requirements
```bash
pip install simpy numpy matplotlib pandas scipy seaborn
```

## Usage
```bash
python simulation.py
```

Outputs saved to root directory as PNG files and LaTeX tables.

## Files

- `airport-wait-times-modeling.py` - Main simulation code
- `airport.csv` - Historical passenger data
- `scenario_comparison.tex` -  Key results
- `results_plots` - Generated plots and tables
- `README.md`

This project was done in collaboration with Tarreau Bone and Sary Abou.