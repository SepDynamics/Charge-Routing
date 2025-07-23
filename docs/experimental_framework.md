# Experimental Framework

This document outlines potential experiments to validate the charge routing concept.

## Finite Element Simulation
- Model the spiral conductor with alternating layers.
- Use open-source solvers (e.g. FEniCS) to study potential distribution.
- Observe current density uniformity across the geometry.

## Electrorheological Fluid Tests
- Prototype simple electrorheological (ER) fluid using common materials (e.g. cornstarch + oil with conductive particles).
- Apply voltage threshold to observe chain formation.
- Measure transition voltage where conductivity increases.

## Small-Scale Physical Prototype
- Build a planar spiral using copper tape on an insulating substrate.
- Introduce right-angle corners and mid-point role-flipping connections.
- Submerge the assembly in an ER fluid chamber to observe self-balancing behavior.

## Data Collection
- Record current flow and voltage drop across spiral segments.
- Capture thermal imaging to identify hotspots or dendrite formation.

Future iterations can move from the testbed into formal simulation or hardware.

## Testbed Scripts
- Initial geometry prototypes are placed in `/_sep/testbed/`.
- Run `python spiral_demo.py` to generate sample spiral coordinates.
- Run `python spiral_network.py` to evaluate a simple resistor-network
  model of the spiral geometry.
- Run `python spiral_sweep.py` to assess potential uniformity across
  different spiral sizes and cross-link configurations.
