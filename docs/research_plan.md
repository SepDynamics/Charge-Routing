# Research Plan

This document proposes next steps for evaluating the self-regulating charge routing concept.

## Objectives
- Validate that a spiral conductor with recursive role-flipping promotes uniform charge distribution.
- Quantify the effectiveness of the voltage-sensitive dielectric fluid.
- Explore small-scale fabrication options for a passive balancing network.

## Proposed Experiments
1. **Finite Element Simulation**
   - Use FEniCS or a similar solver to model potential distribution in a multi-layer spiral.
   - Examine current density uniformity and compare against simple geometries.
2. **Electrorheological Fluid Tests**
   - Prototype a basic ER fluid with cornstarch or silica particles in oil.
   - Measure conductivity versus applied field strength to determine threshold behavior.
3. **Bench Prototype**
   - Etch or 3D-print a planar spiral with midpoint cross-links.
   - Submerge the device in the ER fluid chamber and log current flow across segments.
4. **Parameter Sweep**
   - Extend the `_sep/testbed/` scripts to vary spiral dimensions and cross-link positions.
   - Record uniformity metrics and impedance changes.

## Environment Setup
Run `./install.sh` for the basic Python environment. For simulation work, execute `./install_research.sh` to install additional tools such as FEniCS and gmsh.

All prototypes should be placed under `/_sep/testbed/` and summarized in this `docs/` directory.
