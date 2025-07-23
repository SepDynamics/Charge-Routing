# Transient RC Simulation

The script `/_sep/testbed/spiral_transient.py` approximates the spiral as an RC network.
Each edge has a unit resistance and capacitance. The solver uses `scipy.integrate.solve_ivp`
to watch how node potentials equalize after a step input.

Run the default simulation with:

```bash
python _sep/testbed/spiral_transient.py
```

It prints the final potential at each node after one second. Use `--turns` and
`--duration` arguments to explore different geometries and time scales.
