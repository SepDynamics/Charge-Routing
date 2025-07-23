# Spiral Network Analysis

This note describes a simplified resistor-network model for the
spiral geometry. The script `/_sep/testbed/spiral_network.py` builds a
square spiral as a graph and solves for node potentials when a DC
source is connected between the endpoints. A cross connection is added
at the midpoint to mimic the proposed role-flipping behavior.

Example usage:

```bash
python _sep/testbed/spiral_network.py
```

The program prints the coordinates of each node along with its computed
potential. This provides a basic sanity check that the geometry evenly
spreads voltage when the midpoint cross-link is present.

For further exploration, `/_sep/testbed/spiral_sweep.py` performs a
parameter sweep over different spiral sizes. It reports the standard
deviation of node potentials with and without the midpoint cross
connection to quantify uniformity.
