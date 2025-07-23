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

Example sweep results using `spiral_sweep.py` for up to six turns:

```
turns   cross   stddev
2       0       0.3333
2       1       0.3489
3       0       0.3118
3       1       0.3334
4       0       0.3028
4       1       0.3282
5       0       0.2981
5       1       0.3260
6       0       0.2955
6       1       0.3248
```

The helper script `/_sep/testbed/spiral_stats.py` computes averages from
these sweeps:

```
Average stddev with cross: 0.3322
Average stddev without cross: 0.3083
```

While the cross connection slightly increases the spread in this simple
model, the tooling demonstrates how different geometries can be rapidly
evaluated.
