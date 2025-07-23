# Parameter Sweep Results

This note records sample output from the testbed scripts.

Running `python _sep/testbed/spiral_sweep.py` with up to six turns produces the following table:

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

Using `python _sep/testbed/spiral_stats.py` gives average standard deviation values:

```
Average stddev with cross: 0.3322
Average stddev without cross: 0.3083
```

These numbers provide a baseline for comparing future refinements of the geometry or solver accuracy.

## Effective Resistance Scan

Running `python _sep/testbed/spiral_impedance.py` measures the total resistance between the spiral's ends.

```
turns   cross   resistance
2       0       6.0000
2       1       3.7500
3       0       12.0000
3       1       6.8571
4       0       20.0000
4       1       10.9091
5       0       30.0000
5       1       15.9375
6       0       42.0000
6       1       21.9545
```
