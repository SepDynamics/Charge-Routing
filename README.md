# Charge Routing Testbed

This folder contains early-stage experiments exploring recursive spiral geometries and dielectric fluids.
Each script should include usage instructions and note any observations.

## Scripts

- `spiral_demo.py` – generates coordinates for a simple square spiral.
- `spiral_network.py` – solves a resistor-network model of the spiral
  with a midpoint cross-link.
- `spiral_sweep.py` – sweeps over turn counts and reports potential
  uniformity with and without the midpoint cross connection.
- `spiral_stats.py` – computes average uniformity from the sweep
  results.
- `spiral_transient.py` – simulates transient RC behavior of the spiral.
- `spiral_impedance.py` – calculates effective resistance for spirals with varying turns and cross-connection options.

## Experimental Results

### Impedance Analysis

The impedance scan shows how effective resistance scales with the number of spiral turns:

| Turns | Cross-Connection | Resistance (Ω) |
|-------|-----------------|----------------|
| 2     | No              | 6.0000         |
| 2     | Yes             | 3.7500         |
| 3     | No              | 12.0000        |
| 3     | Yes             | 6.8571         |
| 4     | No              | 20.0000        |
| 4     | Yes             | 10.9091        |
| 5     | No              | 30.0000        |
| 5     | Yes             | 15.9375        |
| 6     | No              | 42.0000        |
| 6     | Yes             | 21.9545        |

**Key Observations:**
- Resistance increases quadratically with the number of turns (R ≈ n²) for spirals without cross-connections
- Cross-connections reduce resistance by approximately 40-50%
- The resistance reduction from cross-connections becomes more significant as the spiral grows

### Uniformity Statistics

Potential uniformity analysis across the spiral network:

- **Average standard deviation with cross-connection:** 0.3322
- **Average standard deviation without cross-connection:** 0.3083

The cross-connection slightly increases potential variation, but provides better current distribution and lower overall resistance.

### Transient Response

The RC transient simulation shows voltage distribution after steady-state is reached:

| Node | X Position | Y Position | Voltage |
|------|------------|------------|---------|
| 0    | 0          | 0          | 1.000   |
| 1    | 1.0        | 0          | 0.476   |
| 2    | 1.0        | 1.0        | 0.169   |
| 3    | 0.0        | 1.0        | 0.054   |
| 4    | -1.0       | 1.0        | 0.049   |
| 5    | -1.0       | 0.0        | 0.140   |
| 6    | -1.0       | -1.0       | 0.377   |
| 7    | 0.0        | -1.0       | 0.138   |
| 8    | 1.0        | -1.0       | 0.039   |
| 9    | 2.0        | -1.0       | 0.009   |
| 10   | 2.0        | 0.0        | 0.002   |
| 11   | 2.0        | 1.0        | 0.000   |
| 12   | 2.0        | 2.0        | 0.000   |

This shows the voltage decay from the source (node 0) to the sink (node 12) through the spiral path.

## Visualization Suggestions

To better demonstrate the technology and results, consider creating the following visualizations:

### 1. Spiral Geometry Visualization
- **2D plot** showing the spiral path with nodes and edges
- Color-code nodes by their potential values
- Highlight the cross-connection when present
- Animate the construction process turn by turn

### 2. Resistance Scaling Plot
- **Line graph** showing resistance vs. number of turns
- Two lines: with and without cross-connection
- Include theoretical quadratic fit to show R ∝ n² relationship
- Add shaded region showing the resistance reduction from cross-connections

### 3. Potential Distribution Heatmap
- **2D heatmap** overlaid on the spiral geometry
- Use color gradient to show voltage distribution
- Side-by-side comparison: with vs. without cross-connection
- Include equipotential contour lines

### 4. Current Flow Visualization
- **Vector field** showing current direction and magnitude on each edge
- Arrow thickness proportional to current magnitude
- Highlight current concentration areas
- Show how cross-connections redistribute current flow

### 5. Transient Response Animation
- **Time-series animation** showing voltage propagation
- Start with initial conditions and animate to steady-state
- Use color gradient for voltage levels
- Include time slider and play/pause controls

### 6. 3D Surface Plot
- **3D surface** where height represents voltage at each position
- Interpolate between node positions for smooth surface
- Rotate view to show voltage "landscape"
- Compare surfaces with/without cross-connections

### 7. Interactive Parameter Explorer
- **Web-based tool** with sliders for:
  - Number of turns
  - Resistance values
  - Capacitance values
  - Cross-connection on/off
- Real-time update of impedance and voltage distribution
- Export results as CSV or images

### 8. Statistical Analysis Dashboard
- **Multi-panel display** showing:
  - Impedance vs. turns scatter plot with trend lines
  - Uniformity metrics box plots
  - Histogram of node potentials
  - Correlation matrix of parameters

### Implementation Tools
Consider using these libraries for visualization:
- **Python**: matplotlib, seaborn, plotly, networkx for graph visualization
- **Web**: D3.js for interactive visualizations, Three.js for 3D plots
- **Animation**: matplotlib.animation, manim for educational animations

### Example Visualization Code Structure
```python
# Example: Basic spiral visualization with potential heatmap
import matplotlib.pyplot as plt
import networkx as nx
from spiral_network import build_spiral_graph, solve_potentials

def visualize_spiral_potential(n_turns=4, cross=True):
    G = build_spiral_graph(n_turns=n_turns, cross=cross)
    potentials = solve_potentials(G)
    
    # Extract positions and potentials
    pos = nx.get_node_attributes(G, 'pos')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw network with potential-based colors
    nx.draw_networkx(G, pos, node_color=list(potentials.values()),
                     cmap='viridis', node_size=500, with_labels=True)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis')
    sm.set_array(list(potentials.values()))
    plt.colorbar(sm, ax=ax, label='Potential (V)')
    
    plt.title(f'Spiral Network Potential Distribution\n'
              f'({n_turns} turns, cross-connection: {cross})')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
```

These visualizations will help demonstrate:
- The geometric structure of the spiral networks
- How electrical properties scale with size
- The impact of cross-connections on performance
- Dynamic behavior during transient conditions
- Parameter sensitivity and optimization opportunities
