"""
Visualization tools for spiral network analysis.
Demonstrates various ways to visualize the electrical properties of spiral networks.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
import networkx as nx
from scipy.interpolate import griddata
import seaborn

from spiral_network import build_spiral_graph, solve_potentials
from spiral_impedance import total_resistance
from spiral_transient import build_rc_network, rc_odes
from scipy.integrate import solve_ivp


def visualize_spiral_geometry(n_turns=4, cross=True, save_path=None):
    """
    Visualize the spiral network geometry with potential distribution.
    """
    G = build_spiral_graph(n_turns=n_turns, cross=cross)
    potentials = solve_potentials(G)
    
    # Extract positions and potentials
    pos = nx.get_node_attributes(G, 'pos')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw edges
    for edge in G.edges():
        x = [pos[edge[0]][0], pos[edge[1]][0]]
        y = [pos[edge[0]][1], pos[edge[1]][1]]
        ax.plot(x, y, 'k-', linewidth=2, alpha=0.6)
    
    # Draw nodes with potential-based colors
    node_colors = [potentials[node] for node in G.nodes()]
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                   cmap='viridis', node_size=500,
                                   vmin=0, vmax=1, ax=ax)
    
    # Add node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    
    # Highlight cross-connection if present
    if cross:
        mid_nodes = [n for n in G.nodes() if 'cross' in str(G.nodes[n])]
        if len(mid_nodes) >= 2:
            ax.plot([pos[mid_nodes[0]][0], pos[mid_nodes[1]][0]],
                   [pos[mid_nodes[0]][1], pos[mid_nodes[1]][1]],
                   'r--', linewidth=3, label='Cross-connection')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Potential (V)')
    
    plt.title(f'Spiral Network Potential Distribution\n'
              f'({n_turns} turns, cross-connection: {"Yes" if cross else "No"})')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_resistance_scaling(max_turns=8, save_path=None):
    """
    Plot how resistance scales with the number of turns.
    """
    turns_range = range(2, max_turns + 1)
    resistance_no_cross = []
    resistance_with_cross = []
    
    for turns in turns_range:
        # Without cross-connection
        G_no_cross = build_spiral_graph(n_turns=turns, cross=False)
        R_no_cross = total_resistance(G_no_cross)
        resistance_no_cross.append(R_no_cross)
        
        # With cross-connection
        G_cross = build_spiral_graph(n_turns=turns, cross=True)
        R_cross = total_resistance(G_cross)
        resistance_with_cross.append(R_cross)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data
    ax.plot(turns_range, resistance_no_cross, 'bo-', linewidth=2, 
            markersize=8, label='Without cross-connection')
    ax.plot(turns_range, resistance_with_cross, 'ro-', linewidth=2,
            markersize=8, label='With cross-connection')
    
    # Add theoretical quadratic fit
    turns_array = np.array(list(turns_range))
    quadratic_fit = turns_array**2 * (resistance_no_cross[0] / 4)  # Normalize to first point
    ax.plot(turns_range, quadratic_fit, 'k--', alpha=0.5, 
            label='Quadratic fit (R ∝ n²)')
    
    # Shade the resistance reduction region
    ax.fill_between(turns_range, resistance_no_cross, resistance_with_cross,
                    alpha=0.2, color='green', label='Resistance reduction')
    
    ax.set_xlabel('Number of Turns')
    ax.set_ylabel('Resistance (Ω)')
    ax.set_title('Spiral Network Resistance Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(turns_range)
    
    # Add percentage reduction annotations
    for i, turns in enumerate(turns_range):
        reduction = (1 - resistance_with_cross[i] / resistance_no_cross[i]) * 100
        ax.annotate(f'-{reduction:.0f}%', 
                   xy=(turns, (resistance_no_cross[i] + resistance_with_cross[i]) / 2),
                   ha='center', fontsize=8, color='green')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_potential_heatmap(n_turns=4, cross=True, resolution=100, save_path=None):
    """
    Create a 2D heatmap of potential distribution with contour lines.
    """
    G = build_spiral_graph(n_turns=n_turns, cross=cross)
    potentials = solve_potentials(G)
    pos = nx.get_node_attributes(G, 'pos')
    
    # Extract coordinates and potentials
    x_coords = [pos[node][0] for node in G.nodes()]
    y_coords = [pos[node][1] for node in G.nodes()]
    z_values = [potentials[node] for node in G.nodes()]
    
    # Create grid for interpolation
    x_min, x_max = min(x_coords) - 1, max(x_coords) + 1
    y_min, y_max = min(y_coords) - 1, max(y_coords) + 1
    
    xi = np.linspace(x_min, x_max, resolution)
    yi = np.linspace(y_min, y_max, resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Interpolate potential values
    zi = griddata((x_coords, y_coords), z_values, (xi_grid, yi_grid), method='cubic')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    im = ax.imshow(zi, extent=[x_min, x_max, y_min, y_max], 
                   origin='lower', cmap='viridis', aspect='equal')
    
    # Add contour lines
    contours = ax.contour(xi_grid, yi_grid, zi, levels=10, colors='white', 
                         alpha=0.4, linewidths=1)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
    
    # Overlay the network
    for edge in G.edges():
        x = [pos[edge[0]][0], pos[edge[1]][0]]
        y = [pos[edge[0]][1], pos[edge[1]][1]]
        ax.plot(x, y, 'k-', linewidth=2, alpha=0.8)
    
    # Plot nodes
    ax.scatter(x_coords, y_coords, c=z_values, cmap='viridis', 
              s=100, edgecolors='black', linewidth=2, zorder=5)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Potential (V)')
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Potential Distribution Heatmap\n'
                f'({n_turns} turns, cross-connection: {"Yes" if cross else "No"})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_current_flow(n_turns=4, cross=True, save_path=None):
    """
    Visualize current flow through the network with arrows.
    """
    G = build_spiral_graph(n_turns=n_turns, cross=cross)
    potentials = solve_potentials(G)
    pos = nx.get_node_attributes(G, 'pos')
    
    # Calculate currents
    currents = {}
    for u, v in G.edges():
        r = G[u][v].get('resistance', 1.0)
        current = (potentials[u] - potentials[v]) / r
        currents[(u, v)] = abs(current)
    
    # Normalize currents for visualization
    max_current = max(currents.values())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw edges with thickness proportional to current
    for edge, current in currents.items():
        u, v = edge
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        
        # Line thickness based on current
        linewidth = 0.5 + 4 * (current / max_current)
        alpha = 0.3 + 0.7 * (current / max_current)
        
        ax.plot(x, y, 'b-', linewidth=linewidth, alpha=alpha)
        
        # Add arrow to show direction
        if potentials[u] > potentials[v]:
            arrow_start = pos[u]
            arrow_end = pos[v]
        else:
            arrow_start = pos[v]
            arrow_end = pos[u]
        
        # Calculate arrow position (middle of edge)
        arrow_x = (arrow_start[0] + arrow_end[0]) / 2
        arrow_y = (arrow_start[1] + arrow_end[1]) / 2
        dx = arrow_end[0] - arrow_start[0]
        dy = arrow_end[1] - arrow_start[1]
        
        # Scale arrow size by current
        arrow_scale = 0.2 * (current / max_current)
        
        if current > 0.1 * max_current:  # Only show arrows for significant currents
            ax.arrow(arrow_x - dx * arrow_scale/2, arrow_y - dy * arrow_scale/2,
                    dx * arrow_scale, dy * arrow_scale,
                    head_width=0.15, head_length=0.1, fc='red', ec='red', alpha=0.7)
    
    # Draw nodes
    node_colors = [potentials[node] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                          cmap='viridis', node_size=300,
                          vmin=0, vmax=1, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    
    # Add colorbar for potential
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Node Potential (V)')
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Current Flow Visualization\n'
                f'({n_turns} turns, cross-connection: {"Yes" if cross else "No"})\n'
                f'Arrow size ∝ current magnitude')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_comparison_plot(n_turns=4, save_path=None):
    """
    Create side-by-side comparison of spiral with and without cross-connection.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    for ax, cross, title_suffix in [(ax1, False, "Without Cross-connection"),
                                     (ax2, True, "With Cross-connection")]:
        G = build_spiral_graph(n_turns=n_turns, cross=cross)
        potentials = solve_potentials(G)
        pos = nx.get_node_attributes(G, 'pos')
        
        # Calculate resistance
        R = total_resistance(G)
        
        # Draw network
        for edge in G.edges():
            x = [pos[edge[0]][0], pos[edge[1]][0]]
            y = [pos[edge[0]][1], pos[edge[1]][1]]
            ax.plot(x, y, 'k-', linewidth=2, alpha=0.6)
        
        # Draw nodes
        node_colors = [potentials[node] for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                              cmap='viridis', node_size=500,
                              vmin=0, vmax=1, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
        
        ax.set_title(f'{title_suffix}\nR = {R:.2f} Ω')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
    
    # Add shared colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], label='Potential (V)', 
                       orientation='horizontal', pad=0.1)
    
    plt.suptitle(f'Spiral Network Comparison ({n_turns} turns)', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_3d_surface_plot(n_turns=4, cross=True, save_path=None):
    """
    Create a 3D surface plot of the potential distribution.
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    G = build_spiral_graph(n_turns=n_turns, cross=cross)
    potentials = solve_potentials(G)
    pos = nx.get_node_attributes(G, 'pos')
    
    # Extract coordinates and potentials
    x_coords = [pos[node][0] for node in G.nodes()]
    y_coords = [pos[node][1] for node in G.nodes()]
    z_values = [potentials[node] for node in G.nodes()]
    
    # Create grid for interpolation
    resolution = 50
    x_min, x_max = min(x_coords) - 1, max(x_coords) + 1
    y_min, y_max = min(y_coords) - 1, max(y_coords) + 1
    
    xi = np.linspace(x_min, x_max, resolution)
    yi = np.linspace(y_min, y_max, resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Interpolate potential values
    zi = griddata((x_coords, y_coords), z_values, (xi_grid, yi_grid), method='cubic')
    
    # Create 3D figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(xi_grid, yi_grid, zi, cmap='viridis',
                          alpha=0.8, linewidth=0, antialiased=True)
    
    # Plot network edges in 3D
    for edge in G.edges():
        x = [pos[edge[0]][0], pos[edge[1]][0]]
        y = [pos[edge[0]][1], pos[edge[1]][1]]
        z = [potentials[edge[0]], potentials[edge[1]]]
        ax.plot(x, y, z, 'k-', linewidth=2, alpha=0.9)
    
    # Plot nodes
    ax.scatter(x_coords, y_coords, z_values, c=z_values, cmap='viridis',
              s=100, edgecolors='black', linewidth=2)
    
    # Labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Potential (V)')
    ax.set_title(f'3D Potential Surface\n({n_turns} turns, '
                f'cross-connection: {"Yes" if cross else "No"})')
    
    # Colorbar
    fig.colorbar(surf, ax=ax, label='Potential (V)', shrink=0.5)
    
    # Set viewing angle
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_statistical_dashboard(max_turns=8, save_path=None):
    """
    Create a comprehensive statistical analysis dashboard.
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Collect data
    data = {'turns': [], 'cross': [], 'resistance': [], 'uniformity': []}
    
    for turns in range(2, max_turns + 1):
        for cross in [False, True]:
            G = build_spiral_graph(n_turns=turns, cross=cross)
            R = total_resistance(G)
            potentials = solve_potentials(G)
            uniformity = np.std(list(potentials.values()))
            
            data['turns'].append(turns)
            data['cross'].append('Yes' if cross else 'No')
            data['resistance'].append(R)
            data['uniformity'].append(uniformity)
    
    # 1. Resistance vs Turns scatter plot
    ax1 = plt.subplot(2, 3, 1)
    for cross_type in ['Yes', 'No']:
        mask = [c == cross_type for c in data['cross']]
        turns = [t for t, m in zip(data['turns'], mask) if m]
        resistance = [r for r, m in zip(data['resistance'], mask) if m]
        ax1.scatter(turns, resistance, label=f'Cross: {cross_type}', s=100)
    ax1.set_xlabel('Number of Turns')
    ax1.set_ylabel('Resistance (Ω)')
    ax1.set_title('Resistance vs. Turns')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Uniformity box plot
    ax2 = plt.subplot(2, 3, 2)
    uniformity_data = []
    labels = []
    for cross_type in ['No', 'Yes']:
        mask = [c == cross_type for c in data['cross']]
        uniformity_data.append([u for u, m in zip(data['uniformity'], mask) if m])
        labels.append(f'Cross: {cross_type}')
    ax2.boxplot(uniformity_data, labels=labels)
    ax2.set_ylabel('Potential Std Dev')
    ax2.set_title('Uniformity Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 3. Resistance reduction percentage
    ax3 = plt.subplot(2, 3, 3)
    turns_unique = sorted(set(data['turns']))
    reductions = []
    for t in turns_unique:
        r_no_cross = [r for r, turns, cross in zip(data['resistance'], 
                      data['turns'], data['cross']) if turns == t and cross == 'No'][0]
        r_cross = [r for r, turns, cross in zip(data['resistance'], 
                   data['turns'], data['cross']) if turns == t and cross == 'Yes'][0]
        reduction = (1 - r_cross / r_no_cross) * 100
        reductions.append(reduction)
    ax3.bar(turns_unique, reductions, color='green', alpha=0.7)
    ax3.set_xlabel('Number of Turns')
    ax3.set_ylabel('Resistance Reduction (%)')
    ax3.set_title('Cross-connection Impact')
    ax3.grid(True, alpha=0.3)
    
    # 4. Correlation heatmap
    ax4 = plt.subplot(2, 3, 4)
    # Convert categorical to numerical for correlation
    cross_numeric = [1 if c == 'Yes' else 0 for c in data['cross']]
    corr_data = np.array([data['turns'], cross_numeric, 
                         data['resistance'], data['uniformity']]).T
    corr_matrix = np.corrcoef(corr_data.T)
    im = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax4.set_xticks(range(4))
    ax4.set_yticks(range(4))
    ax4.set_xticklabels(['Turns', 'Cross', 'Resistance', 'Uniformity'], rotation=45)
    ax4.set_yticklabels(['Turns', 'Cross', 'Resistance', 'Uniformity'])
    ax4.set_title('Parameter Correlation Matrix')
    
    # Add correlation values
    for i in range(4):
        for j in range(4):
            ax4.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                    ha='center', va='center')
    
    # 5. Resistance scaling fit
    ax5 = plt.subplot(2, 3, 5)
    no_cross_data = [(t, r) for t, r, c in zip(data['turns'], 
                     data['resistance'], data['cross']) if c == 'No']
    turns_no_cross = [d[0] for d in no_cross_data]
    res_no_cross = [d[1] for d in no_cross_data]
    
    # Fit quadratic
    coeffs = np.polyfit(turns_no_cross, res_no_cross, 2)
    fit_turns = np.linspace(min(turns_no_cross), max(turns_no_cross), 100)
    fit_res = np.polyval(coeffs, fit_turns)
    
    ax5.scatter(turns_no_cross, res_no_cross, s=100, label='Data')
    ax5.plot(fit_turns, fit_res, 'r--', label=f'Fit: R = {coeffs[0]:.2f}n² + {coeffs[1]:.2f}n + {coeffs[2]:.2f}')
    ax5.set_xlabel('Number of Turns')
    ax5.set_ylabel('Resistance (Ω)')
    ax5.set_title('Quadratic Scaling (No Cross)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary statistics table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')
    
    # Calculate summary stats
    avg_reduction = np.mean(reductions)
    avg_uniformity_cross = np.mean([u for u, c in zip(data['uniformity'], 
                                    data['cross']) if c == 'Yes'])
    avg_uniformity_no_cross = np.mean([u for u, c in zip(data['uniformity'], 
                                      data['cross']) if c == 'No'])
    
    summary_data = [
        ['Metric', 'Value'],
        ['Average Resistance Reduction', f'{avg_reduction:.1f}%'],
        ['Avg Uniformity (with cross)', f'{avg_uniformity_cross:.4f}'],
        ['Avg Uniformity (no cross)', f'{avg_uniformity_no_cross:.4f}'],
        ['Resistance Scaling Coefficient', f'{coeffs[0]:.2f}'],
        ['Max Turns Analyzed', str(max_turns)]
    ]
    
    table = ax6.table(cellText=summary_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax6.set_title('Summary Statistics', pad=20)
    
    plt.suptitle('Spiral Network Statistical Analysis Dashboard', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def animate_transient_response(n_turns=3, cross=True, duration=2.0, save_path=None):
    """
    Create an animation of the transient response.
    """
    # Build RC network and solve transient
    G = build_rc_network(n_turns=n_turns, cross=cross)
    src = 0
    sink = max(G.nodes)
    n = G.number_of_nodes()
    v0 = np.zeros(n)
    v0[src] = 1.0
    
    # Solve ODE
    t_span = [0, duration]
    t_eval = np.linspace(0, duration, 50)
    sol = solve_ivp(rc_odes, t_span, v0, args=(G, src, sink, 1.0),
                    t_eval=t_eval, max_step=0.05)
    
    # Get positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Initialize plot elements
    for edge in G.edges():
        x = [pos[edge[0]][0], pos[edge[1]][0]]
        y = [pos[edge[0]][1], pos[edge[1]][1]]
        ax.plot(x, y, 'k-', linewidth=2, alpha=0.6)
    
    # Initial node plot
    nodes = ax.scatter([pos[n][0] for n in G.nodes()],
                      [pos[n][1] for n in G.nodes()],
                      c=v0, cmap='viridis', s=500, vmin=0, vmax=1,
                      edgecolors='black', linewidth=2)
    
    # Add labels
    for node in G.nodes():
        ax.text(pos[node][0], pos[node][1], str(node),
               ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(nodes, ax=ax, label='Potential (V)')
    
    # Title
    title = ax.set_title(f'Transient Response Animation (t=0.00s)\n'
                        f'({n_turns} turns, cross-connection: {"Yes" if cross else "No"})')
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    # Animation update function
    def update(frame):
        voltages = sol.y[:, frame]
        nodes.set_array(voltages)
        title.set_text(f'Transient Response Animation (t={sol.t[frame]:.2f}s)\n'
                      f'({n_turns} turns, cross-connection: {"Yes" if cross else "No"})')
        return nodes, title
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(sol.t),
                                  interval=100, blit=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=10)
    
    plt.tight_layout()
    plt.show()
    
    return anim


if __name__ == "__main__":
    # Example usage - generate all visualizations
    print("Generating spiral network visualizations...")
    
    # 1. Basic geometry visualization
    print("1. Creating geometry visualization...")
    visualize_spiral_geometry(n_turns=4, cross=True, save_path="spiral_geometry.png")
    
    # 2. Resistance scaling plot
    print("2. Creating resistance scaling plot...")
    plot_resistance_scaling(max_turns=8, save_path="resistance_scaling.png")
    
    # 3. Potential heatmap
    print("3. Creating potential heatmap...")
    create_potential_heatmap(n_turns=4, cross=True, save_path="potential_heatmap.png")
    
    # 4. Current flow visualization
    print("4. Creating current flow visualization...")
    visualize_current_flow(n_turns=4, cross=True, save_path="current_flow.png")
    
    # 5. Comparison plot
    print("5. Creating comparison plot...")
    create_comparison_plot(n_turns=4, save_path="spiral_comparison.png")
    
    # 6. 3D surface plot
    print("6. Creating 3D surface plot...")
    create_3d_surface_plot(n_turns=4, cross=True, save_path="potential_3d.png")
    
    # 7. Statistical dashboard
    print("7. Creating statistical dashboard...")
    create_statistical_dashboard(max_turns=8, save_path="stats_dashboard.png")
    
    # 8. Transient animation (saves as GIF)
    print("8. Creating transient response animation...")
    animate_transient_response(n_turns=3, cross=True, duration=2.0, 
                              save_path="transient_animation.gif")
    
    print("\nAll visualizations generated successfully!")
    print("Check the current directory for the output files.")