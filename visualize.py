# visualize.py
import matplotlib.pyplot as plt
import networkx as nx
import random
from matplotlib.lines import Line2D

def plot_network(coords, nodes, customers, charging_stations, depot):
    G = nx.Graph()
    for i in nodes:
        G.add_node(i, pos=coords[i])
    
    for i in nodes:
        possible_neighbors = [j for j in nodes if j != i]
        num_edges = random.randint(3, min(5, len(possible_neighbors)))
        neighbors = random.sample(possible_neighbors, num_edges)
        for j in neighbors:
            G.add_edge(i, j)

    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(15, 15))
    
    nx.draw_networkx_nodes(G, pos, nodelist=[depot], node_color='green', node_size=200, label='Depot')
    nx.draw_networkx_nodes(G, pos, nodelist=customers, node_color='blue', node_size=50, label='Customers')
    nx.draw_networkx_nodes(G, pos, nodelist=charging_stations, node_color='red', node_size=150, label='Charging Stations')
    
    nx.draw_networkx_edges(G, pos, alpha=0.1)
    
    plt.title("EVRP Network (3-5 Edges per Node)")
    plt.legend()
    plt.savefig("network.png")
    plt.close()

def plot_routes(coords, routes, pickup, delivery):
    G = nx.DiGraph()
    colors = ['purple', 'orange', 'cyan', 'magenta', 'yellow', 'pink', 'lime', 'brown', 'gray', 'olive']
    styles = ['-', '--', '-.', ':']
    
    # Add all edges for each route, including start and end at depot
    for k, route in enumerate(routes):
        color = colors[k % len(colors)]
        style = styles[k % len(styles)]
        full_route = [0] + route  # Prepend depot since route starts at 0
        for i in range(len(full_route) - 1):
            G.add_edge(full_route[i], full_route[i + 1], color=color, style=style, label=f"V{k}")

    pos = {i: coords[i] for i in coords.keys()}
    plt.figure(figsize=(20, 20))
    
    # Draw all nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=100)
    nx.draw_networkx_nodes(G, pos, nodelist=[0], node_color='green', node_size=200)
    
    charging_stations_in_routes = [n for n in [201, 202, 203, 204] if n in G.nodes()]
    if charging_stations_in_routes:
        nx.draw_networkx_nodes(G, pos, nodelist=charging_stations_in_routes, node_color='red', node_size=150)
    
    # Draw all edges with their respective colors and styles
    legend_handles = []
    for k, route in enumerate(routes):
        color = colors[k % len(colors)]
        style = styles[k % len(styles)]
        full_route = [0] + route
        edgelist = [(full_route[i], full_route[i + 1]) for i in range(len(full_route) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=edgelist, edge_color=color, style=style, arrows=True)
        legend_handles.append(Line2D([0], [0], color=color, linestyle=style, label=f"Vehicle {k}"))

    # Add labels with node ID, pickup, and delivery (default to 0 if not in pickup/delivery)
    labels = {i: f"{i}: P={pickup.get(i, 0)}, D={delivery.get(i, 0)}" for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black')

    plt.title("All Vehicle Routes with Pickup and Delivery")
    plt.legend(handles=legend_handles, loc='upper right')
    plt.savefig("all_routes.png")
    plt.close()