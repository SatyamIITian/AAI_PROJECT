# main.py
from evrp import EVRP
from visualize import plot_network, plot_routes

def main():
    evrp = EVRP("input.evrp", use_two_opt=False)
    routes, distances, energy_consumed, charging_events = evrp.solve()
    
    for k in range(len(routes)):
        print(f"\nVehicle {k} Route:")
        print(f"Path: {' -> '.join(map(str, [0] + routes[k]))}")
        print(f"Distance Traveled: {distances[k]:.2f} km")
        print(f"Energy Consumed: {energy_consumed[k]:.2f} units")
        print(f"Charging Events: {len(charging_events[k])}")
        for idx, (station, amount, time, percent) in enumerate(charging_events[k]):
            print(f"  Charge {idx+1} at Station {station}: {amount:.2f} units, {time:.2f} mins, {percent:.2f}%")

    coords = evrp.get_coords()
    nodes, customers, charging_stations, depot = evrp.get_nodes()
    pickup = evrp.get_pickup()
    delivery = evrp.get_delivery()
    plot_network(coords, nodes, customers, charging_stations, depot)
    plot_routes(coords, routes, pickup, delivery)

if __name__ == "__main__":
    main()