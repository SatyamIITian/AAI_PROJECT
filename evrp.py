# evrp.py
import math
from sklearn.cluster import KMeans
import time

class EVRP:
    def __init__(self, filename, use_two_opt=True):
        self.use_two_opt = use_two_opt
        self._load_from_file(filename)
        self.distances = {i: {j: self._calculate_distance(i, j) for j in self.nodes} for i in self.nodes}
        self.nearest_station = {i: min(self.charging_stations, key=lambda s: self.distances[i][s]) for i in self.nodes}

    def _load_from_file(self, filename):
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        params = {}
        for line in lines[:6]:
            key, value = line.split(': ')
            params[key] = float(value) if '.' in value else int(value)
        
        self.num_vehicles = params['NUM_VEHICLES']
        self.W = params['WEIGHT_CAPACITY']
        self.E = params['ENERGY_CAPACITY']
        self.E_min = params['ENERGY_MIN']
        self.alpha = params['ENERGY_RATE']
        self.r = params['CHARGE_RATE']

        self.coords = {}
        self.nodes = []
        self.customers = []
        self.charging_stations = []
        self.depot = None
        
        i = 6
        depot_data = lines[i].split()
        self.depot = int(depot_data[1])
        self.nodes.append(self.depot)
        self.coords[self.depot] = (float(depot_data[2]), float(depot_data[3]))
        i += 1
        
        num_customers = int(lines[i].split()[1])
        i += 1
        for j in range(num_customers):
            node_data = lines[i + j].split()
            node_id = int(node_data[0])
            self.customers.append(node_id)
            self.nodes.append(node_id)
            self.coords[node_id] = (float(node_data[1]), float(node_data[2]))
        i += num_customers
        
        num_stations = int(lines[i].split()[1])
        i += 1
        for j in range(num_stations):
            station_data = lines[i + j].split()
            station_id = int(station_data[0])
            self.charging_stations.append(station_id)
            self.nodes.append(station_id)
            self.coords[station_id] = (float(station_data[1]), float(station_data[2]))
        
        self.pickup = {}
        self.delivery = {}
        i += num_stations
        assert lines[i] == "PICKUP:"
        i += 1
        for j in range(len(self.nodes)):
            demand_data = lines[i + j].split()
            node_id = int(demand_data[0])
            self.pickup[node_id] = int(demand_data[1])
        
        i += len(self.nodes)
        assert lines[i] == "DELIVERY:"
        i += 1
        for j in range(len(self.nodes)):
            demand_data = lines[i + j].split()
            node_id = int(demand_data[0])
            self.delivery[node_id] = int(demand_data[1])

    def _calculate_distance(self, i, j):
        x1, y1 = self.coords[i]
        x2, y2 = self.coords[j]
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _energy_consumed(self, dist, weight):
        return self.alpha * dist * max(weight, 0)

    def _cluster_customers(self):
        customer_coords = [self.coords[i] for i in self.customers]
        kmeans = KMeans(n_clusters=self.num_vehicles, random_state=42).fit(customer_coords)
        labels = kmeans.labels_
        clusters = [[] for _ in range(self.num_vehicles)]
        for i, label in enumerate(labels):
            clusters[label].append(self.customers[i])
        return clusters

    def _two_opt(self, route, distances):
        best_route = route[:]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    if j - i == 1:
                        continue
                    new_route = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
                    old_dist = (distances[best_route[i-1]][best_route[i]] + 
                                distances[best_route[j-1]][best_route[j]])
                    new_dist = (distances[new_route[i-1]][new_route[i]] + 
                                distances[new_route[j-1]][new_route[j]])
                    if new_dist < old_dist:
                        best_route = new_route
                        improved = True
        return best_route

    def solve(self):
        start_time = time.time()
        clusters = self._cluster_customers()
        
        routes = [[] for _ in range(self.num_vehicles)]
        energies = [self.E for _ in range(self.num_vehicles)]
        weights = [0 for _ in range(self.num_vehicles)]
        distances = [0 for _ in range(self.num_vehicles)]
        energy_consumed = [0 for _ in range(self.num_vehicles)]
        charging_events = [[] for _ in range(self.num_vehicles)]
        current_nodes = [self.depot for _ in range(self.num_vehicles)]

        energy_threshold = self.E * 0.5  # Recharge at 50% capacity

        for k in range(self.num_vehicles):
            print(f"Processing Vehicle {k + 1}/{self.num_vehicles} (Cluster size: {len(clusters[k])})")
            unvisited = set(clusters[k])
            while unvisited:
                current = current_nodes[k]
                nearest = min(unvisited, key=lambda x: self.distances[current][x], default=None)
                if nearest is None:
                    break
                
                dist = self.distances[current][nearest]
                print(f"  Vehicle {k + 1}: Visiting node {nearest} from {current} (Energy: {energies[k]:.2f})")
                if weights[k] >= self.delivery[nearest]:
                    new_weight = weights[k] - self.delivery[nearest] + self.pickup[nearest]
                else:
                    new_weight = self.pickup[nearest]

                if new_weight > self.W:
                    dist_to_depot = self.distances[current][self.depot]
                    energy_needed = self._energy_consumed(dist_to_depot, weights[k])
                    if energies[k] - energy_needed < self.E_min:
                        nearest_station = self.nearest_station[current]
                        dist_to_station = self.distances[current][nearest_station]
                        energy_to_station = self._energy_consumed(dist_to_station, weights[k])
                        if energies[k] > energy_to_station:
                            routes[k].append(nearest_station)
                            energy_consumed[k] += energy_to_station
                            energies[k] -= energy_to_station
                            distances[k] += dist_to_station
                            charge_amount = self.E - energies[k]
                            charging_events[k].append((nearest_station, charge_amount, charge_amount / self.r, charge_amount / self.E * 100))
                            energies[k] = self.E
                            current = nearest_station
                            print(f"  Vehicle {k + 1}: Charging at {nearest_station} (Energy: {energies[k]:.2f})")
                    routes[k].append(self.depot)
                    distances[k] += dist_to_depot
                    energy_consumed[k] += energy_needed
                    energies[k] -= energy_needed
                    weights[k] = 0
                    current_nodes[k] = self.depot
                    print(f"  Vehicle {k + 1}: Returning to depot (Energy: {energies[k]:.2f})")
                    continue

                energy_needed = self._energy_consumed(dist, weights[k])
                if energies[k] - energy_needed < energy_threshold:
                    nearest_station = self.nearest_station[current]
                    dist_to_station = self.distances[current][nearest_station]
                    energy_to_station = self._energy_consumed(dist_to_station, weights[k])
                    if energies[k] > energy_to_station:
                        routes[k].append(nearest_station)
                        energy_consumed[k] += energy_to_station
                        energies[k] -= energy_to_station
                        distances[k] += dist_to_station
                        charge_amount = self.E - energies[k]
                        charging_events[k].append((nearest_station, charge_amount, charge_amount / self.r, charge_amount / self.E * 100))
                        energies[k] = self.E
                        current_nodes[k] = nearest_station
                        print(f"  Vehicle {k + 1}: Charging at {nearest_station} (Energy: {energies[k]:.2f})")
                        continue

                routes[k].append(nearest)
                energy_consumed[k] += energy_needed
                energies[k] -= energy_needed
                distances[k] += dist
                weights[k] = new_weight
                current_nodes[k] = nearest
                unvisited.remove(nearest)

        for k in range(self.num_vehicles):
            current = current_nodes[k]
            dist = self.distances[current][self.depot]
            energy_needed = self._energy_consumed(dist, weights[k])
            if energies[k] - energy_needed < self.E_min:
                nearest_station = self.nearest_station[current]
                dist_to_station = self.distances[current][nearest_station]
                energy_to_station = self._energy_consumed(dist_to_station, weights[k])
                if energies[k] > energy_to_station:
                    routes[k].append(nearest_station)
                    energy_consumed[k] += energy_to_station
                    energies[k] -= energy_to_station
                    distances[k] += dist_to_station
                    charge_amount = self.E - energies[k]
                    charging_events[k].append((nearest_station, charge_amount, charge_amount / self.r, charge_amount / self.E * 100))
                    energies[k] = self.E
                    print(f"  Vehicle {k + 1}: Final charge at {nearest_station} (Energy: {energies[k]:.2f})")
            routes[k].append(self.depot)
            distances[k] += dist
            energy_consumed[k] += energy_needed
            print(f"  Vehicle {k + 1}: Returning to depot (Energy: {energies[k]:.2f})")

        if self.use_two_opt:
            for k in range(self.num_vehicles):
                print(f"Optimizing Vehicle {k + 1}/{self.num_vehicles}")
                full_route = [self.depot] + routes[k]
                depot_indices = [i for i, x in enumerate(full_route) if x == self.depot]
                optimized_route = []
                for i in range(len(depot_indices) - 1):
                    start = depot_indices[i]
                    end = depot_indices[i + 1]
                    segment = full_route[start:end + 1]
                    if len(segment) > 3:
                        optimized_segment = self._two_opt(segment, self.distances)
                        optimized_route.extend(optimized_segment[:-1])
                    else:
                        optimized_route.extend(segment[:-1])
                optimized_route.append(self.depot)
                routes[k] = optimized_route[1:]

                distances[k] = 0
                energy_consumed[k] = 0
                energies[k] = self.E
                weights[k] = 0
                charging_events[k] = []
                current = self.depot
                for next_node in routes[k]:
                    dist = self.distances[current][next_node]
                    if next_node in self.customers:
                        if weights[k] >= self.delivery[next_node]:
                            new_weight = weights[k] - self.delivery[next_node] + self.pickup[next_node]
                        else:
                            new_weight = self.pickup[next_node]
                    else:
                        new_weight = weights[k]
                    
                    energy_needed = self._energy_consumed(dist, weights[k])
                    if next_node in self.charging_stations:
                        charge_amount = self.E - energies[k]
                        if charge_amount > 0:
                            charging_events[k].append((next_node, charge_amount, charge_amount / self.r, charge_amount / self.E * 100))
                        energies[k] = self.E
                    elif energies[k] - energy_needed < energy_threshold:
                        nearest_station = self.nearest_station[current]
                        dist_to_station = self.distances[current][nearest_station]
                        energy_to_station = self._energy_consumed(dist_to_station, weights[k])
                        if energies[k] > energy_to_station:
                            routes[k].insert(routes[k].index(next_node), nearest_station)
                            distances[k] += dist_to_station
                            energy_consumed[k] += energy_to_station
                            energies[k] -= energy_to_station
                            charging_events[k].append((nearest_station, self.E - energies[k], (self.E - energies[k]) / self.r, (self.E - energies[k]) / self.E * 100))
                            energies[k] = self.E
                            continue
                    else:
                        energies[k] -= energy_needed
                        energy_consumed[k] += energy_needed
                        weights[k] = new_weight
                    
                    distances[k] += dist
                    current = next_node

        print(f"Solve time: {time.time() - start_time:.2f} seconds")
        return routes, distances, energy_consumed, charging_events

    def get_coords(self):
        return self.coords

    def get_nodes(self):
        return self.nodes, self.customers, self.charging_stations, self.depot

    def get_pickup(self):
        return self.pickup
    
    def get_delivery(self):
        return self.delivery