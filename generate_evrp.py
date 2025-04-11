# generate_evrp.py
import random

def generate_evrp_file(filename="input.evrp"):
    random.seed(42)
    with open(filename, 'w') as f:
        # Parameters
        f.write("# Parameters\n")
        f.write("NUM_VEHICLES: 10\n")
        f.write("WEIGHT_CAPACITY: 150\n")
        f.write("ENERGY_CAPACITY: 500\n")
        f.write("ENERGY_MIN: 150\n")
        f.write("ENERGY_RATE: 0.1\n")
        f.write("CHARGE_RATE: 50\n\n")

        # Nodes
        f.write("# Nodes\n")
        f.write("DEPOT: 0 29.0 29.0\n")
        f.write("CUSTOMERS: 100\n")
        for i in range(1, 101):
            x = random.uniform(0, 50)
            y = random.uniform(0, 50)
            f.write(f"{i} {x:.2f} {y:.2f}\n")
        f.write("CHARGING_STATIONS: 4\n")
        f.write("201 12.5 12.5\n")
        f.write("202 12.5 37.5\n")
        f.write("203 37.5 12.5\n")
        f.write("204 37.5 37.5\n\n")

        # Demands
        f.write("# Demands\n")
        f.write("PICKUP:\n")
        f.write("0 0\n")
        for i in range(1, 101):
            pickup = random.randint(5, 30)
            f.write(f"{i} {pickup}\n")
        for i in range(201, 205):
            f.write(f"{i} 0\n")
        f.write("DELIVERY:\n")
        f.write("0 0\n")
        for i in range(1, 101):
            delivery = random.randint(5, 30)
            f.write(f"{i} {delivery}\n")
        for i in range(201, 205):
            f.write(f"{i} 0\n")

if __name__ == "__main__":
    generate_evrp_file()