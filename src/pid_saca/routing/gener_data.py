"""Synthetic supervised routing-data generator based on the selected station coordinates."""

import random
import math
import os, pathlib
from pid_saca.paths import REPO_ROOT, STATION_FILE
BASE = pathlib.Path(__file__).resolve().parent

class config:
    BUS_SIZE = 50
    FILE_PATH = STATION_FILE
    STATIONNUM = [16, 17]
    TIANFUPOS = [104.06, 30.67]
    DATANUM = 1024

def readbusstations(file_path):
    bus_stations = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(',')

            if line.startswith("Station_ID,Longitude,Latitude"):
                continue
            if len(parts) >= 3:
                longitude = float(parts[1])
                latitude = float(parts[2])
                bus_stations.append([longitude, latitude])
    return bus_stations

def generate_passengers_distributed(num_stations, total_passengers):


    if num_stations <= 0:
        return []
    if total_passengers == 0:
        return [0] * num_stations
    if num_stations == 1:
        return [total_passengers]


    weights = [random.random() for _ in range(num_stations)]
    total_weight = sum(weights)


    if total_weight == 0:
        result = [0] * num_stations
        result[0] = total_passengers
        return result


    ideal_shares = [(w / total_weight) * total_passengers for w in weights]


    stations = [int(share) for share in ideal_shares]


    remainder = total_passengers - sum(stations)


    fractional_parts = [(ideal_shares[i] - stations[i]) for i in range(num_stations)]


    indices_sorted_by_fraction = sorted(
        range(num_stations),
        key=lambda k: fractional_parts[k],
        reverse=True
    )


    for i in range(remainder):
        stations[indices_sorted_by_fraction[i]] += 1

    return stations

class TspPassengerTimeSolver:
    def __init__(self, stations, passengers, start_pos):

        self.start_pos = start_pos
        self.stations = stations


        self.all_points = [self.start_pos] + self.stations
        self.num_points = len(self.all_points)


        self.passengers_at_station = [0] + passengers
        self.total_passengers = sum(passengers) # BUS_SIZE


        self.dist_matrix = self._compute_dist_matrix()


        self.bestcost = float('inf')
        self.bestroute = []

    def _distance(self, p1, p2):

        R = 6371.0
        lat1 = math.radians(p1[1])
        lon1 = math.radians(p1[0])
        lat2 = math.radians(p2[1])
        lon2 = math.radians(p2[0])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def _compute_dist_matrix(self):

        matrix = [[0.0] * self.num_points for _ in range(self.num_points)]
        for i in range(self.num_points):
            for j in range(i + 1, self.num_points):
                dist = self._distance(self.all_points[i], self.all_points[j])
                matrix[i][j] = dist
                matrix[j][i] = dist
        return matrix

    def solve(self):


        visited = [False] * self.num_points

        visited[0] = True


        self.dfs(
            current_node=0,
            current_path=[0],
            current_cost=0,
            remaining_passengers=self.total_passengers,
            visited=visited
        )


        return self.bestroute, self.bestcost


    def dfs(self, current_node, current_path, current_cost, remaining_passengers, visited):


        if current_cost >= self.bestcost:
            return


        if len(current_path) == self.num_points:

            if current_cost < self.bestcost:
                self.bestcost = current_cost

                self.bestroute = [idx - 1 for idx in current_path[1:]]
            return


        unvisited_stations = []
        for i in range(1, self.num_points):
            if not visited[i]:
                unvisited_stations.append(i)

        if unvisited_stations:


            min_dist_to_next = min(self.dist_matrix[current_node][u] for u in unvisited_stations)


            lower_bound = current_cost + (min_dist_to_next * remaining_passengers)

            if lower_bound >= self.bestcost:
                return


        neighbors = sorted(unvisited_stations, key=lambda u: self.dist_matrix[current_node][u])

        for next_node in neighbors:


            segment_distance = self.dist_matrix[current_node][next_node]


            segment_cost = segment_distance * remaining_passengers


            new_cost = current_cost + segment_cost
            new_remaining_passengers = remaining_passengers - self.passengers_at_station[next_node]


            visited[next_node] = True
            self.dfs(
                next_node,
                current_path + [next_node],
                new_cost,
                new_remaining_passengers,
                visited
            )


            visited[next_node] = False

def generate_routelist(bus_stations, n_dest=None):


    data = []

    print(f"Generating {config.DATANUM} supervised routing samples...")
    print("=" * 60)

    for i in range(config.DATANUM):

        if n_dest is not None:
            num_stations = n_dest
        else:
            num_stations = random.randint(config.STATIONNUM[0], config.STATIONNUM[1])


        selected_station_indices = random.sample(range(len(bus_stations)), num_stations)


        selected_station_coords = [bus_stations[idx] for idx in selected_station_indices]


        people_counts = generate_passengers_distributed(num_stations, config.BUS_SIZE)


        solver = TspPassengerTimeSolver(
            stations=selected_station_coords,
            passengers=people_counts,
            start_pos=config.TIANFUPOS
        )


        local_optimal_order, optimal_cost = solver.solve()


        global_optimal_order = [selected_station_indices[local_idx] for local_idx in local_optimal_order]


        data_item = {
            'station_ids': selected_station_indices,
            'station_coords': selected_station_coords,
            'passengers': people_counts,
            'global_optimal_order': global_optimal_order,
            'optimal_cost': optimal_cost,
            'local_optimal_order': local_optimal_order,
            'avg_cost_per_passenger': optimal_cost / config.BUS_SIZE
        }

        data.append(data_item)


        if (i + 1) % max(1, config.DATANUM // 10) == 0:
            print(f"Progress: {i+1}/{config.DATANUM} | "
                  f"stations: {num_stations} | "
                  f"cost: {optimal_cost:.2f} | "
                  f"avg: {optimal_cost/config.BUS_SIZE:.2f}")

    print("=" * 60)
    print(f"OK: Dataset generation completed with {len(data)} samples")

    return data

def save():
    for i in range(config.STATIONNUM[0], config.STATIONNUM[1] + 1):

        N_dest = i
        datanum = 2 ** 10
        config.DATANUM = datanum
        bus_stations = readbusstations(config.FILE_PATH)
        dataset = generate_routelist(bus_stations, n_dest=N_dest)
        output_dir = REPO_ROOT / "data" / "routing_supervised"
        output_dir.mkdir(parents=True, exist_ok=True)
        file_name = output_dir / f"supervised_dataset_{N_dest}_stations.json"
        import json
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

def generate_single_data(numstations):
    bus_stations = readbusstations(config.FILE_PATH)

    TIANFUPOS = config.TIANFUPOS

    stations = random.sample(bus_stations, numstations)
    passengers = generate_passengers_distributed(len(stations), config.BUS_SIZE)
    BUS_SIZE = sum(passengers)


    solver = TspPassengerTimeSolver(stations, passengers, TIANFUPOS)


    bestroute, bestcost = solver.solve()


    print(f"Station coordinates: {stations}")
    print(f"Passenger counts: {passengers}")
    print(f"Start position: {TIANFUPOS}")
    print("---")
    print(f"Best route (station indices): {bestroute}")
    print(f"Minimum total passenger-time cost: {bestcost:.4f}")
    print(f"Minimum average passenger-time cost: {(bestcost / BUS_SIZE):.4f}")
    return stations, passengers, TIANFUPOS, bestroute, bestcost

if __name__ == "__main__":
    numstations = 18
    generate_single_data(numstations)
