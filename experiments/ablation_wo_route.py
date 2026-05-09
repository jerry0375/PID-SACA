"""Routing ablation that replaces the learned attention router with a greedy nearest-neighbor dispatcher."""

import torch
import numpy as np
import _bootstrap  # noqa: F401
from pid_saca.algorithms import saca as SACA
import pathlib
import csv
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GreedyDispatcher:


    def __init__(self, dest_coords, departure_station, dist_k):

        self.dest_coords = dest_coords
        self.departure_station = departure_station
        self.dist_k = dist_k

    def _haversine_distance(self, loc1, locs2):

        R = 6371.0

        if loc1.ndim == 1:
            loc1 = loc1.reshape(1, 2)

        lon1, lat1 = loc1[:, 0], loc1[:, 1]
        lon2, lat2 = locs2[:, 0], locs2[:, 1]

        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    def dispatch(self, orders, buses):
        if not orders or not buses:
            return {}, {}

        bus_routes = {}
        bus_orders = {}
        remaining_orders = orders.copy()
        available_buses = [bid for bid in buses if buses[bid][0] == 0.0]

        for bus_id in available_buses:
            if not remaining_orders:
                break

            capacity = int(buses[bus_id][1])




            current_order_dists = [self.dist_k[d] for d in remaining_orders]


            sorted_indices = np.argsort(current_order_dists)[:capacity]


            assigned_orders = [remaining_orders[i] for i in sorted_indices]


            keep_mask = np.ones(len(remaining_orders), dtype=bool)
            keep_mask[sorted_indices] = False
            remaining_orders = [remaining_orders[i] for i in range(len(remaining_orders)) if keep_mask[i]]

            if not assigned_orders:
                continue




            route_sequence = []


            to_visit = assigned_orders.copy()
            current_loc = self.departure_station

            while to_visit:


                target_coords = self.dest_coords[to_visit] # (M, 2)
                dists = self._haversine_distance(current_loc, target_coords)


                nearest_idx = np.argmin(dists)
                next_dest_id = to_visit[nearest_idx]


                route_sequence.append(next_dest_id)


                current_loc = self.dest_coords[next_dest_id]


                to_visit.pop(nearest_idx)

            bus_routes[bus_id] = route_sequence
            bus_orders[bus_id] = assigned_orders

        return bus_routes, bus_orders

class AblationEnv(SACA.BusBookingEnv):


    def __init__(self, config):

        super().__init__(config)


        print(">>> [Ablation Info] Loading GreedyDispatcher (Haversine) instead of AttentionRouter...")
        self.dispatcher = GreedyDispatcher(self.dest_coords, self.config.departure_station, self.dist_k)


        self.reset()

def train_ablation_wo_route():

    config = SACA.Config()
    config.lambda_or = 0
    run_name = "wo_AttnRoute"


    env = AblationEnv(config)


    state_dim = config.num_destinations + len(env.buses)
    action_dim = 2 * config.num_destinations

    sac_agent = SACA.SAC(config, state_dim, action_dim, env.dispatcher, env.dest_coords)

    print(f"=== Starting Ablation Training: {run_name} (Spherical Distance) ===")


    log = {
        "total_reward": [],
        "revenue": [],
        "orr": [],
        "total_distance": [],
        "cost": []
    }

    for episode in range(config.max_episodes):
        state = env.reset()
        total_reward = 0.0
        episode_log = {k: [] for k in ["revenue", "orr", "total_distance", "cost"]}
        step_count = 0

        for t in range(config.time_slots_per_episode):
            p, a = sac_agent.select_action(state, deterministic=False)
            action = np.concatenate([p, a])


            next_state, reward, done, info = env.step((p, a))
            total_reward += reward
            step_count += 1

            sac_agent.store_transition(state, action, reward, next_state, done)

            if t % 5 == 0:
                sac_agent.update()

            for k in episode_log.keys():
                episode_log[k].append(info[k])

            state = next_state
            if done:
                break


        avg_reward_ep = total_reward / max(1, step_count)
        log["total_reward"].append(avg_reward_ep)
        for k in episode_log.keys():
            log[k].append(np.mean(episode_log[k]))


        if (episode + 1) % config.eval_interval == 0:
            print(f"Episode {episode+1}")
            print(f"  Avg Reward: {avg_reward_ep:.2f}")
            print(f"  Avg Revenue: {np.mean(episode_log['revenue']):.2f}")
            print(f"  Avg ORR: {np.mean(episode_log['orr']):.3f}")
            print(f"  Avg Distance: {np.mean(episode_log['total_distance']):.2f}km")
            print(f"  Avg Cost: {np.mean(episode_log['cost']):.2f}")
            print(f"  (Mode: {run_name})\n")


        K = config.conv_window
        n_ep = len(log["total_reward"])
        if n_ep >= max(config.min_episodes, 2 * K):
            last_K = np.array(log["total_reward"][-K:])
            prev_K = np.array(log["total_reward"][-2*K:-K])
            ma_last = last_K.mean()
            ma_prev = prev_K.mean()
            delta = abs(ma_last - ma_prev)

            delta_thr = config.conv_delta_ratio * (abs(ma_last) + 1e-8)
            std_thr = 0.1 * (abs(ma_last) + 1e-8)
            std_last = last_K.std()

            if (delta <= delta_thr) and (std_last <= std_thr):
                print(f"[Early Stopping] Converged at episode {episode+1}")
                break


    save_log(log, run_name)
    print("Training Finished.")

def save_log(log, run_name):
    path = pathlib.Path(__file__).resolve().parents[1] / "log"
    path.mkdir(exist_ok=True)
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = path / f"ablation_log_{run_name}_{timestamp}.csv"

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        keys = list(log.keys())
        writer.writerow(['Episode'] + keys)
        num_records = len(log[keys[0]])
        for i in range(num_records):
            row = [i+1] + [f"{log[k][i]:.4f}" for k in keys]
            writer.writerow(row)
    print(f"Log saved to {filename}")

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    train_ablation_wo_route()
