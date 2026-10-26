"""JDRL-style PPO baseline. Routing currently reuses the shared attention route model instead of a separate POMO checkpoint."""

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from pid_saca.routing import attention_model as am
from pid_saca.routing import gener_data as gd
from pid_saca.paths import REPO_ROOT, ROUTING_MODEL_DIR
import datetime
import csv

BASEFILE = REPO_ROOT


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config:
    def __init__(self):
        # Scenario parameters.
        self.departure_station = np.array([104.06, 30.67])
        self.num_destinations = 30
        self.bus_capacity = 80
        self.bus_speed = 30
        self.beta_d = 0.15
        self.price_range = [0.0, 1.0]
        self.time_slot_duration = 1.0

        # Simulated demand parameters.
        self.demand_fluctuation = 0.0

        self.demand_frequency = 1.0

        self.demand_amplitude = 12


        self.jdrl_lambda = 0


        # Learning hyperparameters.
        self.gamma = 0.99
        self.lr = 3e-4
        self.hidden_dim = 256
        self.clip_ratio = 0.2        # PPO Clip
        self.ppo_epochs = 10
        self.entropy_coef = 0.01
        self.gae_lambda = 0.95


        # Attention model dimensions.
        self.attention_heads = 4
        self.attention_layers = 2
        self.embedding_dim = 128


        # Reward shaping parameters.
        self.lambda_or = 4.0


        # Training schedule.
        self.episodes = 100
        self.time_slots_per_episode = 24
        self.max_episodes = 1000
        self.eval_interval = 10


        # Convergence and logging controls.
        self.min_episodes = 180
        self.max_episodes = 180
        self.conv_window = 2000
        self.conv_delta_ratio = 0.01
        self.conv_std_ratio = 0.05

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "Embed dim must be divisible by num heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.out_proj(attn_output)
        return self.norm(attn_output)

class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ff_linear1 = nn.Linear(embed_dim, hidden_dim)
        self.ff_linear2 = nn.Linear(hidden_dim, embed_dim)
        self.ff_norm = nn.LayerNorm(embed_dim)
        self.ff_activation = nn.ReLU()

    def forward(self, x):
        x = x + self.attn(x)
        residual = x
        ff_output = self.ff_linear2(self.ff_activation(self.ff_linear1(x)))
        ff_output = self.ff_norm(ff_output)
        return residual + ff_output

class AttentionDispatcherRouter(nn.Module):
    def __init__(self, config, dest_coords, dist_k):
        super().__init__()
        self.config = config
        self.dest_coords = dest_coords
        self.dist_k = dist_k
        self.device = device

        EMBEDDING_DIM = 64
        N_HEADS = 8
        N_LAYERS = 3
        self.route_model = am.AttentionRouteModel(EMBEDDING_DIM, N_HEADS, N_LAYERS).to(self.device)
        model_path = ROUTING_MODEL_DIR / "best_model.pth"
        if model_path.exists():
            self.route_model.load_state_dict(torch.load(str(model_path), map_location=self.device))
        self.route_model.eval()

        stats_path = ROUTING_MODEL_DIR / "normalization_stats.pt"
        if stats_path.exists():
            self.stats = torch.load(str(stats_path), map_location=self.device)
        else:
            self.stats = {'mean': torch.zeros(3), 'std': torch.ones(3)}

    def dispatch(self, orders, buses):
        if len(orders) == 0 or len(buses) == 0:
            return {}, {}

        bus_routes = {}
        bus_orders = {}
        remaining_orders = orders.copy()
        available_buses = [bid for bid in buses if buses[bid][0] == 0.0]

        mean = self.stats['mean'].to(self.device)
        std = self.stats['std'].to(self.device)
        start_node = torch.tensor(self.config.departure_station, device=self.device, dtype=torch.float32).unsqueeze(0)
        start_norm = (start_node - mean[None, :2]) / std[None, :2]

        for bus_id in available_buses:
            if len(remaining_orders) == 0:
                break
            bus_capacity = int(buses[bus_id][1])

            order_dists = [self.dist_k[d] for d in remaining_orders]
            sorted_indices = np.argsort(order_dists)[:bus_capacity]
            assigned_orders = [remaining_orders[i] for i in sorted_indices]

            keep_mask = np.ones(len(remaining_orders), dtype=bool)
            keep_mask[sorted_indices] = False
            remaining_orders = [remaining_orders[i] for i in range(len(remaining_orders)) if keep_mask[i]]

            if len(assigned_orders) == 0:
                continue

            unique_dests = list(set(assigned_orders))
            loc_np = self.dest_coords[unique_dests]
            loc_tensor = torch.tensor(loc_np, device=self.device, dtype=torch.float32).unsqueeze(0)
            weight_list = [assigned_orders.count(d) for d in unique_dests]
            weight_tensor = torch.tensor(weight_list, device=self.device, dtype=torch.float32).unsqueeze(0)

            loc_norm = (loc_tensor - mean[None, None, :2]) / std[None, None, :2]
            weight_norm = (weight_tensor - mean[2]) / std[2]

            with torch.no_grad():
                route_indices, _ = self.route_model(loc_norm, start_norm, weight_norm)

            pred_indices = (route_indices.squeeze(0).cpu().numpy() - 1).tolist()
            route = [unique_dests[idx] for idx in pred_indices if 0 <= idx < len(unique_dests)]

            bus_routes[bus_id] = route
            bus_orders[bus_id] = assigned_orders

        return bus_routes, bus_orders

class BusBookingEnv:
    def __init__(self, config):
        self.config = config
        self.init_destinations()
        self.dispatcher = AttentionDispatcherRouter(
            config, self.dest_coords, self.dist_k
        ).to(device)
        self.reset()

    def init_destinations(self):
        self.dest_coords = np.array(gd.readbusstations(gd.config.FILE_PATH))
        dep_lon = torch.full((self.config.num_destinations,), float(self.config.departure_station[0]), device=device)
        dep_lat = torch.full((self.config.num_destinations,), float(self.config.departure_station[1]), device=device)
        dest_lons = torch.tensor(self.dest_coords[:, 0], device=device, dtype=torch.float32)
        dest_lats = torch.tensor(self.dest_coords[:, 1], device=device, dtype=torch.float32)
        self.dist_k = am.haversine_torch(dep_lon, dep_lat, dest_lons, dest_lats, device).detach().cpu().numpy()
        self.dist_max = np.max(self.dist_k)
        self.dist_min = np.min(self.dist_k)

    def reset(self):
        self.time_slots = []
        self.current_p = np.zeros(self.config.num_destinations)
        self.num_buses = 10
        self.buses = {i: [0.0, self.config.bus_capacity] for i in range(self.num_buses)}
        self.N_p = np.random.poisson(self.config.demand_amplitude, self.config.num_destinations)
        return self.get_state()

    def get_state(self):
        bus_remaining_time = [bus[0] for bus in self.buses.values()]
        state = np.concatenate([self.N_p, bus_remaining_time])
        return torch.FloatTensor(state)

    def demand_function(self, p_k):
        multiplier = (self.dist_max + self.dist_min - self.dist_k) / self.dist_max
        F_pk = 1.0 - multiplier * ((np.exp(p_k) - 1.0) / (math.e - 1.0))
        F_pk = np.clip(F_pk, 0.0, 1.0)
        return self.N_p * F_pk

    def supply_function(self, p_k, a_k):
        self.N_s = sum([bus[1] for bus in self.buses.values() if bus[0] == 0.0])
        if self.N_s == 0:
            return np.zeros_like(p_k)
        return self.N_s * a_k

    def calculate_route_distance(self, route):
        if len(route) == 0: return 0.0
        path_coords = np.array(
            [self.config.departure_station] +
            [self.dest_coords[k] for k in route] +
            [self.config.departure_station]
        )
        lon1, lat1, lon2, lat2 = map(np.radians, [path_coords[:-1, 0], path_coords[:-1, 1], path_coords[1:, 0], path_coords[1:, 1]])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return np.sum(6371.0 * c)

    def update_bus_state(self, bus_routes):
        for bus_id in self.buses:
            self.buses[bus_id][0] = max(0.0, self.buses[bus_id][0] - self.config.time_slot_duration)
        available_buses = [bid for bid in self.buses if self.buses[bid][0] == 0.0]
        for i, (bus_id, route) in enumerate(bus_routes.items()):
            if i >= len(available_buses): break
            bid = available_buses[i]
            dis = self.calculate_route_distance(route)
            self.buses[bid][0] = dis / self.config.bus_speed
            self.buses[bid][1] = self.config.bus_capacity

    def step(self, action):
        self.current_p, a_k = action
        a_k = np.clip(a_k, 0, 1)
        a_k = a_k / (np.sum(a_k) + 1e-6)

        current_N_p = self.N_p.copy()
        D_k = self.demand_function(self.current_p)
        S_k = self.supply_function(self.current_p, a_k)
        O_k = np.minimum(D_k, S_k).astype(int)


        if np.sum(O_k) == 0:
            self.update_bus_state({})
            self.N_p = np.random.poisson(self.config.demand_amplitude + self.config.demand_fluctuation * np.sin(len(self.time_slots) * self.config.demand_frequency), self.config.num_destinations)
            self.time_slots.append(len(self.time_slots) + 1)
            done = len(self.time_slots) >= self.config.time_slots_per_episode
            next_state = self.get_state()
            bus_rewards = np.zeros(self.num_buses)
            return next_state, 0.0, bus_rewards, done, {"revenue": 0.0, "orr": 0.0, "orr_effective": 0.0, "cost": 0.0, "total_distance": 0.0, "orders_accepted": 0}


        orders = []
        for k in range(self.config.num_destinations):
            orders.extend([k] * O_k[k])

        bus_routes, bus_orders = self.dispatcher.dispatch(orders, self.buses)

        fulfilled_O_k = np.zeros(self.config.num_destinations, dtype=int)
        for assigned_list in bus_orders.values():
            for dest_id in assigned_list:
                fulfilled_O_k[dest_id] += 1


        cost_total = 0.0
        bus_rewards = np.zeros(self.num_buses)
        total_orders_accepted = np.sum(fulfilled_O_k)
        total_potential_demand = np.sum(current_N_p)

        for bus_id, route in bus_routes.items():
            carried_orders = bus_orders.get(bus_id, [])
            revenue_bus = sum([self.current_p[dest_id] for dest_id in carried_orders])

            dis_i = self.calculate_route_distance(route)
            cost_bus = self.config.beta_d * dis_i
            cost_total += cost_bus


            num_orders_this_bus = len(carried_orders)
            if total_orders_accepted > 0:
                orr_share = num_orders_this_bus / total_orders_accepted
            else:
                orr_share = 0.0
            global_orr = total_orders_accepted / (total_potential_demand + 1e-6)
            orr_bonus = self.config.lambda_or * global_orr * orr_share

            bus_rewards[bus_id] = revenue_bus - cost_bus + orr_bonus

        self.update_bus_state(bus_routes)

        revenue = np.sum(self.current_p * fulfilled_O_k)
        R_t = revenue - cost_total
        ORR_t = np.sum(fulfilled_O_k) / (np.sum(current_N_p) + 1e-6)
        orr_effective = np.sum(fulfilled_O_k) / (np.sum(D_k) + 1e-6)

        global_reward = R_t + self.config.lambda_or * ORR_t

        self.N_p = np.random.poisson(self.config.demand_amplitude + self.config.demand_fluctuation * np.sin(len(self.time_slots) * self.config.demand_frequency), self.config.num_destinations)
        self.time_slots.append(len(self.time_slots) + 1)
        done = len(self.time_slots) >= self.config.time_slots_per_episode


        next_state = self.get_state()

        info = {
            "revenue": revenue,
            "orr": ORR_t,
            "orr_effective": orr_effective,
            "cost": cost_total,
            "total_distance": cost_total / self.config.beta_d,
            "orders_accepted": int(np.sum(fulfilled_O_k)),
            "orders_nominal": int(np.sum(O_k))
        }
        return next_state, global_reward, bus_rewards, done, info

class JDRLNetwork(nn.Module):
    def __init__(self, config, state_dim, action_dim):
        super(JDRLNetwork, self).__init__()
        self.config = config


        self.spatial_dim = config.num_destinations
        self.attr_dim = state_dim - self.spatial_dim


        self.spatial_net = nn.Sequential(
            nn.Linear(self.spatial_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )


        self.attr_net = nn.Sequential(
            nn.Linear(self.attr_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )


        self.fusion_net = nn.Sequential(
            nn.Linear(128 + 64, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU()
        )


        self.fc_p_mean = nn.Linear(config.hidden_dim, config.num_destinations)
        self.fc_a_mean = nn.Linear(config.hidden_dim, config.num_destinations)

        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

        # Critic Head (Value Function)
        self.value_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, state):
        spatial_input = state[:, :self.spatial_dim]
        attr_input = state[:, self.spatial_dim:]

        g_t = self.spatial_net(spatial_input)
        z_t = self.attr_net(attr_input)

        h_t = torch.cat([g_t, z_t], dim=-1)
        features = self.fusion_net(h_t)
        return features

    def get_action_distribution(self, state):
        features = self.forward(state)
        p_mean = torch.sigmoid(self.fc_p_mean(features))
        a_mean = torch.sigmoid(self.fc_a_mean(features))
        action_mean = torch.cat([p_mean, a_mean], dim=-1)

        log_std_clamped = torch.clamp(self.log_std, min=-3.0, max=0.0)
        action_std = torch.exp(log_std_clamped).expand_as(action_mean)
        return Normal(action_mean, action_std)

    def get_value(self, state):
        features = self.forward(state)
        return self.value_head(features)

class JDRLAgent:
    def __init__(self, config, state_dim, action_dim):
        self.config = config
        self.network = JDRLNetwork(config, state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.lr)

        # On-Policy Buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.bus_rewards_buffer = []
        self.dones = []

    def select_action(self, state, deterministic=False):
        state = state.unsqueeze(0).to(device)
        with torch.no_grad():
            dist = self.network.get_action_distribution(state)
            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            value = self.network.get_value(state)

        action_np = action.cpu().numpy()[0]
        p = np.clip(action_np[:self.config.num_destinations], 0, 1)
        a = np.clip(action_np[self.config.num_destinations:], 0, 1)
        return p, a, action_np, log_prob.item(), value.item()

    def store(self, state, action, log_prob, value, bus_rewards, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.bus_rewards_buffer.append(bus_rewards)
        self.dones.append(done)

    def calculate_advantages(self):
        total_rewards_per_bus = np.sum(self.bus_rewards_buffer, axis=0)
        worst_bus_idx = np.argmin(total_rewards_per_bus)


        rewards_matrix = np.array(self.bus_rewards_buffer)


        sum_rewards = np.sum(rewards_matrix, axis=1)
        worst_bus_rewards = rewards_matrix[:, worst_bus_idx]

        jdrl_rewards = self.config.jdrl_lambda * sum_rewards + \
                       (1 - self.config.jdrl_lambda) * worst_bus_rewards
        jdrl_rewards = torch.tensor(jdrl_rewards, dtype=torch.float32, device=device)

        values = torch.tensor(self.values + [0], dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)


        returns = []
        gae = 0
        for t in reversed(range(len(self.bus_rewards_buffer))):
            delta = jdrl_rewards[t] + self.config.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            returns.insert(0, gae + values[t])

        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages = returns - values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def update(self):
        if len(self.states) == 0: return 0, 0

        states = torch.stack(self.states).to(device)
        actions = torch.tensor(np.array(self.actions), dtype=torch.float32, device=device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=device)
        returns, advantages = self.calculate_advantages()

        actor_losses, critic_losses = [], []

        for _ in range(self.config.ppo_epochs):
            dist = self.network.get_action_distribution(states)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
            new_values = self.network.get_value(states).squeeze(-1)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * advantages

            actor_loss = -torch.min(surr1, surr2).mean() - self.config.entropy_coef * entropy
            critic_loss = F.mse_loss(new_values, returns)

            self.optimizer.zero_grad()
            (actor_loss + 0.5 * critic_loss).backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())


        self.states, self.actions, self.log_probs, self.values = [], [], [], []
        self.bus_rewards_buffer, self.dones = [], []

        return np.mean(critic_losses), np.mean(actor_losses)

def evaluate_policy(agent, config, episodes=5):
    eval_env = BusBookingEnv(config)
    eval_env.dispatcher.load_state_dict(agent.network.state_dict(), strict=False)

    metrics = {"avg_reward": [], "revenue": [], "orr": [], "total_distance": [], "cost": []}

    for _ in range(episodes):
        state = eval_env.reset()
        totals = {"reward_sum": 0.0, "revenue": [], "orr": [], "total_distance": [], "cost": []}
        step_count = 0
        for _ in range(config.time_slots_per_episode):
            p, a, _, _, _ = agent.select_action(state, deterministic=True)
            next_state, r_global, _, done, info = eval_env.step((p, a))
            totals["reward_sum"] += r_global
            step_count += 1
            totals["revenue"].append(info["revenue"])
            totals["orr"].append(info["orr"])
            totals["total_distance"].append(info["total_distance"])
            totals["cost"].append(info["cost"])
            state = next_state
            if done: break

        metrics["avg_reward"].append(float(totals["reward_sum"] / max(1, step_count)))
        metrics["revenue"].append(float(np.mean(totals["revenue"])) if totals["revenue"] else 0.0)
        metrics["orr"].append(float(np.mean(totals["orr"])) if totals["orr"] else 0.0)
        metrics["total_distance"].append(float(np.mean(totals["total_distance"])) if totals["total_distance"] else 0.0)
        metrics["cost"].append(float(np.mean(totals["cost"])) if totals["cost"] else 0.0)

    summary = {k: float(np.mean(v)) for k, v in metrics.items()}
    print(f"Evaluation summary over {episodes} episodes: {summary}")
    return summary

def train_jdrl(config, run_name=None):
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    print(f"Training JDRL on device: {device}")

    env = BusBookingEnv(config)
    state_dim = config.num_destinations + env.num_buses
    action_dim = 2 * config.num_destinations

    agent = JDRLAgent(config, state_dim, action_dim)


    log = {
        "total_reward": [], "revenue": [], "orr": [],
        "total_distance": [], "cost": []
    }

    for episode in range(config.max_episodes):
        state = env.reset()
        episode_log = {k: [] for k in ["revenue", "orr", "total_distance", "cost"]}
        total_global_reward = 0
        step_count = 0

        for t in range(config.time_slots_per_episode):
            p, a, action_full, log_prob, value = agent.select_action(state)

            next_state, reward, bus_rewards, done, info = env.step((p, a))

            agent.store(state, action_full, log_prob, value, bus_rewards, done)

            total_global_reward += reward
            step_count += 1
            for k in episode_log: episode_log[k].append(info[k])

            state = next_state
            if done: break

        cl, al = agent.update()


        avg_reward = total_global_reward / max(1, step_count)
        log["total_reward"].append(avg_reward)
        for k in episode_log: log[k].append(np.mean(episode_log[k]))

        if (episode + 1) % config.eval_interval == 0:
            print(f"Episode {episode+1}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Revenue: {np.mean(episode_log['revenue']):.2f}")
            print(f"  Avg ORR: {np.mean(episode_log['orr']):.3f}")
            print(f"  Avg Distance: {np.mean(episode_log['total_distance']):.2f}km")
            print(f"  Avg Cost: {np.mean(episode_log['cost']):.2f}")
            print(f"  JDRL Loss (Actor/Critic): {al:.4f} / {cl:.4f}\n")


        K = config.conv_window
        if len(log["total_reward"]) >= max(config.min_episodes, 2 * K):
            last_K = np.array(log["total_reward"][-K:])
            prev_K = np.array(log["total_reward"][-2*K:-K])
            ma_last = last_K.mean()
            ma_prev = prev_K.mean()
            delta = abs(ma_last - ma_prev)
            delta_thr = config.conv_delta_ratio * (abs(ma_last) + 1e-8)
            std_thr = 0.1 * (abs(ma_last) + 1e-8)

            if (delta <= delta_thr) and (last_K.std() <= std_thr) and (episode + 1 >= config.max_episodes):
                print(f"[Early Stopping] Converged at episode {episode+1}")
                break


    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name: timestamp = f"{run_name}_{timestamp}"
    log_dir = BASEFILE / "log"
    log_dir.mkdir(exist_ok=True)

    csv_file = log_dir / f'jdrl_training_log_{timestamp}.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        keys = list(log.keys())
        writer.writerow(['Episode'] + keys)
        for i in range(len(log[keys[0]])):
            writer.writerow([i+1] + [f"{log[k][i]:.4f}" for k in keys])
    print(f"Training log saved to '{csv_file}'")


    plt.figure(figsize=(18, 12))
    plt.subplot(2, 3, 1)
    plt.plot(log["total_reward"], label='Reward', color='blue')
    plt.title("Average Reward per Episode")
    plt.xlabel("Episode"); plt.ylabel("Reward"); plt.grid(True, alpha=0.3); plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(log["revenue"], label='Revenue', color='green')
    plt.title("Average Revenue per Time Slot")
    plt.xlabel("Episode"); plt.ylabel("Revenue"); plt.grid(True, alpha=0.3); plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(log["cost"], label='Cost', color='red')
    plt.title("Average Cost per Time Slot")
    plt.xlabel("Episode"); plt.ylabel("Cost"); plt.grid(True, alpha=0.3); plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(log["orr"], label='ORR', color='orange')
    plt.title("Average ORR per Time Slot")
    plt.xlabel("Episode"); plt.ylabel("ORR"); plt.grid(True, alpha=0.3); plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(log["total_distance"], label='Distance', color='purple')
    plt.title("Average Distance per Time Slot")
    plt.xlabel("Episode"); plt.ylabel("Distance (km)"); plt.grid(True, alpha=0.3); plt.legend()

    plt.subplot(2, 3, 6)
    plt.plot(log["revenue"], label='Revenue', color='green', alpha=0.7)
    plt.plot(log["cost"], label='Cost', color='red', alpha=0.7)
    plt.title("Revenue vs Cost Comparison")
    plt.xlabel("Episode"); plt.ylabel("Value"); plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()

    return agent, env

if __name__ == "__main__":
    config = Config()
    train_jdrl(config)
