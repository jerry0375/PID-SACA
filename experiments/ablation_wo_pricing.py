"""Pricing ablation with fixed prices; the policy only learns the allocation portion of the action."""


import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from scipy.spatial import distance_matrix
import random
from collections import deque
import matplotlib.pyplot as plt
import _bootstrap  # noqa: F401
from pid_saca.routing import attention_model as am
from pid_saca.routing import gener_data as gd
from pid_saca.paths import REPO_ROOT, ROUTING_MODEL_DIR
BASEFILE = REPO_ROOT


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


FIXED_PRICE = 0.5

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
        self.demand_fluctuation = 10.0

        self.demand_frequency = 1.0

        self.demand_amplitude = 12


        # PID/Lagrangian constraint parameters.
        self.target_orr = 0.5
        self.pid_kp = 1
        self.pid_ki = 1
        self.pid_kd = 0.5

        self.lambda_init = 4.0
        self.lambda_min = 0.0
        self.lambda_max = 200000.0
        self.pid_update_freq = 1


        # Learning hyperparameters.
        self.gamma = 0.99
        self.tau = 0.005
        self.lr = 3e-4
        self.alpha = 0.2
        self.hidden_dim = 256
        self.batch_size = 256
        self.memory_size = 100000


        # Attention model dimensions.
        self.attention_heads = 4
        self.attention_layers = 2
        self.embedding_dim = 128


        # Reward shaping parameters.
        self.lambda_or = 4.0


        # Training schedule.
        self.episodes = 100
        self.time_slots_per_episode = 24
        self.target_entropy = -2 * self.num_destinations

        # Convergence and logging controls.
        self.min_episodes = 180
        self.max_episodes = 200
        self.conv_window = 20
        self.conv_delta_ratio = 0.01
        self.conv_std_ratio = 0.05
        self.eval_interval = 10


class PIDLagrangianController:
    def __init__(self, config):
        self.kp = config.pid_kp
        self.ki = config.pid_ki
        self.kd = config.pid_kd
        self.lr = getattr(config, "pid_lr", 0.05)
        self.target = config.target_orr
        self.min_val = config.lambda_min
        self.max_val = config.lambda_max

        init_val = max(float(config.lambda_init), 1e-6)
        self.log_val = math.log(init_val)
        self.current_val = float(np.clip(math.exp(self.log_val), self.min_val, self.max_val))
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, current_orr):
        error = self.target - current_orr

        self.integral += error
        self.integral = float(np.clip(self.integral, -50.0, 50.0))

        derivative = error - self.prev_error

        delta_x = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        step_x = float(np.clip(self.lr * delta_x, -1.0, 1.0))
        max_log = math.log(max(self.max_val, 1.0))
        min_log = math.log(max(self.min_val, 1e-6))
        self.log_val = float(np.clip(self.log_val + step_x, min_log, max_log))
        self.current_val = float(np.clip(math.exp(self.log_val), self.min_val, self.max_val))

        self.prev_error = error

        return self.current_val, error


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
        self.buses = {i: [0.0, self.config.bus_capacity] for i in range(10)}
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
            return np.zeros(self.config.num_destinations)

        return self.N_s * a_k

    def calculate_reward(self, O_k, N_p, cost_total, dynamic_lambda=None):
        lam = dynamic_lambda if dynamic_lambda is not None else self.config.lambda_or

        p_k = self.current_p
        revenue = np.sum(p_k * O_k)
        R_t = revenue - cost_total

        ORR_t = np.sum(O_k) / (np.sum(N_p) + 1e-6)

        total_reward = R_t + lam * ORR_t

        return total_reward, R_t, ORR_t

    def step(self, action, dynamic_lambda=None):
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
            return self.get_state(), 0.0, done, {"revenue": 0.0, "orr": 0.0, "orr_effective": 0.0, "cost": 0.0, "total_distance": 0.0, "orders_accepted": 0, "lambda": dynamic_lambda if dynamic_lambda is not None else self.config.lambda_or}

        orders = []
        for k in range(self.config.num_destinations):
            orders.extend([k] * O_k[k])

        bus_routes, bus_orders = self.dispatcher.dispatch(orders, self.buses)

        fulfilled_O_k = np.zeros(self.config.num_destinations, dtype=int)
        for assigned_list in bus_orders.values():
            for dest_id in assigned_list:
                fulfilled_O_k[dest_id] += 1

        cost_total = 0.0
        for route in bus_routes.values():
            cost_total += self.config.beta_d * self.calculate_route_distance(route)

        self.update_bus_state(bus_routes)

        reward, revenue, orr = self.calculate_reward(fulfilled_O_k, current_N_p, cost_total, dynamic_lambda)
        orr_effective = np.sum(fulfilled_O_k) / (np.sum(D_k) + 1e-6)

        self.N_p = np.random.poisson(self.config.demand_amplitude + self.config.demand_fluctuation * np.sin(len(self.time_slots) * self.config.demand_frequency), self.config.num_destinations)
        self.time_slots.append(len(self.time_slots) + 1)

        next_state = self.get_state()
        done = len(self.time_slots) >= self.config.time_slots_per_episode

        info = {
            "revenue": revenue,
            "orr": orr,
            "orr_effective": orr_effective,
            "cost": cost_total,
            "total_distance": cost_total / self.config.beta_d,
            "orders_accepted": int(np.sum(fulfilled_O_k)),
            "orders_nominal": int(np.sum(O_k)),
            "lambda": dynamic_lambda if dynamic_lambda is not None else self.config.lambda_or
        }
        return next_state, reward, done, info

    def calculate_route_distance(self, route):
        if len(route) == 0:
            return 0.0

        path_coords = np.array(
            [self.config.departure_station] +
            [self.dest_coords[k] for k in route] +
            [self.config.departure_station]
        )

        lon1, lat1 = path_coords[:-1, 0], path_coords[:-1, 1]
        lon2, lat2 = path_coords[1:, 0], path_coords[1:, 1]

        R = 6371.0

        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        distances = R * c

        return np.sum(distances)

    def update_bus_state(self, bus_routes):
        for bus_id in self.buses:
            self.buses[bus_id][0] = max(0.0, self.buses[bus_id][0] - self.config.time_slot_duration)

        available_buses = [bid for bid in self.buses if self.buses[bid][0] == 0.0]
        for i, (bus_id, route) in enumerate(bus_routes.items()):
            if i < len(available_buses):
                assigned_bus = available_buses[i]
                route_dist = self.calculate_route_distance(route)
                travel_time = route_dist / self.config.bus_speed
                self.buses[assigned_bus][0] = max(0.0, travel_time - self.config.time_slot_duration)
                self.buses[assigned_bus][1] = self.config.bus_capacity


class SpatialActor(nn.Module):
    def __init__(self, config, state_dim, action_dim, dest_coords):
        super(SpatialActor, self).__init__()
        self.config = config
        self.action_dim = action_dim

        self.register_buffer("dest_coords", torch.tensor(dest_coords, dtype=torch.float32))

        self.station_embed_dim = 64
        self.input_proj = nn.Linear(3, self.station_embed_dim)

        self.attn_layer = AttentionLayer(
            embed_dim=self.station_embed_dim,
            num_heads=4,
            hidden_dim=128
        )

        self.num_buses = state_dim - config.num_destinations
        self.fusion_dim = self.station_embed_dim + self.num_buses

        self.fc1 = nn.Linear(self.fusion_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)


        self.fc_p = nn.Linear(config.hidden_dim, config.num_destinations)
        self.fc_a = nn.Linear(config.hidden_dim, config.num_destinations)

        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state):
        batch_size = state.size(0)

        demands = state[:, :self.config.num_destinations]
        bus_states = state[:, self.config.num_destinations:]

        coords_batch = self.dest_coords.unsqueeze(0).expand(batch_size, -1, -1)
        demands_expanded = demands.unsqueeze(-1)

        station_features = torch.cat([demands_expanded, coords_batch], dim=-1)

        x_embed = self.input_proj(station_features)
        x_attn = self.attn_layer(x_embed)

        global_demand_feat = x_attn.mean(dim=1)

        fusion_input = torch.cat([global_demand_feat, bus_states], dim=-1)

        x = F.relu(self.fc1(fusion_input))
        x = F.relu(self.fc2(x))


        p_mean = torch.full((batch_size, self.config.num_destinations), FIXED_PRICE, device=state.device)
        a_mean = torch.sigmoid(self.fc_a(x))

        action_mean = torch.cat([p_mean, a_mean], dim=-1)

        log_std_scalar = torch.tanh(x[:, :1] * 0.1)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std_scalar + 1)
        std = log_std.exp().expand_as(action_mean)

        return action_mean, std

    def sample(self, state):
        mean, std = self.forward(state)


        batch_size = state.size(0)


        p_fixed = torch.full((batch_size, self.config.num_destinations), FIXED_PRICE, device=state.device)


        a_mean = mean[:, self.config.num_destinations:]
        a_std = std[:, self.config.num_destinations:]

        normal_a = Normal(a_mean, a_std)
        x_t_a = normal_a.rsample()
        a_trans = torch.sigmoid(x_t_a)

        y_t = torch.cat([p_fixed, a_trans], dim=-1)


        log_prob = normal_a.log_prob(x_t_a).sum(dim=-1)
        correction = torch.log(a_trans * (1 - a_trans) + 1e-6).sum(dim=-1)
        log_prob -= correction

        return y_t, log_prob, mean


class Critic(nn.Module):
    def __init__(self, config, state_dim, action_dim):
        super(Critic, self).__init__()
        self.config = config

        self.q1_net = self._build_q_net(state_dim, action_dim)
        self.q2_net = self._build_q_net(state_dim, action_dim)

    def _build_q_net(self, state_dim, action_dim):
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)
        return q1, q2

    def q1_forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1_net(sa)


class SAC:
    def __init__(self, config, state_dim, action_dim, dispatcher, dest_coords):
        self.config = config
        self.actor = SpatialActor(config, state_dim, action_dim, dest_coords).to(device)
        self.critic = Critic(config, state_dim, action_dim).to(device)
        self.target_critic = Critic(config, state_dim, action_dim).to(device)
        self.dispatcher = dispatcher

        self.optimizer_actor = optim.Adam(
            self.actor.parameters(),
            lr=config.lr
        )
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=config.lr)

        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(param.data)

        self.memory = deque(maxlen=config.memory_size)
        self.target_entropy = getattr(config, "target_entropy", -action_dim)
        self.log_alpha = torch.tensor(
            math.log(config.alpha),
            device=device,
            requires_grad=True,
        )
        self.optimizer_alpha = optim.Adam([self.log_alpha], lr=config.lr)

    def select_action(self, state, deterministic=False, greedy_samples=0):
        state = state.unsqueeze(0).to(device)
        with torch.no_grad():
            if greedy_samples > 0:
                best_action = None
                best_q = -float("inf")
                for _ in range(greedy_samples):
                    candidate, _, _ = self.actor.sample(state)
                    q_val = self.critic.q1_forward(state, candidate).item()
                    if q_val > best_q:
                        best_q = q_val
                        best_action = candidate
                action_tensor = best_action if best_action is not None else self.actor(state)[0]
            elif deterministic:
                action_tensor = self.actor(state)[0]
            else:
                action_tensor = self.actor.sample(state)[0]
        action = action_tensor.detach().cpu().numpy()[0]


        p = np.full(self.config.num_destinations, FIXED_PRICE)
        a = action[self.config.num_destinations:]
        return p, a

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.config.batch_size:
            return

        batch = random.sample(self.memory, self.config.batch_size)
        state = torch.cat([s.unsqueeze(0) for s, _, _, _, _ in batch], dim=0).to(device)
        action = torch.tensor(
            np.stack([a for _, a, _, _, _ in batch]),
            dtype=torch.float32,
            device=device,
        )
        reward = torch.FloatTensor([r for _, _, r, _, _ in batch]).unsqueeze(1).to(device)
        next_state = torch.cat([ns.unsqueeze(0) for _, _, _, ns, _ in batch], dim=0).to(device)
        done = torch.FloatTensor([d for _, _, _, _, d in batch]).unsqueeze(1).to(device)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            target_q1, target_q2 = self.target_critic(next_state, next_action)
            alpha = self.alpha
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob.unsqueeze(1)
            target_q = reward + (1 - done) * self.config.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        self.actor.train()
        self.critic.eval()
        action_pi, log_prob_pi, _ = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, action_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        alpha = self.alpha
        actor_loss = (alpha * log_prob_pi.unsqueeze(1) - q_pi).mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        self.critic.train()

        alpha_loss = -(self.log_alpha * (log_prob_pi + self.target_entropy).detach()).mean()
        self.optimizer_alpha.zero_grad()
        alpha_loss.backward()
        self.optimizer_alpha.step()

        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "optimizer_actor": self.optimizer_actor.state_dict(),
            "optimizer_critic": self.optimizer_critic.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "optimizer_alpha": self.optimizer_alpha.state_dict(),
        }

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.optimizer_actor.load_state_dict(state["optimizer_actor"])
        self.optimizer_critic.load_state_dict(state["optimizer_critic"])
        self.log_alpha.data.copy_(state["log_alpha"].to(device))
        self.optimizer_alpha.load_state_dict(state["optimizer_alpha"])


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

        self.embedding = nn.Linear(2, config.embedding_dim)
        self.encoder = nn.Sequential(
            *[AttentionLayer(
                config.embedding_dim,
                config.attention_heads,
                config.hidden_dim
            ) for _ in range(config.attention_layers)]
        )

        self.decoder_proj = nn.Linear(config.embedding_dim * 2, 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        EMBEDDING_DIM=64
        N_HEADS=8
        N_LAYERS=3
        self.route_model = am.AttentionRouteModel(EMBEDDING_DIM, N_HEADS, N_LAYERS).to(self.device)
        model_path = ROUTING_MODEL_DIR / "best_model.pth"
        self.route_model.load_state_dict(torch.load(str(model_path), map_location=self.device))
        self.route_model.eval()

        stats_path = ROUTING_MODEL_DIR / "normalization_stats.pt"
        self.stats = torch.load(str(stats_path), map_location=self.device)

    def embed_destinations(self, dest_ids):
        if len(dest_ids) == 0:
            return torch.zeros(1, 0, self.config.embedding_dim, device=device)
        coords = torch.tensor(self.dest_coords[dest_ids], dtype=torch.float32, device=device)
        embedded = self.embedding(coords)
        return embedded.unsqueeze(0)

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

            route = []
            for idx in pred_indices:
                if idx >= 0 and idx < len(unique_dests):
                    dest_id = unique_dests[idx]
                    route.append(dest_id)

            bus_routes[bus_id] = route
            bus_orders[bus_id] = assigned_orders

        return bus_routes, bus_orders


def evaluate_policy(agent, config, episodes=5):
    eval_env = BusBookingEnv(config)
    eval_env.dispatcher.load_state_dict(agent.dispatcher.state_dict())
    eval_env.dispatcher.to(device)
    metrics = {"avg_reward": [], "revenue": [], "orr": [], "total_distance": [], "cost": []}
    for _ in range(episodes):
        state = eval_env.reset()
        totals = {"reward_sum": 0.0, "revenue": [], "orr": [], "total_distance": [], "cost": []}
        step_count = 0
        for _ in range(config.time_slots_per_episode):
            p, a = agent.select_action(state, greedy_samples=10)
            next_state, reward, done, info = eval_env.step((p, a))
            totals["reward_sum"] += reward
            step_count += 1
            totals["revenue"].append(info["revenue"])
            totals["orr"].append(info["orr"])
            totals["total_distance"].append(info["total_distance"])
            totals["cost"].append(info["cost"])
            state = next_state
            if done:
                break
        metrics["avg_reward"].append(float(totals["reward_sum"] / max(1, step_count)))
        metrics["revenue"].append(float(np.mean(totals["revenue"])) if totals["revenue"] else 0.0)
        metrics["orr"].append(float(np.mean(totals["orr"])) if totals["orr"] else 0.0)
        metrics["total_distance"].append(float(np.mean(totals["total_distance"])) if totals["total_distance"] else 0.0)
        metrics["cost"].append(float(np.mean(totals["cost"])) if totals["cost"] else 0.0)
    summary = {k: float(np.mean(v)) for k, v in metrics.items()}
    print(f"Evaluation summary over {episodes} episodes: {summary}")
    return summary


def train_ablation_wo_pricing(config, run_name=None):
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(42)

    print(f"Training on device: {device}")
    print(f"[Ablation] Fixed Price P = {FIXED_PRICE}")

    env = BusBookingEnv(config)
    state_dim = config.num_destinations + len(env.buses)
    action_dim = 2 * config.num_destinations
    sac = SAC(config, state_dim, action_dim, env.dispatcher, env.dest_coords)

    pid_controller = PIDLagrangianController(config)
    current_lambda = config.lambda_init

    log = {
        "total_reward": [], "revenue": [], "orr": [],
        "total_distance": [], "cost": [], "lambda": []
    }

    for episode in range(config.max_episodes):
        state = env.reset()
        total_reward = 0.0
        episode_log = {k: [] for k in ["revenue", "orr", "total_distance", "cost"]}
        step_count = 0
        episode_lambda = current_lambda

        for t in range(config.time_slots_per_episode):
            p, a = sac.select_action(state, deterministic=False)
            action = np.concatenate([p, a])

            next_state, reward, done, info = env.step((p, a), dynamic_lambda=episode_lambda)
            total_reward += reward
            step_count += 1

            sac.store_transition(state, action, reward, next_state, done)

            if t % 5 == 0:
                update_result = sac.update()
                if update_result is not None:
                    critic_loss, actor_loss = update_result

            for k in episode_log.keys():
                episode_log[k].append(info[k])
            state = next_state

            if done:
                break

        avg_orr_ep = np.mean(episode_log['orr'])

        if (episode + 1) % config.pid_update_freq == 0:
            current_lambda, error = pid_controller.update(avg_orr_ep)

        log["lambda"].append(current_lambda)
        avg_reward_ep = total_reward / max(1, step_count)
        log["total_reward"].append(avg_reward_ep)
        for k in episode_log.keys():
            log[k].append(np.mean(episode_log[k]))

        if (episode + 1) % config.eval_interval == 0:
            print(f"Episode {episode+1}")
            print(f"  [Ablation] Fixed Price P = {FIXED_PRICE}")
            print(f"  Lambda (PID): {current_lambda:.4f} | Target ORR: {config.target_orr} | Curr ORR: {avg_orr_ep:.3f}")
            print(f"  Avg Reward: {avg_reward_ep:.2f}")
            print(f"  Avg Revenue: {np.mean(episode_log['revenue']):.2f}")
            print(f"  Avg ORR: {np.mean(episode_log['orr']):.3f}")
            print(f"  Avg Distance: {np.mean(episode_log['total_distance']):.2f}km")
            print(f"  Avg Cost: {np.mean(episode_log['cost']):.2f}\n")

        K = config.conv_window
        n_ep = len(log["total_reward"])
        if n_ep >= max(config.min_episodes, 2000 * K):
            last_K = np.array(log["total_reward"][-K:])
            prev_K = np.array(log["total_reward"][-2*K:-K])
            ma_last = last_K.mean()
            ma_prev = prev_K.mean()
            delta = abs(ma_last - ma_prev)

            delta_thr = config.conv_delta_ratio * (abs(ma_last) + 1e-8)
            std_thr = 0.1 * (abs(ma_last) + 1e-8)
            std_last = last_K.std()

            if (delta <= delta_thr) and (std_last <= std_thr):
                print(f"[Early Stopping] Converged at episode {episode+1}: "
                      f"MA_prev={ma_prev:.3f}, MA_last={ma_last:.3f}, "
                      f"delta={delta:.3f}<=thr={delta_thr:.3f}, "
                      f"std={std_last:.3f}<=thr={std_thr:.3f}")
                break

    eval_summary = evaluate_policy(sac, config, episodes=5)

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name:
        timestamp = f"{run_name}_{timestamp}"

    log_dir = BASEFILE / "log"
    log_dir.mkdir(exist_ok=True)
    param_dir = BASEFILE / "param"
    param_dir.mkdir(exist_ok=True)

    import csv

    csv_file = log_dir / f'ablation_wo_pricing_training_log_{timestamp}.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        keys = list(log.keys())
        writer.writerow(['Episode'] + keys)
        num_records = len(log[keys[0]])
        for i in range(num_records):
            row = [i + 1] + [f"{log[k][i]:.4f}" for k in keys]
            writer.writerow(row)
    print(f"Training log saved to '{csv_file}'")

    plt.figure(figsize=(18, 12))

    plt.subplot(2, 3, 1)
    plt.plot(log["total_reward"], label='Reward', color='blue')
    plt.title("Average Reward per Episode (w/o Pricing)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(log["revenue"], label='Revenue', color='green')
    plt.title("Average Revenue per Time Slot")
    plt.xlabel("Episode")
    plt.ylabel("Revenue")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(log["cost"], label='Cost', color='red')
    plt.title("Average Cost per Time Slot")
    plt.xlabel("Episode")
    plt.ylabel("Cost")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(log["orr"], label='ORR', color='orange')
    plt.title("Average ORR per Time Slot")
    plt.xlabel("Episode")
    plt.ylabel("ORR")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(log["total_distance"], label='Distance', color='purple')
    plt.title("Average Distance per Time Slot")
    plt.xlabel("Episode")
    plt.ylabel("Distance (km)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 3, 6)
    plt.plot(log["revenue"], label='Revenue', color='green', alpha=0.7)
    plt.plot(log["cost"], label='Cost', color='red', alpha=0.7)
    plt.title("Revenue vs Cost Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    return sac, env


if __name__ == "__main__":
    config = Config()
    sac_agent, env = train_ablation_wo_pricing(config)
