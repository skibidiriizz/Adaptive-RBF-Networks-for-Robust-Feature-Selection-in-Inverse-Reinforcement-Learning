import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from stable_baselines3 import PPO
import gymnasium as gym
from collections import defaultdict
import math


class BanditKSelector:
    # multi-armed bandit for selecting optimal K for RBF centers
    #   uses Upper Confidence Bound (UCB) algorithm  
    def __init__(self, k_candidates, exploration_weight=2.0):
        self.k_candidates = k_candidates
        self.exploration_weight = exploration_weight
        self.counts = defaultdict(int)    # number of times each K was selected  
        self.rewards = defaultdict(float) # total reward for each K  
        self.values = defaultdict(float)  # average reward for each K  
        
    def select_k(self):
        # select next K to try using UCB algorithm  
        # if any K has not been tried, select it
        for k in self.k_candidates:
            if self.counts[k] == 0:
                return k
                
        # otherwise, use UCB formula to select K
        total_counts = sum(self.counts.values())
        ucb_values = {}
        
        for k in self.k_candidates:
            exploitation = self.values[k]
            exploration = self.exploration_weight * math.sqrt(math.log(total_counts) / self.counts[k])
            ucb_values[k] = exploitation + exploration
            
        return max(ucb_values, key=ucb_values.get)
        
    def update(self, k, reward):
        # update statistics for the selected K with the observed reward  
        self.counts[k] += 1
        n = self.counts[k]
        
        # incremental average update
        value = self.values[k]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.rewards[k] += reward
        self.values[k] = new_value
        
    def get_best_k(self):
        # return K with highest average reward  
        if not self.values:
            return self.k_candidates[0]  # default if no K has been tried
            
        return max(self.values, key=self.values.get)
        
    def get_stats(self):
        # return statistics for all K values  
        stats = {}
        for k in self.k_candidates:
            if self.counts[k] > 0:
                stats[k] = {
                    'count': self.counts[k],
                    'average_reward': self.values[k],
                    'total_reward': self.rewards[k]
                }
            else:
                stats[k] = {'count': 0, 'average_reward': 0, 'total_reward': 0}
        return stats

class RBF_IRL:
    def __init__(self, 
                 expert_demos, 
                 state_space_dim, 
                 n_rbf_centers=None,  # can be None when using k_selection
                 kernel_width_method="adaptive",  # "fixed", "adaptive", "learned", or "per_cluster"
                 base_kernel_width=0.5,  # starting width for fixed or initialization for others
                 width_scale_factor=1.0,  # used for adaptive scaling
                 learning_rate=0.01, 
                 epochs=100,
                 l2_regularization_lambda=0.0,
                 env_name="Pendulum-v1",
                 n_rollout_episodes=10,
                 k_candidates=None,  # list of K values to try
                 k_selection_trials=10):  # number of bandit trials for K selection
        self.expert_demos = expert_demos
        self.state_space_dim = state_space_dim
        self.n_rbf_centers = n_rbf_centers
        self.kernel_width_method = kernel_width_method
        self.base_kernel_width = base_kernel_width
        self.width_scale_factor = width_scale_factor
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_regularization_lambda = l2_regularization_lambda
        self.rbf_centers = None
        self.kernel_widths = None  # will store individual widths for each center
        self.weights = None  # reward weights
        self.env_name = env_name
        self.n_rollout_episodes = n_rollout_episodes
        
        # k selection parameters
        self.k_candidates = k_candidates if k_candidates is not None else [5, 10, 15, 20, 25, 30]
        self.k_selection_trials = k_selection_trials
        
        # initialize bandit if we are doing K selection
        self.using_k_selection = n_rbf_centers is None and k_candidates is not None
        if self.using_k_selection:
            self.bandit = BanditKSelector(self.k_candidates)
        
    def gaussian_kernel(self, state, center, width):
        # gaussian kernel with specified width  
        return np.exp(-np.linalg.norm(state - center)**2 / (2 * width**2))
        
    def generate_rbf_features(self, states):
        # generate RBF features using current centers and widths  
        if self.rbf_centers is None:
            raise ValueError("RBF centers must be initialized. Call compute_rbf_centers first.")
        
        features = np.zeros((len(states), len(self.rbf_centers)))
        for i, state in enumerate(states):
            for j, (center, width) in enumerate(zip(self.rbf_centers, self.kernel_widths)):
                features[i, j] = self.gaussian_kernel(state, center, width)
        return features
        
    def compute_rbf_centers(self, k=None, max_iter=300, n_init=4):
        """ compute RBF centers using K-means clustering and determine kernel widths
         *
         * args:
         *   k: number of clusters to use (overrides self.n_rbf_centers if provided)
         *   max_iter: maximum number of K-means iterations
         *   n_init: number of K-means initializations to try
         *
         * returns:
         *   inertia: the K-means inertia (sum of squared distances to closest centroid)
         """
        states = np.concatenate(self.expert_demos)
        
        # use provided k or instance k
        k_to_use = k if k is not None else self.n_rbf_centers
        
        # run K-means with k-means++ initialization
        kmeans = KMeans(
            n_clusters=k_to_use, 
            random_state=0,
            init='k-means++',
            n_init=n_init, 
            max_iter=max_iter
        ).fit(states)
        
        self.rbf_centers = kmeans.cluster_centers_
        
        # determine kernel widths based on the chosen method
        if self.kernel_width_method == "fixed":
            # use same width for all centers
            self.kernel_widths = np.full(k_to_use, self.base_kernel_width)
        
        elif self.kernel_width_method == "adaptive":
            # IMPROVED VERSION: more variation in widths
            
            # get cluster assignments and calculate density information
            labels = kmeans.labels_
            cluster_sizes = np.bincount(labels, minlength=k_to_use)
            
            # initialize widths array
            self.kernel_widths = np.zeros(k_to_use)
            
            for i in range(k_to_use):
                # method 1: based on nearest center distance (but weighted by cluster density)
                other_centers = np.delete(self.rbf_centers, i, axis=0)
                if len(other_centers) > 0:
                    # find distance to nearest center
                    min_dist = np.min(np.linalg.norm(self.rbf_centers[i] - other_centers, axis=1))
                    # adjust based on relative cluster size (density)
                    relative_size = cluster_sizes[i] / np.mean(cluster_sizes)
                    # dense clusters get smaller widths, sparse clusters get larger widths
                    # this factor provides more variability
                    density_factor = 1.0 / (relative_size ** 0.5)  # square root dampens extreme values
                    self.kernel_widths[i] = min_dist * self.width_scale_factor * density_factor
                else:
                    self.kernel_widths[i] = self.base_kernel_width
                
                # method 2: add cluster dispersion information
                cluster_points = states[labels == i]
                if len(cluster_points) >= 5:  # need enough points to estimate dispersion
                    # calculate average distance from points to center
                    dists_to_center = np.linalg.norm(cluster_points - self.rbf_centers[i], axis=1)
                    dispersion = np.mean(dists_to_center)
                    # blend dispersion with nearest-center distance
                    # higher dispersion → wider kernel
                    self.kernel_widths[i] = 0.5 * self.kernel_widths[i] + 0.5 * dispersion * self.width_scale_factor
        
        elif self.kernel_width_method == "adaptive_improved":
            # NEW METHOD: multivariate-aware adaptive widths
            # for high-dimensional state spaces, widths can vary per dimension
            
            # initialize with zeros
            self.kernel_widths = np.zeros(k_to_use)
            
            # get cluster assignments
            labels = kmeans.labels_
            
            for i in range(k_to_use):
                # get points in this cluster
                cluster_points = states[labels == i]
                
                if len(cluster_points) >= 5:
                    # compute covariance matrix for this cluster
                    cov_matrix = np.cov(cluster_points, rowvar=False)
                    
                    # use average variance across dimensions as width
                    avg_variance = np.mean(np.diag(cov_matrix))
                    self.kernel_widths[i] = np.sqrt(avg_variance) * self.width_scale_factor
                else:
                    # default for small clusters
                    self.kernel_widths[i] = self.base_kernel_width
                    
                # ensure width is not too small or too large
                self.kernel_widths[i] = max(0.1, min(self.kernel_widths[i], 2.0))
        
        elif self.kernel_width_method == "per_cluster":
            # compute the average distance from each center to its assigned data points
            self.kernel_widths = np.zeros(k_to_use)
            labels = kmeans.labels_
            
            for i in range(k_to_use):
                # get points assigned to this cluster
                cluster_points = states[labels == i]
                if len(cluster_points) > 0:
                    # compute average distance from center to points
                    distances = np.linalg.norm(cluster_points - self.rbf_centers[i], axis=1)
                    # use median instead of mean for robustness
                    self.kernel_widths[i] = np.median(distances) * self.width_scale_factor
                else:
                    # if no points are assigned to this cluster, use the base width
                    self.kernel_widths[i] = self.base_kernel_width
        
        elif self.kernel_width_method == "learned":
            # initialize with adaptive widths but we willl learn them during training
            # use per_cluster initialization for more variance
            self.kernel_widths = np.zeros(k_to_use)
            labels = kmeans.labels_
            
            for i in range(k_to_use):
                cluster_points = states[labels == i]
                if len(cluster_points) > 0:
                    distances = np.linalg.norm(cluster_points - self.rbf_centers[i], axis=1)
                    self.kernel_widths[i] = np.median(distances) * self.width_scale_factor
                else:
                    self.kernel_widths[i] = self.base_kernel_width
                    
                # add some random noise to create initial variance (helps optimization)
                self.kernel_widths[i] *= np.random.uniform(0.8, 1.2)
        
        else:
            raise ValueError(f"Unknown kernel width method: {self.kernel_width_method}")
        
        # apply width constraints (avoid extreme values)
        min_allowed = 0.05 * self.base_kernel_width
        max_allowed = 10.0 * self.base_kernel_width
        self.kernel_widths = np.clip(self.kernel_widths, min_allowed, max_allowed)
        
        print(f"Computed {k_to_use} RBF centers with {self.kernel_width_method} kernel widths")
        print(f"Kernel widths range: {np.min(self.kernel_widths):.4f} to {np.max(self.kernel_widths):.4f}")
        print(f"Mean width: {np.mean(self.kernel_widths):.4f}, Std dev: {np.std(self.kernel_widths):.4f}")
        
        return kmeans.inertia_
        
    def compute_feature_expectations(self, trajectories):
        all_states = np.concatenate(trajectories)
        feature_matrix = self.generate_rbf_features(all_states)
        return np.mean(feature_matrix, axis=0)
    
    def get_reward(self, state):
        phi = self.generate_rbf_features([state]).flatten()
        return np.dot(self.weights, phi)
    
    def create_custom_env(self):
        # create a gym environment with our custom reward function  
        env = gym.make(self.env_name)
        
        # use a closure to maintain reference to self
        def custom_reward_wrapper(env):
            class CustomRewardEnv(gym.Wrapper):
                def __init__(self, env, reward_func):
                    super().__init__(env)
                    self.reward_func = reward_func
                
                def step(self, action):
                    obs, _, terminated, truncated, info = self.env.step(action)
                    reward = self.reward_func(obs)
                    return obs, reward, terminated, truncated, info
            
            return CustomRewardEnv(env, self.get_reward)
        
        return custom_reward_wrapper(env)
    
    def collect_rollouts(self):
        # collect rollouts using the current reward function  
        # create environment with current reward function
        env = self.create_custom_env()
        
        # train a policy using the current reward function
        # use shorter training for efficiency during IRL iterations
        model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003, n_steps=1024)
        model.learn(total_timesteps=500)
        
        # collect trajectories using the trained policy
        rollouts = []
        for _ in range(self.n_rollout_episodes):
            states = []
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                states.append(obs)
                obs = next_obs
                done = terminated or truncated
            
            rollouts.append(np.array(states))
            
        env.close()
        return rollouts
    
    def select_optimal_k(self):
        # selects optimal K using multi-armed bandit and cluster quality metrics
        # uses approximate silhouette score for efficiency with larger datasets
          
        print("Selecting K using multi-armed bandit and expert data clustering quality...")
        best_k = None
        best_score = -np.inf  # higher is better for silhouette
        
        # get states from expert demonstrations
        states = np.concatenate(self.expert_demos)
        
        for trial in range(self.k_selection_trials):
            # use the bandit to select the next K to evaluate
            k = self.bandit.select_k()
            
            # fit K-means with k-means++ initialization
            kmeans = KMeans(n_clusters=k, n_init = 3, init='k-means++', max_iter=100, random_state=trial).fit(states)
            
            # use approximate silhouette score for efficiency with larger datasets
            if len(states) > 10000:
                # sample a subset of data for approximate silhouette calculation
                sample_indices = np.random.choice(len(states), min(5000, len(states)), replace=False)
                sample_states = states[sample_indices]
                sample_labels = kmeans.labels_[sample_indices]
                score = silhouette_score(sample_states, sample_labels)
            else:
                # use full dataset for smaller datasets
                score = silhouette_score(states, kmeans.labels_)
                
            print(f"Trial {trial+1}/{self.k_selection_trials} for K={k}, Silhouette Score: {score:.4f}")
            
            # update bandit with the observed reward (silhouette score)
            self.bandit.update(k, score)
            
            # track best K
            if score > best_score:
                best_score = score
                best_k = k
        
        print(f"\nSelected K={best_k} (best silhouette score: {best_score:.4f})")
        self.n_rbf_centers = best_k
        self.compute_rbf_centers()  # final fit with the selected K
        return best_k
        
    def train(self):
        # train the IRL model to find the reward weights  
        # if using K selection, run bandit algorithm to select K
        if self.using_k_selection:
            self.select_optimal_k()
        else:
            self.compute_rbf_centers()
        
        # initialize weights randomly
        self.weights = np.random.uniform(low=-1, high=1, size=self.n_rbf_centers)
        
        # if we are using learned kernel widths, we will need to optimize them too
        optimizing_widths = self.kernel_width_method == "learned"
        if optimizing_widths:
            # combine weights and widths for optimization
            params = np.concatenate([self.weights, self.kernel_widths])
        
        # compute expert feature expectations
        expert_feature_expectations = self.compute_feature_expectations(self.expert_demos)
        
        losses = []
        print("\nStarting IRL training...")
        for epoch in range(self.epochs):
            # collect rollouts using current reward function
            rollouts = self.collect_rollouts()
            
            # compute feature expectations for the rollouts
            rollout_feature_expectations = self.compute_feature_expectations(rollouts)
            
            # compute gradient for weights
            weight_gradient = expert_feature_expectations - rollout_feature_expectations
            
            # add L2 regularization if specified
            if self.l2_regularization_lambda > 0:
                weight_gradient -= 2 * self.l2_regularization_lambda * self.weights
            
            # update weights
            self.weights += self.learning_rate * weight_gradient
            
            # if we are optimizing kernel widths too, update them
            if optimizing_widths:
                # a simple approach: adjust widths to increase reward difference between 
                # expert and non-expert states
                
                # sample states from expert demos
                expert_states = []
                for demo in self.expert_demos:
                    indices = np.random.choice(len(demo), min(5, len(demo)))
                    expert_states.extend(demo[indices])
                expert_states = np.array(expert_states)
                
                # sample states from rollouts
                rollout_states = []
                for rollout in rollouts:
                    indices = np.random.choice(len(rollout), min(5, len(rollout)))
                    rollout_states.extend(rollout[indices])
                rollout_states = np.array(rollout_states)
                
                # for each RBF center, adjust width to maximize difference
                for i, (center, width) in enumerate(zip(self.rbf_centers, self.kernel_widths)):
                    # try slightly different widths
                    test_widths = [width * 0.9, width, width * 1.1]
                    best_diff = -np.inf
                    best_width = width
                    
                    for test_width in test_widths:
                        # compute features with this width for expert states
                        expert_features = np.array([self.gaussian_kernel(s, center, test_width) for s in expert_states])
                        expert_reward = np.mean(expert_features * self.weights[i])
                        
                        # compute features with this width for rollout states
                        rollout_features = np.array([self.gaussian_kernel(s, center, test_width) for s in rollout_states])
                        rollout_reward = np.mean(rollout_features * self.weights[i])
                        
                        # we want to maximize expert reward and minimize rollout reward
                        diff = expert_reward - rollout_reward
                        
                        if diff > best_diff:
                            best_diff = diff
                            best_width = test_width
                    
                    # update width for this center
                    self.kernel_widths[i] = best_width
            
            # compute and store loss
            loss = np.linalg.norm(expert_feature_expectations - rollout_feature_expectations)
            losses.append(loss)
            
            # print progress
            print(f"Epoch {epoch + 1}/{self.epochs}, Feature Expectation Loss: {loss:.4f}")
            if optimizing_widths and (epoch + 1) % 5 == 0:
                print(f"  Kernel width range: {np.min(self.kernel_widths):.4f} to {np.max(self.kernel_widths):.4f}")
        
        return losses
    
    def visualize_reward(self, grid_resolution=20):
        # visualize the learned reward function (for 2D state spaces)  
        if self.state_space_dim != 2:
            print("Reward visualization only supports 2D state spaces")
            return
            
        # create a grid of states
        x = np.linspace(-1, 1, grid_resolution)
        y = np.linspace(-1, 1, grid_resolution)
        X, Y = np.meshgrid(x, y)
        
        # compute reward for each state
        rewards = np.zeros((grid_resolution, grid_resolution))
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                state = np.array([X[i, j], Y[i, j]])
                rewards[i, j] = self.get_reward(state)
        
        # plot the reward landscape
        plt.figure(figsize=(10, 8))
        plt.contourf(X, Y, rewards, cmap='viridis')
        plt.colorbar(label='Reward')
        plt.title(f'Learned Reward Function (K={self.n_rbf_centers})')
        plt.xlabel('State Dimension 1')
        plt.ylabel('State Dimension 2')
        
        # plot the RBF centers
        if self.state_space_dim == 2:
            # create a figure with varying circle sizes based on kernel widths
            normalized_widths = self.kernel_widths / np.max(self.kernel_widths) * 200
            plt.scatter(
                self.rbf_centers[:, 0],
                self.rbf_centers[:, 1],
                c='red',
                marker='o',
                s=normalized_widths,
                alpha=0.5,
                edgecolors='black',
                label='RBF Centers (size = relative width)'
            )
            plt.legend()
            
        plt.show()
    
    def plot_training_curve(self, losses):
        # plot the training loss curve  
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(losses)+1), losses, marker='o')
        plt.title(f'IRL Training Curve (K={self.n_rbf_centers})')
        plt.xlabel('Epoch')
        plt.ylabel('Feature Expectation Loss')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def visualize_kernel_widths(self):
        # visualize the distribution of kernel widths  
        plt.figure(figsize=(10, 6))
        plt.hist(self.kernel_widths, bins=15)
        plt.title(f'Kernel Width Distribution ({self.kernel_width_method} method)')
        plt.xlabel('Kernel Width')
        plt.ylabel('Frequency')
        plt.axvline(np.mean(self.kernel_widths), color='r', linestyle='dashed', 
                   label=f'Mean: {np.mean(self.kernel_widths):.4f}')
        plt.axvline(np.median(self.kernel_widths), color='g', linestyle='dashed', 
                   label=f'Median: {np.median(self.kernel_widths):.4f}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def visualize_bandit_results(self):
        # visualize the results of the K selection process  
        if not self.using_k_selection:
            print("K selection was not used")
            return
            
        stats = self.bandit.get_stats()
        ks = sorted(stats.keys())
        counts = [stats[k]['count'] for k in ks]
        rewards = [stats[k]['average_reward'] for k in ks]
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # plot selection counts
        color = 'tab:blue'
        ax1.set_xlabel('Number of RBF Centers (K)')
        ax1.set_ylabel('Selection Count', color=color)
        ax1.bar(ks, counts, alpha=0.7, color=color, label='Selection Count')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # create second y-axis for average rewards
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Average Reward', color=color)
        ax2.plot(ks, rewards, 'o-', color=color, linewidth=2, label='Average Reward')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # highlight the best K
        best_k = self.bandit.get_best_k()
        best_k_idx = ks.index(best_k)
        ax2.plot(best_k, rewards[best_k_idx], 'o', color='green', markersize=12, 
                 label=f'Best K={best_k}')
        
        # add a legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
        
        plt.title('Multi-Armed Bandit K Selection Results')
        plt.tight_layout()
        plt.show()


"""
# get expert demos preloaded
import pickle
with open("pendulum_trajectories.pkl", "rb") as f:
    expert_demos = pickle.load(f)
print(len(expert_demos))

# or from pretrained model
if __name__ == '__main__':
    env = gym.make("Pendulum-v1")
    model = PPO.load("ppo_pendulum")
    expert_demos = []
    
    for _ in range(200):
        obs, _ = env.reset()
        trajectory = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            trajectory.append(obs.flatten())
            obs = next_obs
        expert_demos.append(np.array(trajectory))

    with open("pendulum_trajectories.pkl", "wb") as f:
        pickle.dump(expert_demos, f)


# Using adaptive kernel widths
irl_adaptive = RBF_IRL(
    expert_demos=expert_demos,
    state_space_dim=3,
    n_rbf_centers=20,
    kernel_width_method="adaptive",  # automatically adapts to the data distribution
    width_scale_factor=0.5,  # controls how wide the kernels are relative to data spacing
    learning_rate=0.01,
    epochs=10
)
losses = irl_adaptive.train()
irl_adaptive.plot_training_curve(losses)
irl_adaptive.visualize_kernel_widths()
"""
 