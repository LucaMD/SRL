import torch

class PrioritizedExperienceReplay:
    def __init__(self, beta_start, alpha, sample_alpha, epsilon, use_gpu):
        # PER important weights and params
        self.beta_start = beta_start
        self.alpha = alpha
        self.epsilon = epsilon
        self.gpu = use_gpu
        self.sample_alpha = sample_alpha
        self.device = torch.device('cuda' if use_gpu else 'cpu')

    def create_weights(self, rewards, start_prob):
        """Create weights for prioritized experience replay"""
        reward = torch.Tensor(rewards).to(self.device)
        self.prob = reward.abs() + start_prob
        temp = 1.0 / self.prob
        infinity = torch.Tensor([float('inf')]).to(self.device)
        temp[temp == infinity] = 1.0
        norm = 1.0 / reward.shape[0]
        self.imp_weight = (temp * norm) ** self.beta_start
        self.count = torch.zeros_like(self.prob)

    def sample(self, batch_size):
        # Prioritised exp replay based batch sampling
        if torch.rand([1]).item() < self.sample_alpha:

            # sometimes, random is just best
            batch_ids = torch.randint(high=self.prob.shape[0], size=[batch_size], device=self.device)
        else:
            # if no alpha defined, do PER
            batch_ids = torch.multinomial(self.prob, batch_size, replacement=True)

        # update the count of how often these state were sampled from the whole dataset 
        self.count[batch_ids] += 1

        return batch_ids.to(self.device)

    def update_weights(self, batch_ids, error):
        # Set the selection weight/prob to the abs prediction error and update the importance sampling weight
        new_weights = pow((error + self.epsilon), self.alpha)
        self.prob[batch_ids] = new_weights
        temp = 1.0 / new_weights
        norm = 1 / self.prob.shape[0]
        self.imp_weight[batch_ids] = (temp * norm) ** self.beta_start

    def calculate_importance_sampling_weights(self, batch_ids):
        with torch.no_grad():
            # Calculate the importance sampling weights for PER
            imp_sampling_weights = self.imp_weight[batch_ids] / self.imp_weight.max()
            imp_sampling_weights[torch.isnan(imp_sampling_weights)] = 1
            imp_sampling_weights = imp_sampling_weights.clamp(min=0.001)
        return imp_sampling_weights
