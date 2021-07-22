import sys
import numpy as np
from gym import spaces
import gym


def PAdam(A, b, x_start, length, bound=10, step_size=1.0):
    obj_vals = []

    def Obj_func(x):
        return np.dot(np.dot(A, x), x) + np.dot(x, b)

    def Grad_eval(x):
        return np.dot(A + A.transpose(), x).flatten() + b

    x = x_start
    time_since_grad_reset = 0
    beta_1 = 0.9
    beta_2 = 0.999
    grad_mean = 0
    grad_var = 0
    for i in range(length):
        obj_vals.append(Obj_func(x))
        epsilon = 1e-8
        current_grad = Grad_eval(x)
        grad_mean = beta_1 * grad_mean + (1.0 - beta_1) * current_grad
        grad_var = beta_2 * grad_var + (1.0 - beta_2) * np.square(current_grad)
        time_since_grad_reset += 1
        t = time_since_grad_reset  # t is really t+1 here
        mean_hat = grad_mean / (1 - beta_1 ** t)
        var_hat = grad_var / (1 - beta_2 ** t)
        step_size = 1.0
        x_action_delta = step_size * np.divide(mean_hat, np.sqrt(var_hat) + epsilon)
        # The clipping operation could cause issues with adam from a motivational stand point
        x = np.clip(x - x_action_delta, -bound, bound)

    return obj_vals


class SwitchingquadraticEnv(gym.Env):
    """
    This is enviroment is for a random quadratic problem not necessarly convex
    with a switching based optimizer. Our agent choses between a random new point, SGD, or adam.
    """

    def Obj_func(self, x):
        A = self.A
        b = self.b

        return np.dot(np.dot(A, x), x) + np.dot(x, b)

    def Grad_eval(self, x):
        A = self.A
        b = self.b

        return np.dot(A + A.transpose(), x).flatten() + b

    def __init__(self, config):
        super(SwitchingquadraticEnv, self).__init__()
        self.dim_list = config["dim"]
        self.dim = np.random.choice(self.dim_list)  # size of matrix dim* dim
        # size of our search cube (-bound, bound)^dim
        self.bound = config["bound"]
        self.num_steps = config["num_steps"]
        self.switching_style = config["switching_style"]

        """
        Current state is  (-bound, bound)^dim
        objective value deltas are (-inf, inf)^(h_len,1)
        """
        bound = self.bound
        dim = self.dim
        # More complicated matrix generation

        random_m = np.random.uniform(-1, 1, size=(self.dim, self.dim))
        # More complicated matrix generation
        # old code
        if config["old_A_style"]:
            self.A = 0.5 * (random_m + np.transpose(random_m))
        else:
            self.A = np.tril(random_m) + np.tril(random_m, k=-1).T

        self.b = np.random.uniform(-1, 1, size=self.dim)
        self.x = np.random.uniform(low=-bound, high=bound, size=dim)

        """
        Adding memory depth
        """
        self.memory_depth = config["Memory_depth"]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9 + self.memory_depth,), dtype=np.float32
        )

        """
        Action space should be trinary choice

        """
        if self.switching_style == "All":
            self.action_space = spaces.Discrete(3)
        elif self.switching_style == "RandAdam":
            self.action_space = spaces.Discrete(2)
        elif self.switching_style == "RandGD":
            self.action_space = spaces.Discrete(2)
        elif self.switching_style == "AdamGD":
            self.action_space = spaces.Discrete(2)
        else:
            raise ValueError(f"'{self.switching_style}' is not a valid swithing style")

        # All of our mutable state
        self.x = np.random.uniform(low=-bound, high=bound, size=dim)
        self.best_val = self.Obj_func(self.x)
        self.time_since_grad_reset = 0
        self.step_count = 0
        self.grad_mean = self.Grad_eval(self.x)
        self.total_grad_norm = np.linalg.norm(self.Grad_eval(self.x))
        self.grad_var = np.zeros(shape=(dim,), dtype=np.float32)
        temp_padam_val = min(
            PAdam(
                self.A, self.b, self.x, self.num_steps, bound=self.bound, step_size=1.0,
            )
        )
        if (
            temp_padam_val < self.Obj_func(self.x)
            and abs(temp_padam_val - self.Obj_func(self.x)) > 1.0
        ):
            self.padam_reward_normalize = abs(temp_padam_val - self.Obj_func(self.x))
        else:
            self.padam_reward_normalize = 1.0

    def reset(self):
        self.dim = np.random.choice(self.dim_list)
        random_m = np.random.uniform(-1, 1, size=(self.dim, self.dim))
        self.A = 0.5 * (random_m + np.transpose(random_m))
        self.b = np.random.uniform(-1, 1, size=self.dim)
        self.x = np.random.uniform(low=-self.bound, high=self.bound, size=self.dim)
        self.best_val = self.Obj_func(self.x)
        self.total_grad_norm = np.linalg.norm(self.Grad_eval(self.x))
        self.grad_mean = self.Grad_eval(self.x)
        self.grad_var = np.zeros(shape=(self.dim,), dtype=np.float32)
        self.time_since_grad_reset = 0
        self.step_count = 0
        temp_padam_val = min(
            PAdam(
                self.A, self.b, self.x, self.num_steps, bound=self.bound, step_size=1.0,
            )
        )
        if (
            temp_padam_val < self.Obj_func(self.x)
            and abs(temp_padam_val - self.Obj_func(self.x)) > 1.0
        ):
            self.padam_reward_normalize = abs(temp_padam_val - self.Obj_func(self.x))
        else:
            self.padam_reward_normalize = 1.0

        return_list = []

        return_list.append(np.linalg.norm(self.Grad_eval(self.x)))  # Grad norm

        return_list.append(self.Obj_func(self.x))  # current value
        return_list.append(self.best_val)  # best value
        return_list.append(self.step_count)
        return_list.append(self.dim)
        # new observation information
        return_list.append(np.linalg.norm(self.A))
        return_list.append(np.linalg.norm(self.b))
        return_list.append(np.linalg.norm(self.x))
        return_list.append(self.total_grad_norm)

        self.memory = []
        for x in range(self.memory_depth):
            self.memory.append(-1.0)
            return_list.append(-1.0)

        return_array = np.asarray(return_list, dtype=np.float32)

        # return our observation
        return return_array

    def step(self, action):
        beta_1 = 0.9
        beta_2 = 0.999

        bound = self.bound
        dim = self.dim
        if self.switching_style == "AdamGD":
            action += 1
        if self.switching_style == "RandAdam" and action == 1:
            action = 2

        self.memory = self.memory[1:]
        self.memory.append(action)

        if action == 0:  # Random search
            self.x = np.random.uniform(low=-bound, high=bound, size=dim)
            self.grad_mean = self.Grad_eval(self.x)
            self.grad_var = np.zeros(shape=(dim,), dtype=np.float32)
            self.time_since_grad_reset = 0

        elif action == 1:  # PGD

            step_size = 1.0  # Make this dynamic later
            current_grad = self.Grad_eval(self.x)
            x_action_delta = step_size * current_grad

            self.grad_mean = beta_1 * self.grad_mean + (1.0 - beta_1) * current_grad
            self.grad_var = beta_2 * self.grad_var + (1.0 - beta_2) * np.square(
                current_grad
            )

            self.x = np.clip(self.x - x_action_delta, -bound, bound)
            self.time_since_grad_reset += 1

        elif action == 2:  # PAdam
            """
            Adam uses running averages of gradients and second moments. Thus we should
            reset these each time we do a random step, and we should compute them for each
            PGD step as well

            """
            epsilon = 1e-8

            current_grad = self.Grad_eval(self.x)

            self.grad_mean = beta_1 * self.grad_mean + (1.0 - beta_1) * current_grad
            self.grad_var = beta_2 * self.grad_var + (1.0 - beta_2) * np.square(
                current_grad
            )

            self.time_since_grad_reset += 1
            t = self.time_since_grad_reset  # t is really t+1 here

            mean_hat = self.grad_mean / (1 - beta_1 ** t)

            var_hat = self.grad_var / (1 - beta_2 ** t)

            step_size = 1.0
            x_action_delta = step_size * np.divide(mean_hat, np.sqrt(var_hat) + epsilon)

            # The clipping operation could cause issues with adam from a motivational stand point
            self.x = np.clip(self.x - x_action_delta, -bound, bound)

        else:
            sys.exit(f"Action outside of bounds with value: {action}")

        current_obj = self.Obj_func(self.x)

        if current_obj < self.best_val:
            # normalize reward
            if self.padam_reward_normalize < 1.0:
                reward = abs(self.best_val - current_obj) / (
                    self.num_steps * self.dim * self.bound
                )
            else:
                reward = abs(self.best_val - current_obj) / (
                    self.padam_reward_normalize
                )
            self.best_val = current_obj
        else:
            reward = 0.0

        """
        Here update the observation space. 
        """

        self.step_count += 1

        return_list = []

        return_list.append(np.linalg.norm(self.Grad_eval(self.x)))  # Grad norm

        return_list.append(self.Obj_func(self.x))  # current value
        return_list.append(self.best_val)  # best value
        return_list.append(self.step_count)
        return_list.append(self.dim)
        # new observation information
        return_list.append(np.linalg.norm(self.A))
        return_list.append(np.linalg.norm(self.b))
        return_list.append(np.linalg.norm(self.x))
        self.total_grad_norm += np.linalg.norm(self.Grad_eval(self.x))
        return_list.append(self.total_grad_norm / (self.step_count + 1.0))
        for x in self.memory:
            return_list.append(x)

        return_array = np.asarray(return_list, dtype=np.float32)

        return (
            return_array,
            reward,
            self.step_count == self.num_steps,
            {"Obj_val": self.Obj_func(self.x), "best_val": self.best_val},
        )

    def render(self, mode="human", close=False):
        pass

    def close(self):
        pass
