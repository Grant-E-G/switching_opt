# Filter tensorflow version warnings

import os

# https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
import warnings

# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=Warning)
import tensorflow as tf

tf.get_logger().setLevel("INFO")
tf.autograph.set_verbosity(0)
import logging

tf.get_logger().setLevel(logging.ERROR)


import pickle
import datetime
import numpy as np

from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy

import matplotlib
import matplotlib.pyplot as plt



import yaml

from envs import PAdam, SwitchingquadraticEnv

# fix saving bug
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def make_env(rank, config, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        # env = gym.make(env_id)
        env = SwitchingquadraticEnv(config)
        env.seed(seed + rank)
        return env

    set_global_seeds(seed)
    return _init


def trainit(yaml_file, retrain="None"):
    with open(yaml_file) as infile:
        yaml_config = yaml.load(infile, Loader=yaml.FullLoader)

    "Base path needs to be changed"
    basepath = yaml_config["Basepath"]

    env_config = yaml_config["env_config"]
    if "ppo2_nsteps" in yaml_config.keys():
        ppo2_nsteps = yaml_config["ppo2_nsteps"]
    else:
        ppo2_nsteps = 1000
    if "policy_kwargs" in yaml_config.keys():
        policy_kwargs = yaml_config["policy_kwargs"]
    else:
        policy_kwargs = None

    check_env(SwitchingquadraticEnv(env_config))

    num_cpu = yaml_config["num_cpus"]  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv(
        [make_env(i, env_config) for i in range(num_cpu)], start_method="spawn"
    )

    print(env)

    model = PPO2(
        MlpPolicy,
        env,
        policy_kwargs=policy_kwargs,
        n_steps=ppo2_nsteps,
        nminibatches=2,
        verbose=1,
        tensorboard_log=basepath + yaml_config["TF_logpath"],
    )
    if retrain != "None":
        model = PPO2.load(retrain)

    """
    To do move code block below
    """

    def evaluate(model, env, num_steps=1000):
        """
        Evaluate a RL agent
        :param model: (BaseRLModel object) the RL Agent
        :param num_steps: (int) number of timesteps to evaluate it
        :return: (float) Mean reward for the last 100 episodes
        
        Double check this evaluation code
        """
        episode_rewards = [0.0]
        obs = env.reset()
        print(num_steps / 100 + 1)
        for j in range(int(num_steps / 100) + 1):
            for i in range(100):
                # _states are only useful when using LSTM policies
                action, _states = model.predict(obs)

                obs, reward, done, info = env.step(action)
                # print(reward)
                # Stats
                episode_rewards[-1] += reward
                if done.all():
                    obs = env.reset()
                    # print(not(i == 99 and j == int(num_steps/100)))
                    if not (i == 99 and j == int(num_steps / 100)):
                        # print('appended')
                        episode_rewards.append(0.0)
        # Compute mean reward for the last 100 episodes
        flattened_episode_rewards = np.concatenate(episode_rewards)

        mean_ep_reward = np.mean(flattened_episode_rewards)
        std_ep_reward = np.std(flattened_episode_rewards)
        print(
            "Mean reward:",
            mean_ep_reward,
            "Num episodes:",
            len(flattened_episode_rewards),
            "Std reward:",
            std_ep_reward,
        )

        return mean_ep_reward, std_ep_reward

    """
    End of code block
    """

    # Random Agent, before training
    mean_reward_before_train, std_reward_before_train = evaluate(
        model, env, num_steps=500000
    )
    # mean_reward, std_reward = evaluate_policy(model,eval_env, n_eval_episodes=10)
    print(f"Mean reward  before training:  {mean_reward_before_train}")

    # Train the agent
    # int(4e7)
    model.learn(total_timesteps=yaml_config["total_timesteps"], log_interval=10)
    # Save the agent
    model_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    print(f"{model_time}")
    save_string = basepath + f"agents/{yaml_config['savetag']}_{model_time}"
    ensure_dir(save_string)
    model.save(save_string)

    mean_reward_after_train, std_reward_after_train = evaluate(
        model, env, num_steps=500000
    )
    print(
        f"Mean reward  before training:  {mean_reward_before_train} +- {std_reward_before_train}"
    )
    print(
        f"Mean reward  after training:  {mean_reward_after_train} +- {std_reward_after_train}"
    )
    return save_string


"""
Beginning of plotting helper functions


"""


def complex_evaluate_vec(model, env, num_steps=1000, agent_dim=40):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: Dictionary of lists
    
    env.something = side effects?
    """
    action_list = []
    current_list = []
    best_list = []
    A_b_list = []
    starting_x = []
    obs = env.reset()
    for i in range(num_steps):
        # this will be wrong
        if env.get_attr("step_count") == [0] * agent_dim:
            starting_x.append(env.get_attr("x"))
            A_b_list.append((env.get_attr("A"), env.get_attr("b")))
            best_list.append(env.get_attr("best_val"))
        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)
        action_list.append(action)

        obs, reward, done, info = env.step(action)
        # {"Obj_val": self.Obj_func(self.x), "best_val": self.best_val},

        current_list.append([x["Obj_val"] for x in info])
        best_list.append([x["best_val"] for x in info])

        # Stats
        if done.all():
            obs = env.reset()

    return_dict = {
        "action_list": action_list,
        "current_list": current_list,
        "best_list": best_list,
        "A_b_list": A_b_list,
        "starting_x": starting_x,
    }

    return return_dict


def generate_ab_pairs(complex_dictionary):
    list_of_pairs = []
    for (A, b) in complex_dictionary["A_b_list"]:
        for a_i, b_i in zip(A, b):
            list_of_pairs.append((a_i, b_i))
    return list_of_pairs


def PGD(A, b, x_start, length, bound=10, step_size=1.0):
    obj_vals = []

    def Obj_func(x):

        return np.dot(np.dot(A, x), x) + np.dot(x, b)

    def Grad_eval(x):

        return np.dot(A + A.transpose(), x).flatten() + b

    x = x_start
    for i in range(length):
        obj_vals.append(Obj_func(x))
        current_grad = Grad_eval(x)
        x_action_delta = step_size * current_grad
        x = np.clip(x - x_action_delta, -bound, bound)

    return obj_vals


def generate_PAdam_runs(complex_dictionary, bound):
    list_of_runs = []
    flat_start = [x for y in complex_dictionary["starting_x"] for x in y]
    A_bs = generate_ab_pairs(complex_dictionary)
    for (A, b), x in zip(A_bs, flat_start):
        list_of_runs.append(PAdam(A, b, x, 101, bound))
    return list_of_runs


def generate_Abx0(complex_dictionary, lastbest_list):
    list_triples = []
    flat_start = [x for y in complex_dictionary["starting_x"] for x in y]
    A_bs = generate_ab_pairs(complex_dictionary)
    for (A, b), x, fval in zip(A_bs, flat_start, lastbest_list):
        list_triples.append((A, b, x, fval))
    return list_triples


def RS(A, b, x_start, length, bound=10, step_size=1.0):  # bounds are static
    obj_vals = []

    def Obj_func(x):

        return np.dot(np.dot(A, x), x) + np.dot(x, b)

    def Grad_eval(x):

        return np.dot(A + A.transpose(), x).flatten() + b

    x = x_start
    dim = x.shape
    for i in range(length):
        obj_vals.append(Obj_func(x))
        new_x = np.random.uniform(low=-bound, high=bound, size=dim)
        if Obj_func(new_x) < Obj_func(x):
            x = new_x

    return obj_vals


def generate_RS_runs(complex_dictionary, bound):
    list_of_runs = []
    flat_start = [x for y in complex_dictionary["starting_x"] for x in y]
    A_bs = generate_ab_pairs(complex_dictionary)
    for (A, b), x in zip(A_bs, flat_start):
        list_of_runs.append(RS(A, b, x, 101, bound))
    return list_of_runs


def generate_PGD_runs(complex_dictionary, bound):
    list_of_runs = []
    flat_start = [x for y in complex_dictionary["starting_x"] for x in y]
    A_bs = generate_ab_pairs(complex_dictionary)
    for (A, b), x in zip(A_bs, flat_start):
        list_of_runs.append(PGD(A, b, x, 101, bound))
    return list_of_runs


def unravel_c_dic(complex_dictionary, length=101):
    dim = len(complex_dictionary["best_list"])

    n = 0
    array_list = []
    while (n + 1) * 101 < dim + 1:
        array = np.asarray(complex_dictionary["best_list"][n * 101 : (n + 1) * 101])
        array_list.append(array)
        n += 1

    return np.concatenate(array_list, axis=1)


def unravel_c_dic_lastbest(complex_dictionary, length=101):
    dim = len(complex_dictionary["best_list"])

    n = 0
    array_list = []
    lastbest = []
    while (n + 1) * 101 < dim + 1:
        array = np.asarray(complex_dictionary["best_list"][n * 101 : (n + 1) * 101])
        array_list.append(array)
        n += 1
    return_array = np.concatenate(array_list, axis=1)
    lastbest = return_array[-1]
    return lastbest


def generate_action_stack(complex_dictionary):
    """
    Code for fixed dim
    
    """
    my_np_arrays = []
    for i in range(int(10000 / 100)):
        current_np = np.asarray(
            (complex_dictionary["action_list"][i * (100) : (i + 1) * 100])
        )
        my_np_arrays.append(current_np)

    stacked_data = np.concatenate(my_np_arrays, axis=1)

    return stacked_data


"""
End of plotting helper functions
"""


def process_dm_name(dm):
    dm = str(dm)
    dm = dm.replace("[", "")
    dm = dm.replace("]", "")
    dm = dm.replace("_", ", ")
    return dm


def plot_box(dm, save_subfolder, pickle_tuple, yaml_config, multi=False):
    env_config = yaml_config["env_config"]

    plot_tag = yaml_config["plot_tag"]
    PAdam_trial, PGD_trial, _, RS_trial, complex_dictionary = pickle_tuple
    ml_PAdam = np.transpose(unravel_c_dic(complex_dictionary)) - np.asarray(PAdam_trial)
    ml_pgd = np.transpose(unravel_c_dic(complex_dictionary)) - np.asarray(PGD_trial)
    ml_RS = np.transpose(unravel_c_dic(complex_dictionary)) - np.asarray(RS_trial)

    dm = process_dm_name(dm)

    for x, y in [
        (ml_PAdam, "Adam"),
        (ml_pgd, "Gradient Decscent"),
        (ml_RS, "Random Search"),
    ]:
        plt.figure(figsize=(15, 5))
        boxdict = plt.boxplot(
            x,
            showfliers=False,
            showmeans=True,
            medianprops={"Color": "Red", "label": "Median"},
            meanprops={"label": "Mean"},
            widths=0.5,
        )

        plt.xticks(rotation=40)
        plt.rc("xtick", labelsize=7)
        plt.xticks(list(range(1, 102)), list(range(101)))

        plt.legend([boxdict["medians"][0], boxdict["means"][0]], ["Median", "Mean"])
        # plt.tight_layout()
        if multi:
            plot_title = f"{plot_tag}: {dm} mixed dimentional box plot of ML - {y} (Negative is better) "
        else:
            plot_title = (
                f"{plot_tag}: {dm} dimention box plot of ML - {y} (Negative is better) "
            )
        plt.title(
            plot_title, wrap=True,
        )
        plt.xlabel("number of steps")
        plt.ylabel("Objective value difference")
        plt.grid(True, axis="y")
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

        savepath = save_subfolder + "plots/" + f"Boxplot_dm_{dm}_type_{y}_{now}.png"
        ensure_dir(savepath)
        print(savepath)
        plt.savefig(savepath, bbox_inches="tight")
        plt.clf()

    return 0


def plot_action_dist(dm, save_subfolder, pickle_tuple, yaml_config, multi=False):

    _, _, _, _, complex_dictionary = pickle_tuple
    stacked_data = generate_action_stack(complex_dictionary)
    _, n_runs = stacked_data.shape
    env_config = yaml_config["env_config"]
    plot_tag = yaml_config["plot_tag"]
    if env_config["switching_style"] == "AdamGD":
        stacked_data += 1
    if env_config["switching_style"] == "RandAdam":
        stacked_data *= 2
    plt.figure(figsize=(8, 6))
    plt.plot(np.count_nonzero(stacked_data == 0, axis=1) / n_runs, label="Random")
    plt.plot(np.count_nonzero(stacked_data == 1, axis=1) / n_runs, label="Gradient")
    plt.plot(np.count_nonzero(stacked_data == 2, axis=1) / n_runs, label="Adam")
    plt.axis([0, 100, 0, 1.0])
    plt.grid()
    plt.legend()

    if multi:
        dm = process_dm_name(dm)
        plot_title = f"{plot_tag}: {dm} mixed dimentional Action distrubtion"
    else:
        plot_title = f"{plot_tag}: {dm} dimentional Action distrubtion"

    plt.title(
        plot_title, wrap=True,
    )
    plt.xlabel("number of steps")
    plt.ylabel("Probability of action")

    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

    savepath = save_subfolder + "plots/" + f"Action_dist_dm_{dm}_{now}.png"
    ensure_dir(savepath)
    print(savepath)
    plt.savefig(savepath, bbox_inches="tight")
    plt.clf()

    return 0


def plot_mean_quartile(dm, save_subfolder, pickle_tuple, yaml_config, multi=False):
    """
    generalize to multi-dimentional case
    
    """
    env_config = yaml_config["env_config"]
    plot_tag = yaml_config["plot_tag"]
    PAdam_trial, PGD_trial, _, RS_trial, complex_dictionary = pickle_tuple

    ml_PAdam = np.transpose(unravel_c_dic(complex_dictionary)) - np.asarray(PAdam_trial)
    ml_pgd = np.transpose(unravel_c_dic(complex_dictionary)) - np.asarray(PGD_trial)
    # multiple line plot

    ml_PAdam_data = np.mean(ml_PAdam, axis=0)
    ml_PAdam_q1 = np.quantile(ml_PAdam, 0.25, axis=0)
    ml_PAdam_q3 = np.quantile(ml_PAdam, 0.75, axis=0)

    ml_pgd_data = np.mean(ml_pgd, axis=0)
    ml_pgd_q1 = np.quantile(ml_pgd, 0.25, axis=0)
    ml_pgd_q3 = np.quantile(ml_pgd, 0.75, axis=0)

    xvalues = list(range(len(ml_PAdam_data)))
    plt.figure(figsize=(8, 6))
    plt.plot(ml_PAdam_data, "blue", label="ML -Adam mean")
    plt.plot(ml_PAdam_q1, "lightblue", linestyle=":")
    plt.plot(ml_PAdam_q3, "lightblue", linestyle="--")
    plt.fill_between(
        xvalues,
        ml_PAdam_q1,
        ml_PAdam_q3,
        color="lightblue",
        alpha=0.3,
        label="ML - Adam Q1 to Q3",
    )

    xvalues = list(range(len(ml_pgd_data)))
    plt.plot(ml_pgd_data, "red", label=" ML - SGD mean")
    plt.plot(ml_pgd_q1, "salmon", linestyle=":")
    plt.plot(ml_pgd_q3, "salmon", linestyle="--")
    plt.fill_between(
        xvalues,
        ml_pgd_q1,
        ml_pgd_q3,
        color="salmon",
        alpha=0.3,
        label="ML - SGD Q1 to Q3",
    )
    if multi:
        dm = process_dm_name(dm)
        plot_title = f"{plot_tag}: {dm} mixed dimension difference  Quartile plot (Negative is better)"
    else:
        plot_title = (
            f"{plot_tag}: {dm} dimension difference  Quartile plot (Negative is better)"
        )
    plt.title(
        plot_title, wrap=True,
    )
    plt.xlabel("number of steps")
    plt.ylabel("mean objective value")

    plt.grid()

    plt.legend()
    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

    savepath = save_subfolder + "plots/" + f"diff_quartile_plot_dm_{dm}_{now}.png"
    ensure_dir(savepath)
    print(savepath)
    plt.savefig(savepath, bbox_inches="tight")
    plt.clf()

    ml_data_mean = np.mean(np.transpose(unravel_c_dic(complex_dictionary)), axis=0)
    pgd_data_mean = np.mean(np.asarray(PGD_trial), axis=0)
    RS_data_mean = np.mean(np.asarray(RS_trial), axis=0)
    PAdam_data_mean = np.mean(np.asarray(PAdam_trial), axis=0)

    plt.plot(ml_data_mean, label="Learned Optimizer")
    plt.plot(pgd_data_mean, label="Stochastic Gradient")
    plt.plot(PAdam_data_mean, label="Adam")
    plt.plot(RS_data_mean, label="Random Search")

    mean_plot_title = f"{dm} dim mean plot"
    plt.title(
        mean_plot_title, wrap=True,
    )
    plt.xlabel("number of steps")
    plt.ylabel("mean objective value")

    plt.grid()

    plt.legend()

    savepath = save_subfolder + "plots/" + f"meanonly_dm_{dm}_{now}.png"
    ensure_dir(savepath)
    print(savepath)
    plt.savefig(savepath, bbox_inches="tight")
    plt.clf()

    return 0


def generate_plot_data(
    dm,
    save_string,
    save_subfolder,
    env_config,
    yaml_config,
    multi=False,
    savedata_type=0,
):
    if multi:
        confignew = env_config

    else:
        confignew = env_config
        confignew["dim"] = [dm]

    if "ppo2_nsteps" in yaml_config.keys():
        ppo2_nsteps = yaml_config["ppo2_nsteps"]
    else:
        ppo2_nsteps = 1000
    if "policy_kwargs" in yaml_config.keys():
        policy_kwargs = yaml_config["policy_kwargs"]
    else:
        policy_kwargs = None

    check_env(SwitchingquadraticEnv(confignew))
    bound = env_config["bound"]
    num_cpu = yaml_config["num_cpus"]
    env = SubprocVecEnv(
        [make_env(i, confignew) for i in range(num_cpu)], start_method="spawn"
    )

    print(env)

    model = PPO2(
        MlpPolicy,
        env,
        policy_kwargs=policy_kwargs,
        n_steps=ppo2_nsteps,
        nminibatches=2,
        verbose=1,
    )
    model = PPO2.load(save_string)

    complex_dictionary = complex_evaluate_vec(
        model, env, num_steps=10000, agent_dim=num_cpu
    )
    PAdam_trial = generate_PAdam_runs(complex_dictionary, bound)
    print(f'PAdam_trail len: {len(PAdam_trial)}')
    RS_trial = generate_RS_runs(complex_dictionary, bound)
    PGD_trial = generate_PGD_runs(complex_dictionary, bound)
    if multi:
        pickle_savepath = (
            save_subfolder + "data/" + f"{env_config['dim']}_diff_quartile.pickle"
        )
    else:
        pickle_savepath = save_subfolder + "data/" + f"{dm}_diff_quartile.pickle"
    print(f"pickle save path: {pickle_savepath}")
    pickle_tuple = (
        PAdam_trial,
        PGD_trial,
        np.transpose(unravel_c_dic(complex_dictionary)),
        RS_trial,
        complex_dictionary,
    )
    if savedata_type == 2:
        ensure_dir(pickle_savepath)
        pickle.dump(pickle_tuple, open(pickle_savepath, "wb"))
    elif savedata_type == 1:
        ensure_dir(pickle_savepath)
        lastbest_list = unravel_c_dic_lastbest(complex_dictionary, length=101)
        pickle.dump(
            generate_Abx0(complex_dictionary, lastbest_list),
            open(pickle_savepath, "wb"),
        )

    return pickle_tuple


def update_score(pickle_tuple):
    """
    Should be the mean on a run, and the we can compute mean/variance across runs
    """
    PAdam_trial, PGD_trial, _, RS_trial, complex_dictionary = pickle_tuple
    ml_PAdam = np.transpose(unravel_c_dic(complex_dictionary)) - np.asarray(PAdam_trial)
    ml_pgd = np.transpose(unravel_c_dic(complex_dictionary)) - np.asarray(PGD_trial)
    ml_RS = np.transpose(unravel_c_dic(complex_dictionary)) - np.asarray(RS_trial)
    return_list = []
    for x in [ml_PAdam, ml_pgd, ml_RS]:
        run_means = np.mean(x, axis=1)
        size = max(run_means.shape)
        total_score = np.mean(run_means)
        std = np.sqrt(np.var(run_means) * (size / (size - 1)))
        return_list.append((total_score, std))
    return return_list


def plotit(yaml_file, save_string, generate_newdata=False, savedata_type=0):
    with open(yaml_file) as infile:
        yaml_config = yaml.load(infile, Loader=yaml.FullLoader)

    basepath = yaml_config["Basepath"]
    env_config = yaml_config["env_config"]
    #
    save_string = save_string.rstrip(".zip")
    save_subfolder = basepath + "/" + save_string.lstrip(basepath + "agents/") + "/"

    """
    Plotting restart here.
    """
    Adamscore, GDscore, RSscore = [], [], []
    dm = str(env_config["dim"]).replace(", ", "_")

    multi = True
    if generate_newdata:
        pickle_tuple = generate_plot_data(
            dm,
            save_string,
            save_subfolder,
            env_config,
            yaml_config,
            multi=multi,
            savedata_type=savedata_type,
        )
        plot_mean_quartile(dm, save_subfolder, pickle_tuple, yaml_config, multi)
        plot_action_dist(dm, save_subfolder, pickle_tuple, yaml_config, multi)
        plot_box(dm, save_subfolder, pickle_tuple, yaml_config, multi)
    else:
        # os.path.exists()
        if multi:
            pickle_savepath = (
                save_subfolder + "data/" + f"{env_config['dim']}_diff_quartile.pickle"
            )
        else:
            pickle_savepath = save_subfolder + "data/" + f"{dm}_diff_quartile.pickle"

        if os.path.exists(pickle_savepath):
            with open(pickle_savepath, "rb") as pickle_file:
                pickle_tuple = pickle.load(pickle_file)
            plot_mean_quartile(dm, save_subfolder, pickle_tuple, yaml_config, multi)
            plot_action_dist(dm, save_subfolder, pickle_tuple, yaml_config, multi)
            plot_box(dm, save_subfolder, pickle_tuple, yaml_config, multi)
        else:
            pickle_tuple = generate_plot_data(
                dm, save_string, save_subfolder, env_config, yaml_config, multi
            )
            plot_mean_quartile(dm, save_subfolder, pickle_tuple, yaml_config, multi)
            plot_action_dist(dm, save_subfolder, pickle_tuple, yaml_config, multi)
            plot_box(dm, save_subfolder, pickle_tuple, yaml_config, multi)
    scores = update_score(pickle_tuple)
    Adamscore.append(scores[0])
    GDscore.append(scores[1])
    RSscore.append(scores[2])

    multi = False
    for dm in yaml_config["plot_dims"]:

        if generate_newdata:
            pickle_tuple = generate_plot_data(
                dm,
                save_string,
                save_subfolder,
                env_config,
                yaml_config,
                multi=multi,
                savedata_type=savedata_type,
            )
            plot_mean_quartile(dm, save_subfolder, pickle_tuple, yaml_config, multi)
            plot_action_dist(dm, save_subfolder, pickle_tuple, yaml_config, multi)
            plot_box(dm, save_subfolder, pickle_tuple, yaml_config, multi)
        else:
            # os.path.exists()
            if multi:
                pickle_savepath = (
                    save_subfolder
                    + "data/"
                    + f"{env_config['dim']}_diff_quartile.pickle"
                )
            else:
                pickle_savepath = (
                    save_subfolder + "data/" + f"{dm}_diff_quartile.pickle"
                )

            if os.path.exists(pickle_savepath):
                with open(pickle_savepath, "rb") as pickle_file:
                    pickle_tuple = pickle.load(pickle_file)
                plot_mean_quartile(dm, save_subfolder, pickle_tuple, yaml_config, multi)
                plot_action_dist(dm, save_subfolder, pickle_tuple, yaml_config, multi)
                plot_box(dm, save_subfolder, pickle_tuple, yaml_config, multi)
            else:
                pickle_tuple = generate_plot_data(
                    dm,
                    save_string,
                    save_subfolder,
                    env_config,
                    yaml_config,
                    multi=multi,
                    savedata_type=savedata_type,
                )
                plot_mean_quartile(dm, save_subfolder, pickle_tuple, yaml_config, multi)
                plot_action_dist(dm, save_subfolder, pickle_tuple, yaml_config, multi)
                plot_box(dm, save_subfolder, pickle_tuple, yaml_config, multi)

        scores = update_score(pickle_tuple)
        Adamscore.append(scores[0])
        GDscore.append(scores[1])
        RSscore.append(scores[2])

    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    pickle_savepath_scoredata = (
        save_subfolder + "smalldata/" + f"score_data_{now}.pickle"
    )
    print(f"pickle save path: {pickle_savepath_scoredata}")
    pickle_tuple = (
        Adamscore,
        GDscore,
        RSscore,
    )
    ensure_dir(pickle_savepath_scoredata)
    pickle.dump(pickle_tuple, open(pickle_savepath_scoredata, "wb"))
    return 0
