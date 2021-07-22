import pickle
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

import matlab.engine

eng = matlab.engine.start_matlab()


def unravel_c_dic_lastbest(complex_dictionary):
    """
    Selected the last best result from our runs. Warning length not properly implemented.
    """
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


def generate_ab_pairs(complex_dictionary):
    list_of_pairs = []
    for (A, b) in complex_dictionary["A_b_list"]:
        for a_i, b_i in zip(A, b):
            list_of_pairs.append((a_i, b_i))
    return list_of_pairs


def generate_Abx0(complex_dictionary, lastbest_list):
    list_triples = []
    flat_start = [x for y in complex_dictionary["starting_x"] for x in y]
    A_bs = generate_ab_pairs(complex_dictionary)
    for (A, b), x, fval in zip(A_bs, flat_start, lastbest_list):
        list_triples.append((A, b, x, fval))
    return list_triples


def quad_solve(Abx0, printout=False):
    A, b, x0, fval = Abx0
    blank = matlab.double([])
    opts_active = eng.optimoptions(
        "quadprog", "Algorithm", "active-set", "MaxIterations", 100
    )
    opts_int = eng.optimoptions(
        "quadprog", "Algorithm", "interior-point-convex", "MaxIterations", 100
    )
    opts_trust = eng.optimoptions(
        "quadprog", "Algorithm", "trust-region-reflective", "MaxIterations", 100
    )
    dim = len(x0)
    H = 2.0 * A
    mat_H = matlab.double(H.tolist())
    mat_b = matlab.double(b.tolist())
    mat_x0 = matlab.double(x0.tolist())
    mat_lb = matlab.double([-10.0] * dim)
    mat_ub = matlab.double([10.0] * dim)
    [x_active, fval_active, exitflag, output] = eng.quadprog(
        mat_H,
        mat_b,
        blank,
        blank,
        blank,
        blank,
        mat_lb,
        mat_ub,
        mat_x0,
        opts_active,
        nargout=4,
    )
    if printout:
        print(exitflag)
        print(output)
        print(fval_active)
    [x_int, fval_int, exitflag, output] = eng.quadprog(
        mat_H,
        mat_b,
        blank,
        blank,
        blank,
        blank,
        mat_lb,
        mat_ub,
        mat_x0,
        opts_int,
        nargout=4,
    )
    if printout:
        print(exitflag)
        print(output)
        print(fval_int)
    [x_trust, fval_trust, exitflag, output] = eng.quadprog(
        mat_H,
        mat_b,
        blank,
        blank,
        blank,
        blank,
        mat_lb,
        mat_ub,
        mat_x0,
        opts_trust,
        nargout=4,
    )
    if printout:
        print(exitflag)
        print(output)
        print(fval_trust)
    return [fval, fval_active, fval_int, fval_trust]


def plot_box(rlso, active, inter, trust, title_string, save_string, exclude_inter=True):

    plot_tag = f"Matlab quadprog Algorithms vs RLSO. {title_string}"

    if exclude_inter:
        """
        Interior point method doesn't allow us to set a starting point, and
        produces extremely bad results on average because of that.
        """
        data = [rlso, active, trust]
    else:
        data = [rlso, active, inter, trust]
    plt.figure(figsize=(6, 6))
    boxdict = plt.boxplot(
        data,
        showfliers=False,
        showmeans=True,
        medianprops={"Color": "Red", "label": "Median"},
        meanprops={"label": "Mean"},
        widths=0.5,
    )

    plt.legend([boxdict["medians"][0], boxdict["means"][0]], ["Median", "Mean"])
    # plt.tight_layout()

    plt.title(
        plot_tag, wrap=True,
    )
    plt.xlabel("Algorithms")
    plt.ylabel("Function value")
    plt.grid(True, axis="y")
    if exclude_inter:
        labels = ["RLSO", "Active-set", "Trust Region"]
        plt.xticks([1, 2, 3], labels)
    else:
        labels = ["RLSO", "Active-set", "Interior point", "Trust Region"]
        plt.xticks([1, 2, 3, 4], labels)

    plt.savefig(save_string, bbox_inches="tight")
    # plt.clf()

    return 0


def main():
    """
    A basic commandline tool for running matlab experiments

    """
    parser = argparse.ArgumentParser(
        description="Generate bar plots comparing Matlab solvers to RLSO"
    )
    parser.add_argument(
        "--path", type=str, default=None, help="the path to saved testing data for RLSO"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1000,
        help="Number of test problems to use defaults to 1000",
    )
    parser.add_argument(
        "--savestring",
        type=str,
        default=None,
        help="Name to save our plot as. Defaults to random int .png",
    )
    parser.add_argument(
        "--titlestring",
        type=str,
        default=None,
        help="String to append to our plot title. Defaults to None",
    )
    parser.add_argument(
        "--excludeinter",
        type=bool,
        default=True,
        help="Exclude interior point method form testing. Reasons Convex only, and cannot set starting point. Default True",
    )

    args = parser.parse_args()

    if args.path is None:
        print("Path to pickle data needed!")
        return

    pickle_savepath = args.path
    with open(pickle_savepath, "rb") as pickle_file:
        pickle_tuple = pickle.load(pickle_file)

    complex_dic = pickle_tuple[4]
    lastbest = unravel_c_dic_lastbest(complex_dic)
    Abx0 = generate_Abx0(complex_dic, lastbest)

    if args.trials is None:
        trials = 1000
    else:
        trials = args.trials
    if args.savestring is None:
        save_string = str(np.random.randint(low=1e15, high=9e15)) + ".png"
    else:
        save_string = args.savestring

    start = time.time()
    print(f"start time {start}")

    solutions = []
    for tuples in Abx0[:trials]:
        solutions.append(quad_solve(tuples, printout=False))
    end = time.time()
    print(f"Matlab run time: {end - start}")

    rlso = []
    active = []
    inter = []
    trust = []
    for [x, y, z, w] in solutions:
        rlso.append(x)
        active.append(y)
        inter.append(z)
        trust.append(w)
    title_string = args.titlestring
    plot_box(
        rlso,
        active,
        inter,
        trust,
        title_string,
        save_string,
        exclude_inter=args.excludeinter,
    )

    return


if __name__ == "__main__":
    main()
