import pickle
import argparse

from tabulate import tabulate


def construct_table_data(data, sigma_style=0, dim_list_override=None):
    table_data = []
    if dim_list_override is None:
        dim_list = [
            "5, 10, 15, 25 mixed",
            5,
            7,
            10,
            12,
            15,
            18,
            20,
            25,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
        ]
    else:
        dim_list = dim_list_override
    print("Printing raw table data")
    for x in range(len(dim_list)):
        row = [dim_list[x]]
        if sigma_style == 1:
            for i in range(3):
                append_str = (
                    str(round(data[i][x][0], 2))
                    + " $\sigma=$ "
                    + str(round(data[i][x][1], 2))
                )
                row.append(append_str)
        elif sigma_style == 0:
            for i in range(3):
                append_str = str(round(data[i][x][0], 2))
                row.append(append_str)
        elif sigma_style == 2:
            for i in range(3):
                append_str = str(round(data[i][x][0], 2))
                row.append(append_str)
                row.append(str(round(data[i][x][1], 2)))

        table_data.append(row)
        print(row)
    print("End of raw table data")
    return table_data


def print_table(table_data, title, tablefmt="simple", sigma_style=0):
    if sigma_style == 2:
        headers = [
            "dimension",
            "Score - Adam",
            "sigma",
            "Score - GD",
            "sigma",
            "Score - Random Search",
            "sigma",
        ]
    else:
        headers = ["dimension", "Score - Adam", "Score - GD", "Score - Random Search"]
    if tablefmt == "github":
        print(f"### {title}")
    if tablefmt == "latex_raw":
        print(tabulate(table_data, headers, tablefmt="latex_raw", colalign=("right",)))
    else:
        print(tabulate(table_data, headers, tablefmt=tablefmt))


def main():
    """
    A basic commandline tool for generating tables from our saved data

    """
    parser = argparse.ArgumentParser(
        description="Generate table summary for our experiments from our saved small data."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="the path to saved testing smalldata for RLSO",
    )
    parser.add_argument(
        "--sigma_type",
        type=int,
        default=0,
        help="How to show sigma data. 0 shows no sigma data. 1 shows sigma data in the same column. 2 shows sigma data in a separate column. ",
    )
    parser.add_argument(
        "--titlestring",
        type=str,
        default=None,
        help="String to append to our plot title. Defaults to None",
    )
    parser.add_argument(
        "--dim_list",
        type=list,
        default=None,
        help="List of dimensions the experiments were run on.",
    )
    parser.add_argument(
        "--table_type",
        type=int,
        default=0,
        help="Type of table. 0 for Latex, and 1 for markdown for github.",
    )
    args = parser.parse_args()

    if args.path is None:
        print("Path to pickle data needed!")
        return

    pickle_savepath = args.path
    with open(pickle_savepath, "rb") as pickle_file:
        data = pickle.load(pickle_file)

    table_data = construct_table_data(
        data, sigma_style=args.sigma_type, dim_list_override=args.dim_list
    )

    if args.table_type == 0:
        table_type = "latex_raw"
    else:
        table_type = "github"

    print_table(
        table_data, args.titlestring, tablefmt=table_type, sigma_style=args.sigma_type
    )
    return


if __name__ == "__main__":
    main()
