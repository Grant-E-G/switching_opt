import argparse
from train_generic import trainit, plotit


def main():
    """
    A basic commandline wrapper for our training and plotting functions.

    """
    parser = argparse.ArgumentParser(
        description="Train a switching optimizer and plot output"
    )
    parser.add_argument("--config", type=str, help="the path to the configuration file")
    parser.add_argument("--train", type=bool, default=True, help="train a new agent")
    parser.add_argument(
        "--plottrained",
        type=bool,
        default=True,
        help="create plots for the agent trained ",
    )
    parser.add_argument(
        "--plotonly",
        type=str,
        default="None",
        help="creates plots for a previous trained agent at path",
    )
    parser.add_argument(
        "--newplotdata",
        type=bool,
        default=False,
        help="Generate new data for the plots",
    )
    parser.add_argument(
        "--retrain",
        type=str,
        default="None",
        help="Location for model to be retrained",
    )

    parser.add_argument(
        "--savedata_type",
        type=int,
        default=0,
        help="0,1,2 save nothing, save problem & best solution, save everything",
    )

    args = parser.parse_args()
    print(args.config)
    if args.train and args.plotonly == "None":
        save_string = trainit(args.config)
        if args.plottrained:
            plotit(args.config, save_string)
    elif args.retrain != "None":
        save_string = trainit(args.config, retrain=args.retrain)
        plotit(args.config, save_string)
    elif args.plotonly != "None":
        plotit(args.config, args.plotonly, args.newplotdata, args.savedata_type)
    else:
        print("Please provide a path for plotting")


if __name__ == "__main__":
    main()
