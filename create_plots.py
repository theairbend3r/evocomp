import pathlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_lineplot():
    paths = pathlib.Path("./")
    dir_names = []

    for d in paths.iterdir():
        if (
            d.is_dir()
            and d.name.find("crossover") != -1
            and d.name.split("__")[-1][-1].isdigit()
        ):
            dir_names.append(d.name)

    train_df = []
    for f in dir_names:
        df = pd.read_csv(f"{f}/results.txt", delimiter=" ")
        df["enemies"] = f.split("__")[2].split("_")[-1]
        df["run"] = f.split("__")[-1].split("_")[-1]
        df["crossover"] = f.split("__")[0].split("_")[-1]
        train_df.append(df)

    train_df = pd.concat(train_df)
    train_df = train_df.reset_index(drop=True)

    crossover = train_df["crossover"].unique().tolist()
    enemies = train_df["enemies"].unique().tolist()

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)
    fig.suptitle("Best and Average Fitness for EA-Enemy Combo")
    for e in range(len(enemies)):
        for c in range(len(crossover)):
            # entities to plot
            mean_of_mean_series = (
                train_df[
                    (train_df["crossover"] == crossover[c])
                    & (train_df["enemies"] == enemies[e])
                ]
                .groupby(["gen"])["mean"]
                .mean()
            )

            mean_of_max_series = (
                train_df[
                    (train_df["crossover"] == crossover[c])
                    & (train_df["enemies"] == enemies[e])
                ]
                .groupby(["gen"])["best"]
                .mean()
            )

            # convert to dataframe for auto-legend labels
            df_to_plot = pd.DataFrame(
                {
                    "Mean": mean_of_mean_series,
                    "Max": mean_of_max_series,
                }
            )

            sns.lineplot(data=df_to_plot, ax=axes[e, c])
            axes[e, c].set(
                xlabel="Generations",
                ylabel="Fitness",
                title=f"Crossover-{crossover[c]} Enemy-{enemies[e]}",
            )
            plt.tight_layout()

    fig.savefig("./lineplot.png")
    plt.show()


def create_boxplot_enemy_group():
    test_df = pd.read_csv("./test_results.txt", delimiter=" ")
    print(test_df)

    crossover = test_df["crossover"].unique().tolist()
    enemies = test_df["enemies"].unique().tolist()
    print(crossover, enemies)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    fig.tight_layout()
    fig.suptitle("Gain Box Plots for Enemy Groups")
    for e in range(len(enemies)):
        for c in range(len(crossover)):
            # entities to plot
            mean_of_fitness_series = (
                test_df[
                    (test_df["crossover"] == crossover[c])
                    & (test_df["enemies"] == enemies[e])
                ]
                .groupby(["run"])["fitness"]
                .mean()
            )

            test_df_to_plot = pd.DataFrame(
                {
                    f"EA {c+1}": mean_of_fitness_series,
                }
            )

            sns.boxplot(data=test_df_to_plot, ax=axes[e, c])
            axes[e, c].set(
                ylabel="Individual Gain",
                title=f"Trained on: {' '.join(str(enemies[e]))}, Tested on: All",
            )
            plt.tight_layout()

    fig.savefig("./boxplot_enemy_group.png")
    plt.show()


def create_barplot():
    df = pd.read_csv("./results_of_best_against_all/test_results_all_enemies.txt", delimiter=' ')
    sns.barplot(data=df, x='enemy', y='gain').set(title='Best indvidual\'s gain against all enemies')
    plt.savefig("./barplot_enemies.png")

    plt.show()


if __name__ == "__main__":
    create_lineplot()
    create_boxplot_enemy_group()
    create_barplot()