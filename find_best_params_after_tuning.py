import pathlib
import pandas as pd

# dir with run_x in the name are the tuned params.
paths = pathlib.Path("./")

best_mean_fitness = 0
best_filename = None

for d in paths.iterdir():
    if (
        d.is_dir()
        and d.name.find("crossover") != -1
        and d.name.split("__")[-1][-1] == "x"
    ):
        df = pd.read_csv(f"./{d.name}/results.txt", delimiter=" ")
        best_mean_across_all_generations = df["best"].mean()
        if best_mean_across_all_generations > best_mean_fitness:
            best_mean_fitness = best_mean_across_all_generations
            best_filename = d.name


with open("./best_tuned_params.txt", "w") as f:
    f.write("best_mean_fitness best_filename")
    f.write(f"\n{best_mean_fitness} {best_filename}")

print(best_mean_fitness, best_filename)
