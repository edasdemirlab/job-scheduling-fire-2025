import os
import pandas as pd
import matplotlib.pyplot as plt

# Folder path
folder_path = "cluster_first_ccr_experiments/erdi_parameters"

# File names
file_names = [
    "run_no_1_value_n0_d0.csv",
    "run_no_2_value_n0_d1.csv",
    "run_no_3_value_n1_d0.csv",
    "run_no_4_value_n1_d1.csv",
    "run_no_5_value_n2_d0.csv",
    "run_no_6_value_n2_d1.csv",
    "run_no_13_value_n0_d0.csv",
    "run_no_14_value_n0_d1.csv",
    "run_no_15_value_n1_d0.csv",
    "run_no_16_value_n1_d1.csv",
    "run_no_17_value_n2_d0.csv",
    "run_no_18_value_n2_d1.csv",
    "run_no_0_combination-results-2uav-60v-erdi.csv"
]

# Extract only "run_no_x" part for x-axis labels
short_labels = {file: "_".join(file.split("_")[:3]) for file in file_names}

# Initialize a DataFrame
data = []

# Read each file and append data
for file in file_names:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    df["File"] = short_labels[file]  # Assign cleaned labels: "run_no_x"
    data.append(df)

# Combine all data into one DataFrame
df_all = pd.concat(data, ignore_index=True)

# Create and save ordered boxplots for each column
columns_to_plot = ["total_value", "gap", "total_python_time"]
for col in columns_to_plot:
    # Compute medians for sorting
    median_values = df_all.groupby("File")[col].median().sort_values()

    # Reorder data based on median values
    ordered_files = median_values.index.tolist()
    df_all["File"] = pd.Categorical(df_all["File"], categories=ordered_files, ordered=True)

    # Plot the ordered boxplot
    plt.figure(figsize=(16, 8))
    df_all.boxplot(column=col, by="File", vert=True, grid=False)
    plt.title(f"Boxplot of {col} (Ordered by Median)", fontsize=16)
    plt.suptitle("")  # Remove default title
    plt.xlabel("Experiment (Run Number)", fontsize=14)
    plt.ylabel(col, fontsize=14)
    plt.xticks(rotation=45, fontsize=12)  # Rotate x-axis labels for readability
    plt.yticks(fontsize=12)
    plt.savefig(f"{col}_boxplot_sorted.png", bbox_inches="tight")  # Save plot
    plt.close()


print("Boxplots saved successfully.")



# Extract `total_python_time` for `run_no_0` and `run_no_2`
df_run_0 = df_all[df_all["File"] == "run_no_0"][["total_python_time"]].reset_index(drop=True)
df_run_2 = df_all[df_all["File"] == "run_no_2"][["total_python_time"]].reset_index(drop=True)

# Ensure both have 315 rows
if len(df_run_0) == 315 and len(df_run_2) == 315:
    # Scatter plot comparing `total_python_time` of `run_no_0` vs `run_no_2`
    plt.figure(figsize=(10, 6))
    plt.scatter(df_run_0["total_python_time"], df_run_2["total_python_time"], alpha=0.7, color="blue")
    plt.title("Comparison of total_python_time: run_no_0 vs. run_no_2", fontsize=14)
    plt.xlabel("total_python_time (run_no_0)", fontsize=12)
    plt.ylabel("total_python_time (run_no_2)", fontsize=12)
    plt.grid(True)
    plt.savefig("scatter_run0_vs_run2.png", bbox_inches="tight")  # Save the plot
    plt.close()
else:
    print(f"Warning: Mismatched row counts (run_no_0: {len(df_run_0)}, run_no_2: {len(df_run_2)}). Check data integrity.")



# Extract `total_value` for `run_no_0` and `run_no_2`
df_run_0 = df_all[df_all["File"] == "run_no_0"][["total_value"]].reset_index(drop=True)
df_run_2 = df_all[df_all["File"] == "run_no_2"][["total_value"]].reset_index(drop=True)

# Ensure both have 315 rows before plotting
if len(df_run_0) == 315 and len(df_run_2) == 315:
    # Scatter plot comparing `total_value` of `run_no_0` vs `run_no_2`
    plt.figure(figsize=(10, 6))
    plt.scatter(df_run_0["total_value"], df_run_2["total_value"], alpha=0.7, color="green")
    # 1:1 Reference Line
    min_val = min(df_run_0["total_value"].min(), df_run_2["total_value"].min())
    max_val = max(df_run_0["total_value"].max(), df_run_2["total_value"].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="1:1 Reference Line (y=x)")

    plt.title("Comparison of total_value: run_no_0 vs. run_no_2", fontsize=14)
    plt.xlabel("total_value (run_no_0)", fontsize=12)
    plt.ylabel("total_value (run_no_2)", fontsize=12)
    plt.grid(True)
    plt.savefig("scatter_run0_vs_run2_total_value.png", bbox_inches="tight")  # Save the plot

    print("Scatter plot saved as scatter_run0_vs_run2_total_value.png")
else:
    print(f"Warning: Mismatched row counts (run_no_0: {len(df_run_0)}, run_no_2: {len(df_run_2)}). Check data integrity.")