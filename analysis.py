import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch  # Added to avoid NameError

# Assume const.DEVICE is defined elsewhere in your code
# For demonstration, let's define a default:
class Const:
    DEVICE = "cpu"  # Or "cuda" if available and desired
const = Const()


def visualize_correlations():
    """
    Loads data, calculates correlations between X stats and y values,
    generates visualizations for those correlations, and calculates
    and visualizes feature cross-correlations.  Handles missing teams
    and device placement.
    """

    if not os.path.exists("cache/results.csv") or not os.path.exists("cache/teams.csv"):
        print("Missing cache files.  Run clean.run() first or ensure the files exist.")  # Print message instead of assuming a "clean" function.
        return  # Exit function since files are missing

    odds = pd.read_csv("cache/results.csv")
    odds = odds.drop(["win_team2", "win_pct_team2"], axis=1)
    teams = pd.read_csv("cache/teams.csv")
    teams = teams.set_index("SchoolY").T.to_dict()
    odds = np.array(odds)
    X_temp = odds[:, :2]
    X_list = []  # Use lists for dynamic appending before numpy conversion
    y_list = []

    for i, (t1, t2) in enumerate(X_temp):
        if t1 not in teams or t2 not in teams:
            # print(f"Skipping row {i}: {t1}, {t2} - one or both teams not found")
            continue

        team1_values = np.array(list(teams[t1].values()))
        team2_values = np.array(list(teams[t2].values()))
        new_row_x = np.concatenate([team1_values, team2_values])  # Flatten the team data into a single row
        X_list.append(new_row_x)
        y_list.append(odds[i, 2:])

    X = np.array(X_list)
    y = np.array(y_list, dtype=np.float64)

    # Convert to tensors *after* the data processing and cleaning
    X = torch.tensor(X, dtype=torch.float32).to(const.DEVICE)
    y = torch.tensor(y, dtype=torch.float32).to(const.DEVICE)

    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()


    # Calculate correlations for y[0]
    correlations_y0 = np.corrcoef(X_np.T, y_np[:, 0])[
        :-1, -1
    ]  # Correlations of each X feature with y[0]

    # Calculate correlations for y[1]
    correlations_y1 = np.corrcoef(X_np.T, y_np[:, 1])[
        :-1, -1
    ]  # Correlations of each X feature with y[1]


    # Create DataFrames for easier plotting
    team_stats_names = list(teams[list(teams.keys())[0]].keys()) # Gets stat names from the first team
    num_stats = len(team_stats_names)
    print("Number of features: ", num_stats)
    feature_names = [f"{team}_" + stat_name for team in ["team1", "team2"] for stat_name in team_stats_names]


    df_corr_y0 = pd.DataFrame(
        {"feature": feature_names, "correlation": correlations_y0}
    )
    df_corr_y1 = pd.DataFrame(
        {"feature": feature_names, "correlation": correlations_y1}
    )

    # Plotting (Bar plots of correlations with y values)
    plt.figure(figsize=(24, 10))  # Adjust figure size for readability

    plt.subplot(1, 3, 1)  # Subplot for y[0]
    sns.barplot(
        x="correlation",
        y="feature",
        hue="feature",
        data=df_corr_y0.sort_values(by="correlation", ascending=False),
        palette="coolwarm",
    )  # Choose a colormap
    plt.title("Correlation with Win T1")
    plt.xlabel("Correlation Coefficient")
    plt.ylabel("Team Stats")  # Clearer label

    plt.subplot(1, 3, 2)  # Subplot for y[1]
    sns.barplot(
        x="correlation",
        y="feature",
        hue="feature",
        data=df_corr_y1.sort_values(by="correlation", ascending=False),
        palette="coolwarm",
    )  # Choose a colormap
    plt.title("Correlation with Odds T1")
    plt.xlabel("Correlation Coefficient")
    plt.ylabel("Team Stats")  # Clearer label


    # Calculate feature cross-correlation matrix
    correlation_matrix = np.corrcoef(X_np.T)  # Transpose X_np
    # print(correlation_matrix)
    for i in range(len(correlation_matrix)):
        for j in range(len(correlation_matrix[0])):
            if i == j:
                continue
            threshold = 0.95
            if correlation_matrix[i][j] > threshold or correlation_matrix[i][j] < -threshold:
                print(f"High correlation between {feature_names[i]} and {feature_names[j]}: {correlation_matrix[i][j]}")
    # Plotting (Heatmap of feature cross-correlations)
    plt.subplot(1, 3, 3)  # Add another subplot
    sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm",
                xticklabels=feature_names, yticklabels=feature_names)  # Added labels

    plt.title("Feature Cross-Correlation Matrix")
    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()

# Example Usage:
if __name__ == "__main__":
    visualize_correlations()