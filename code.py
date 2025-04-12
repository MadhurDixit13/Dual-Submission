import pandas as pd
import numpy as np
from scipy import stats

# Function to compute pooled weighted statistics for a given group.
def pooled_statistics(df_group):
    # Weighted mean using the number of students as weights.
    total_n = df_group["Num Students"].sum()
    weighted_mean = (df_group["Average Score"] * df_group["Num Students"]).sum() / total_n
    # Pooled variance: combine within-group and between-group variance
    ss_within = ((df_group["Num Students"] - 1) * (df_group["Standard Deviation"]**2)).sum()
    ss_between = (df_group["Num Students"] * (df_group["Average Score"] - weighted_mean)**2).sum()
    pooled_var = (ss_within + ss_between) / (total_n - 1)
    return weighted_mean, pooled_var, total_n

# Function to interpret the t-test result and indicate which method is better.
def interpret_result(p, mean_sing, mean_dual):
    if pd.isna(p):
        return "Insufficient data"
    elif p < 0.05:
        if mean_sing > mean_dual:
            return "Significant difference favoring Single"
        elif mean_dual > mean_sing:
            return "Significant difference favoring Dual"
        else:
            return "Significant difference but means equal"
    else:
        return "Not significant"

# Load the updated CSV data
df = pd.read_csv("data.csv")

# Ensure that numeric columns are read correctly.
df["Average Score"] = pd.to_numeric(df["Average Score"], errors='coerce')
df["Standard Deviation"] = pd.to_numeric(df["Standard Deviation"], errors='coerce')
df["Num Students"] = pd.to_numeric(df["Num Students"], errors='coerce')

# Prepare a list to hold the pooled t-test results for each QuestionGroupID.
results = []

# Process each QuestionGroupID.
for qgrp, group_data in df.groupby("QuestionGroupID"):
    # Separate rows for single and dual submissions (using lower-case for consistency)
    singles = group_data[group_data["Submission Approach"].str.lower() == "single"]
    duals   = group_data[group_data["Submission Approach"].str.lower() == "dual"]
    
    # Total number of students (summing the individual counts)
    N_singles = singles["Num Students"].sum() if not singles.empty else 0
    N_duals   = duals["Num Students"].sum() if not duals.empty else 0

    if not singles.empty and not duals.empty:
        mean_singles, var_singles, total_sing = pooled_statistics(singles)
        mean_duals,   var_duals,   total_dual = pooled_statistics(duals)
        
        SE = np.sqrt(var_singles / total_sing + var_duals / total_dual)
        t_stat = (mean_singles - mean_duals) / SE

        # Compute degrees of freedom using Welch's approximation.
        df_num = (var_singles / total_sing + var_duals / total_dual) ** 2
        df_den = ((var_singles / total_sing) ** 2 / (total_sing - 1)) + ((var_duals / total_dual) ** 2 / (total_dual - 1))
        df_welch = df_num / df_den if df_den > 0 else np.nan

        # Calculate two-tailed p-value
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df_welch)) if not np.isnan(df_welch) else np.nan
    else:
        mean_singles = np.nan
        mean_duals   = np.nan
        t_stat = np.nan
        p_value = np.nan
        total_sing = N_singles
        total_dual = N_duals

    result_text = interpret_result(p_value, mean_singles, mean_duals)

    results.append({
        "QuestionGroupID": qgrp,
        "N_Single": total_sing,
        "N_Dual": total_dual,
        "Mean_Single": mean_singles,
        "Mean_Dual": mean_duals,
        "Pooled_Var_Single": var_singles if not np.isnan(mean_singles) else np.nan,
        "Pooled_Var_Dual": var_duals if not np.isnan(mean_duals) else np.nan,
        "T_stat": t_stat,
        "Degrees_Freedom": df_welch,
        "P_value": p_value,
        "Test_Result": result_text
    })

# Convert the results list into a DataFrame and sort by QuestionGroupID.
results_df = pd.DataFrame(results)
results_df.sort_values("QuestionGroupID", inplace=True)

# Export the final results to an Excel file.
results_df.to_excel("final_t_test_results.xlsx", index=False)
print("Final t-test results with interpretation have been saved to 'final_t_test_results.xlsx'.")
