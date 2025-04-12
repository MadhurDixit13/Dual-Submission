import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("results.csv")  

# Drop rows where required columns are missing
df_clean = df.dropna(subset=["Mean_Single", "Mean_Dual", "Test_Result"]).copy()

# Convert QuestionGroupID to string safely
df_clean.loc[:, "QuestionGroupID"] = df_clean["QuestionGroupID"].astype(str)

# Assign color map based on test result
color_map = {
    "Significant difference favoring Dual": "#2ca02c",
    "Significant difference favoring Single": "#d62728",
    "Not significant": "#1f77b4",
}
df_clean.loc[:, "Color"] = df_clean["Test_Result"].map(color_map)

# Sort for consistent plotting
df_clean = df_clean.sort_values(by="Test_Result")

# Plotting
plt.figure(figsize=(10, 6))
bars = plt.barh(
    y=df_clean["QuestionGroupID"],
    width=df_clean["Mean_Dual"] - df_clean["Mean_Single"],
    color=df_clean["Color"]
)

plt.axvline(0, color='black', linewidth=1)
plt.xlabel("Mean Score Difference (Dual - Single)", fontsize=12)
plt.ylabel("Question Group ID", fontsize=12)
plt.title("Dual vs Single Submission: Where Each Excelled", fontsize=16)
plt.grid(axis='x', linestyle='--', alpha=0.5)

# Add legend manually
for label, color in color_map.items():
    plt.bar(0, 0, color=color, label=label)
plt.legend()

plt.tight_layout()
plt.show()
