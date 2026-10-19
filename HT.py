import os
import json
import pandas as pd
from scipy.stats import ttest_rel

BASE_DIR = "/Users/minahosseini/Documents/Work/Nico/CodeOcean/capsule-1115223"
DATA_DIR = os.path.join(BASE_DIR, "data")

# ------------------------------------------------------------
# LOAD ALL JSONL LOGS WITH EXTRA COLUMNS
# ------------------------------------------------------------

def load_all_data(data_dir):
    records = []
    
    # Sort participants so order is deterministic
    for participant in sorted(os.listdir(data_dir)):
        p_dir = os.path.join(data_dir, participant)
        if not os.path.isdir(p_dir):
            continue
        
        # Static first, then dynamic (as you wanted)
        for condition in ["static", "dynamic"]:
            c_dir = os.path.join(p_dir, condition)
            if not os.path.isdir(c_dir):
                continue
            
            # Sort JSONL files inside condition folder
            for file in sorted(os.listdir(c_dir)):
                if not file.endswith(".jsonl"):
                    continue
                
                f_path = os.path.join(c_dir, file)
                
                # Open JSONL file
                with open(f_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        msg = json.loads(line)
                        
                        # Append a full record with added columns
                        records.append({
                            "participant": participant,
                            "condition": condition,
                            "filename": file,      # NEW column 1
                            "participant_folder": participant,  # NEW column 2
                            "condition_folder": condition,      # NEW column 3
                            "time": msg.get("time"),
                            "message": msg.get("message")
                        })
    
    # Convert to dataframe
    df = pd.DataFrame(records)
    
    # Sort dataframe exactly as requested:
    df = df.sort_values(
        by=["participant", "condition", "filename", "time"],
        ascending=True
    ).reset_index(drop=True)
    
    return df


# Load the data
df = load_all_data(DATA_DIR)

print("Loaded dataframe shape:", df.shape)
df.head()
------------------------------------------ Workload H1 -------------------------------------------
# Filter only NASA TLX ratings messages
tlx_ratings = df[df["message"].apply(
    lambda x: isinstance(x, dict) and x.get("type") == "NASA TLX ratings"
)].copy()

print("TLX ratings shape:", tlx_ratings.shape)
tlx_ratings.head()


def parse_tlx_ratings(row):
    values = row["message"]["value"]
    return pd.Series({
        "Mental Demand": values["Mental Demand"],
        "Physical Demand": values["Physical Demand"],
        "Temporal Demand": values["Temporal Demand"],
        "Performance": values["Performance"],
        "Effort": values["Effort"],
        "Frustration": values["Frustration"],
    })

ratings_values = tlx_ratings.apply(parse_tlx_ratings, axis=1)

ratings_df = pd.concat(
    [
        tlx_ratings[["participant", "condition"]].reset_index(drop=True),
        ratings_values.reset_index(drop=True)
    ],
    axis=1
)

ratings_df.head()


# Filter only NASA TLX weights messages
tlx_weights = df[df["message"].apply(
    lambda x: isinstance(x, dict) and x.get("type") == "NASA TLX weights"
)].copy()

from collections import Counter

def compute_tlx_weights(weight_list):
    counts = Counter(w["selected"] for w in weight_list)
    total = sum(counts.values())  # should be 15
    return {k: v / total for k, v in counts.items()}

weights_df = tlx_weights.copy()
weights_df["weights_dict"] = weights_df["message"].apply(
    lambda m: compute_tlx_weights(m["weights"])
)

weights_df = weights_df[["participant", "condition", "weights_dict"]]
weights_df.head()


tlx_full = pd.merge(
    ratings_df,
    weights_df,
    on=["participant", "condition"],
    how="inner"
)

tlx_full.head()



def weighted_tlx(row):
    w = row["weights_dict"]
    return (
        row["Mental Demand"]   * w.get("Mental Demand", 0) +
        row["Physical Demand"] * w.get("Physical Demand", 0) +
        row["Temporal Demand"] * w.get("Temporal Demand", 0) +
        row["Performance"]     * w.get("Performance", 0) +
        row["Effort"]          * w.get("Effort", 0) +
        row["Frustration"]     * w.get("Frustration", 0)
    )

tlx_full["NASA_TLX_weighted"] = tlx_full.apply(weighted_tlx, axis=1)
tlx_full.head()

#--------------------------------------------
#Extract ratings into a clean table
#--------------------------------------------
def parse_tlx_ratings(row):
    values = row["message"]["value"]
    return pd.Series({
        "Mental Demand": values["Mental Demand"],
        "Physical Demand": values["Physical Demand"],
        "Temporal Demand": values["Temporal Demand"],
        "Performance": values["Performance"],
        "Effort": values["Effort"],
        "Frustration": values["Frustration"],
    })

ratings_df = tlx_ratings.copy()
ratings_values = ratings_df.apply(parse_tlx_ratings, axis=1)

ratings_df = pd.concat(
    [ratings_df[["participant", "condition"]].reset_index(drop=True),
     ratings_values.reset_index(drop=True)],
    axis=1
)
pd.set_option("display.max_colwidth", None)
ratings_df.head()
#---------------
# Filter only NASA TLX weight messages
tlx_weights = df[df['message'].apply(
    lambda x: isinstance(x, dict) and x.get("type") == "NASA TLX weights"
)].copy()

# Check
print(tlx_weights.head())
#-----------------------------------------
#Step 2: Extract and normalize TLX weights
#-----------------------------------------
from collections import Counter

def compute_tlx_weights(weight_list):
    counts = Counter(w["selected"] for w in weight_list)
    total = sum(counts.values())  # should be 15
    return {k: v / total for k, v in counts.items()}

weights_df = tlx_weights.copy()

weights_df["weights_dict"] = weights_df["message"].apply(
    lambda m: compute_tlx_weights(m["weights"])
)

weights_df = weights_df[["participant", "condition", "weights_dict"]]
pd.set_option("display.max_colwidth", None)
weights_df.head(2)

#Step 3: Merge ratings and weights correctly
tlx_full = pd.merge(
    ratings_df,
    weights_df,
    on=["participant", "condition"],
    how="inner"
)
tlx_full.head(1)

#Step 4: Compute weighted NASA-TLX workload
def weighted_tlx(row):
    w = row["weights_dict"]
    return (
        row["Mental Demand"]   * w.get("Mental Demand", 0) +
        row["Physical Demand"] * w.get("Physical Demand", 0) +
        row["Temporal Demand"] * w.get("Temporal Demand", 0) +
        row["Performance"]     * w.get("Performance", 0) +
        row["Effort"]          * w.get("Effort", 0) +
        row["Frustration"]     * w.get("Frustration", 0)
    )

tlx_full["NASA_TLX_weighted"] = tlx_full.apply(weighted_tlx, axis=1)
tlx_full.head()
#-----
#-----
#Step 5: Static vs Dynamic comparison (H1 / H2)
wide_tlx = tlx_full.pivot(
    index="participant",
    columns="condition",
    values="NASA_TLX_weighted"
).dropna()
#--------------
#---------------
from scipy.stats import ttest_rel

t, p = ttest_rel(
    wide_tlx["static"],
    wide_tlx["dynamic"]
)

print(f"Paired t-test: t = {t:.3f}, p = {p:.4f}")
#-----------------
import numpy as np

# Differences between paired observations
diffs = wide_tlx["dynamic"] - wide_tlx["static"]

# Cohen's d
cohen_d = diffs.mean() / diffs.std(ddof=1)  # ddof=1 for sample standard deviation
print(f"Cohen's d = {cohen_d:.3f}")
wide_tlx.mean()
wide_tlx.std()
------------------------------------- Mental Demand H2 ------------------------------------
# ------------------------------------------------------------
# Extract NASA-TLX Mental Demand ratings
# ------------------------------------------------------------

tlx_ratings = df[
    df["message"].apply(
        lambda m: isinstance(m, dict) and m.get("type") == "NASA TLX ratings"
    )
].copy()

def extract_mental_demand(row):
    return row["message"]["value"]["Mental Demand"]

tlx_ratings["Mental_Demand"] = tlx_ratings.apply(extract_mental_demand, axis=1)

mental_df = tlx_ratings[["participant", "condition", "Mental_Demand"]]
mental_df.head()
----------
wide_mental = mental_df.pivot(
    index="participant",
    columns="condition",
    values="Mental_Demand"
).dropna()

wide_mental
-----------
from scipy.stats import ttest_rel

t_stat, p_value = ttest_rel(
    wide_mental["static"],
    wide_mental["dynamic"]
)

print(f"Paired t-test (Mental Demand): t = {t_stat:.3f}, p = {p_value:.4f}")
-----------
import numpy as np

diffs = wide_mental["dynamic"] - wide_mental["static"]

cohen_d = diffs.mean() / diffs.std(ddof=1)

print(f"Cohen's d (paired) = {cohen_d:.3f}")
------------
wide_mental.mean(), wide_mental.std()

----------------------------------------------------subjective fluency (this is our answer H1)
import pandas as pd
from scipy.stats import ttest_rel

# ------------------------------------------------------------
# 1. Extract only "Performance and Interaction Quality results"
# ------------------------------------------------------------
# Filter rows where message is of type "Performance and Interaction Quality results"
subjective_df = df[df['message'].apply(lambda x: isinstance(x, dict) and x.get("type") == "Performance and Interaction Quality results")].copy()
# ------------------------------------------------------------
# 2. Expand the 'value' field to get individual item ratings
# ------------------------------------------------------------
records = []
for _, row in subjective_df.iterrows():
    for item in row['message']['value']:
        if item['index'] <= 8:  
            records.append({
                "participant": row['participant'],
                "condition": row['condition'],
                "index": item['index'],
                "description": item['description'],
                "selected": item['selected']
            })

ratings_df = pd.DataFrame(records)
ratings_df
# ------------------------------------------------------------
# Distribution of ratings (histogram + KDE)
# ------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(7,4))
sns.histplot(
    data=ratings_df,
    x="selected",
    hue="condition",
    bins=5,
    discrete=True,
    kde=True,
    stat="count",
    common_norm=False
)
plt.xlabel("Likert Rating")
plt.ylabel("Count")
#plt.title("Distribution of Fluency Item Ratings (Items 1–8)")
plt.tight_layout()
plt.show()
# ------------------------------------------------------------
# Boxplot of mean fluency per participant (dynamic vs. static)
# ------------------------------------------------------------
# Mean fluency per participant per condition
mean_scores = (
    ratings_df
    .groupby(["participant", "condition"])["selected"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(6,4))
sns.boxplot(
    data=mean_scores,
    x="condition",
    y="selected",
    width=0.4
)
sns.stripplot(
    data=mean_scores,
    x="condition",
    y="selected",
    color="black",
    size=6,
    jitter=True
)

plt.ylabel("Mean Fluency Score")
plt.xlabel("Condition")
#plt.title("Mean Fluency Scores per Participant (Items 1–8)")
sns.despine()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Violin plot (distribution + density)
# ------------------------------------------------------------
plt.figure(figsize=(6,4))
sns.violinplot(
    data=mean_scores,
    x="condition",
    y="selected",
    inner="quartile",
    cut=0
)
plt.ylabel("Mean Fluency Score")
plt.xlabel("Condition")
#plt.title("Distribution of Mean Fluency Scores per Condition")
sns.despine()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# List of indices to reverse-score
# ------------------------------------------------------------
reverse_indices = [2, 4, 6]

# Create a new column 'scored' with reverse scoring applied where necessary
ratings_df['scored'] = ratings_df.apply(
    lambda row: 5 - row['selected'] if row['index'] in reverse_indices else row['selected'],
    axis=1
)

# Now calculate the mean per participant per condition
mean_scores = ratings_df.groupby(['participant', 'condition'])['scored'].mean().reset_index()
mean_scores
#-------------
#Pivot and run paired t-test (dynamic vs static)
#-------------
# Pivot to wide format
pivot_scores = mean_scores.pivot(
    index='participant',
    columns='condition',
    values='scored'
).reset_index()

pivot_scores
#---------------
# Paired t-test
#---------------
t_stat, p_value = ttest_rel(
    pivot_scores['dynamic'],
    pivot_scores['static']
)

print(f"Paired t-test (Fluency, items 1–8): t = {t_stat:.3f}, p = {p_value:.4f}")

#----------------
#Effect size (Cohen’s d for paired samples)
#----------------
pivot_scores['diff'] = pivot_scores['dynamic'] - pivot_scores['static']

mean_diff = pivot_scores['diff'].mean()
std_diff = pivot_scores['diff'].std(ddof=1)

cohens_d = mean_diff / std_diff

print(f"Mean difference: {mean_diff:.3f}")
print(f"SD difference: {std_diff:.3f}")
print(f"Cohen's d (paired): {cohens_d:.3f}")

---------------------------------------------SUS - H3-------------------------------------
def load_sus_data(data_dir):
    records = []
    
    for participant in sorted(os.listdir(data_dir)):
        p_dir = os.path.join(data_dir, participant)
        if not os.path.isdir(p_dir):
            continue
        
        for condition in ["static", "dynamic"]:
            c_dir = os.path.join(p_dir, condition)
            if not os.path.isdir(c_dir):
                continue
            
            for file in sorted(os.listdir(c_dir)):
                if not file.endswith(".jsonl"):
                    continue
                
                f_path = os.path.join(c_dir, file)
                
                with open(f_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            msg = json.loads(line)  # parse JSON
                        except json.JSONDecodeError:
                            continue  # skip lines that are not valid JSON
                        
                        # Make sure message is a dictionary
                        if isinstance(msg, dict):
                            message = msg.get("message", {})
                            if isinstance(message, dict) and message.get("type") == "SUS results":
                                records.append({
                                    "participant": participant,
                                    "condition": condition,
                                    "time": msg.get("time"),
                                    "sus_items": message.get("value", [])
                                })
    
    df = pd.DataFrame(records)
    return df

# Reload SUS data
sus_df = load_sus_data(DATA_DIR)
print("Loaded SUS data shape:", sus_df.shape)
sus_df.head()
# ------------------------------------------------------------
# FUNCTION TO COMPUTE SUS SCORE
# ------------------------------------------------------------
def compute_sus_score(items):
    score = 0
    for item in items:
        idx = item["index"]
        val = item["selected"]
        if idx % 2 == 1:  # odd items (1,3,5,7,9)
            score += (val - 1)
        else:             # even items (2,4,6,8,10)
            score += (5 - val)
    return score * 2.5


sus_df["SUS_score"] = sus_df["sus_items"].apply(compute_sus_score)
sus_wide = sus_df.pivot(index="participant", columns="condition", values="SUS_score")
sus_wide
print(sus_wide.head())
---------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Convert to long format for easier plotting
sus_long = sus_df.melt(
    id_vars=["participant"],
    value_vars=["SUS_score"],
    var_name="Measure",
    value_name="Score"
)

# Separate data for static and dynamic
sus_static = sus_wide['static'].dropna()
sus_dynamic = sus_wide['dynamic'].dropna()

# ------------------------------------------------------------
# Histogram + KDE
# ------------------------------------------------------------
plt.figure(figsize=(8,4))
sns.histplot(sus_static, color="skyblue", kde=True, label="Static", bins=10)
sns.histplot(sus_dynamic, color="salmon", kde=True, label="Dynamic", bins=10, alpha=0.7)
plt.xlabel("SUS Score")
plt.ylabel("Frequency")
#plt.title("Distribution of SUS Scores per Condition")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Boxplot
# ------------------------------------------------------------
# Reset the index for plotting
sus_plot_df = sus_wide.reset_index().melt(id_vars="participant", value_vars=["static", "dynamic"],
                                          var_name="Condition", value_name="SUS_score")
#------------
plt.figure(figsize=(6, 4))
sns.boxplot(x="Condition", y="SUS_score", data=sus_plot_df, palette=["skyblue", "salmon"])
sns.stripplot(x="Condition", y="SUS_score", data=sus_plot_df, color="darkblue", size=6, jitter=True)
#plt.title("SUS Score Distribution per Condition")
plt.ylabel("SUS Score")
plt.xlabel("Condition")
plt.tight_layout()
plt.show()

# ---------------------------------------
# SUS and Fluency Analysis (H3)
# ---------------------------------------
# Note: pandas and ttest_rel already imported at top of file

# ----------------------------
# 1. SUS paired t-test
# ----------------------------
# sus_wide already has pivoted SUS scores: participants x {dynamic, static}
t_sus, p_sus = ttest_rel(sus_wide["dynamic"], sus_wide["static"])

# Compute mean difference and Cohen's d for effect size
sus_diff = sus_wide["dynamic"] - sus_wide["static"]
sus_mean_diff = sus_diff.mean()
sus_sd_diff = sus_diff.std(ddof=1)
cohen_d_sus = sus_mean_diff / sus_sd_diff

print(f"SUS paired t-test: t = {t_sus:.3f}, p = {p_sus:.4f}")
print(f"Mean difference: {sus_mean_diff:.3f}, SD difference: {sus_sd_diff:.3f}")
print(f"Cohen's d (paired): {cohen_d_sus:.3f}")

# ----------------------------
# 2. Fluency paired t-test
# ----------------------------
# pivot_scores already has pivoted fluency scores: participants x {dynamic, static}
t_fluency, p_fluency = ttest_rel(pivot_scores["dynamic"], pivot_scores["static"])

# Compute mean difference and Cohen's d for effect size
fluency_diff = pivot_scores["dynamic"] - pivot_scores["static"]
fluency_mean_diff = fluency_diff.mean()
fluency_sd_diff = fluency_diff.std(ddof=1)
cohen_d_fluency = fluency_mean_diff / fluency_sd_diff

print(f"Fluency paired t-test: t = {t_fluency:.3f}, p = {p_fluency:.4f}")
print(f"Mean difference: {fluency_mean_diff:.3f}, SD difference: {fluency_sd_diff:.3f}")
print(f"Cohen's d (paired): {cohen_d_fluency:.3f}")

# ----------------------------
# 3. Hypothesis testing interpretation
# ----------------------------
def interpret_h3(t_sus, p_sus, t_fluency, p_fluency, alpha=0.05):
    print("\nH3: Participants prefer the dynamic version.")
    print("Measured via SUS and fluency questionnaires.\n")
    
    # SUS interpretation
    print("SUS:")
    if p_sus < alpha and sus_mean_diff > 0:
        print(f"Dynamic SUS scores are significantly higher than static (t={t_sus:.3f}, p={p_sus:.4f}, d={cohen_d_sus:.3f}) → supports H3.")
    else:
        print(f"No significant difference in SUS between dynamic and static (t={t_sus:.3f}, p={p_sus:.4f}, d={cohen_d_sus:.3f}).")
    
    # Fluency interpretation
    print("\nFluency:")
    if p_fluency < alpha and fluency_mean_diff > 0:
        print(f"Dynamic fluency scores are significantly higher than static (t={t_fluency:.3f}, p={p_fluency:.4f}, d={cohen_d_fluency:.3f}) → supports H3.")
    else:
        print(f"No significant difference in fluency between dynamic and static (t={t_fluency:.3f}, p={p_fluency:.4f}, d={cohen_d_fluency:.3f}).")
    
    # Overall H3
    if (p_sus < alpha and sus_mean_diff > 0) and (p_fluency < alpha and fluency_mean_diff > 0):
        print("\nOverall, H3 is supported: participants prefer the dynamic version.")
    else:
        print("\nOverall, H3 is partially or not supported.")

# Run interpretation
interpret_h3(t_sus, p_sus, t_fluency, p_fluency)
# ------------------------------------------ H4 ---------------------------------------------
# Note: json, os, pandas already imported at top of file
from scipy.stats import ttest_1samp

def load_completion_times(data_dir):
    records = []

    for participant in os.listdir(data_dir):
        p_dir = os.path.join(data_dir, participant)
        if not os.path.isdir(p_dir):
            continue

        for condition in ["static", "dynamic"]:
            c_dir = os.path.join(p_dir, condition)
            if not os.path.isdir(c_dir):
                continue

            for file in os.listdir(c_dir):
                if not file.endswith(".jsonl"):
                    continue

                with open(os.path.join(c_dir, file), "r") as f:
                    for line in f:
                        try:
                            msg = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        message = msg.get("message", {})
                        if isinstance(message, dict) and "Completion" in message:
                            comp = message["Completion"]
                            records.append({
                                "participant": participant,
                                "condition": condition,
                                "actor": comp.get("Actor"),
                                "duration_ms": comp.get("Duration")
                            })

    df = pd.DataFrame(records)
    df["duration_s"] = df["duration_ms"] / 1000.0
    return df

completion_df = load_completion_times(DATA_DIR)
completion_df.head()
# --------
human_steps = completion_df[completion_df["actor"] == "Human"]["duration_s"]
human_steps
# --------
PILOT_MEAN = 5.3  # seconds
t_h4, p_h4 = ttest_1samp(human_steps, PILOT_MEAN)

mean_exp = human_steps.mean()
std_exp = human_steps.std()
n_steps = len(human_steps)

print(f"H4 one-sample t-test:")
print(f"t = {t_h4:.3f}, p = {p_h4:.4f}")
print(f"Experimental steps: M = {mean_exp:.2f}s, SD = {std_exp:.2f}s, n = {n_steps}")











import json
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

def extract_completion_times(data_dir):
    durations = []

    for participant in os.listdir(data_dir):
        p_dir = os.path.join(data_dir, participant)
        if not os.path.isdir(p_dir):
            continue

        for condition in ["static", "dynamic"]:
            c_dir = os.path.join(p_dir, condition)
            if not os.path.isdir(c_dir):
                continue

            for file in os.listdir(c_dir):
                if not file.endswith(".jsonl"):
                    continue

                with open(os.path.join(c_dir, file), "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            msg = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        message = msg.get("message", {})
                        if isinstance(message, dict) and "Completion" in message:
                            durations.append({
                                "participant": participant,
                                "condition": condition,
                                "actor": message["Completion"]["Actor"],
                                "duration_ms": message["Completion"]["Duration"]
                            })

    return pd.DataFrame(durations)
-------
completion_df = extract_completion_times(DATA_DIR)

# Convert milliseconds to seconds
completion_df["duration_s"] = completion_df["duration_ms"] / 1000.0

# Use human assembly steps only (consistent with pilot study)
human_steps = completion_df[completion_df["actor"] == "Human"]["duration_s"]
-------
PILOT_MEAN = 5.3  # seconds

t_stat, p_value = ttest_1samp(human_steps, PILOT_MEAN)

mean_time = human_steps.mean()
std_time = human_steps.std(ddof=1)

print(f"H4 – One-sample t-test vs pilot study")
print(f"Mean = {mean_time:.2f} s, SD = {std_time:.2f} s")
print(f"t = {t_stat:.3f}, p = {p_value:.4f}")
--------
alpha = 0.05

if p_value >= alpha:
    print("No significant difference from pilot study → H4 supported.")
else:
    print("Significant difference from pilot study → H4 not supported.")
---------------






------------------------------------------ H4 (the answer) ------------------------------------------
import json
import os
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon
def extract_step_times(data_dir):
    records = []

    for participant in os.listdir(data_dir):
        p_dir = os.path.join(data_dir, participant)
        if not os.path.isdir(p_dir):
            continue

        for condition in ["static", "dynamic"]:
            c_dir = os.path.join(p_dir, condition)
            if not os.path.isdir(c_dir):
                continue

            for file in os.listdir(c_dir):
                if not file.endswith(".jsonl"):
                    continue

                with open(os.path.join(c_dir, file), "r") as f:
                    for line in f:
                        try:
                            msg = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        message = msg.get("message", {})
                        if isinstance(message, dict) and "Completion" in message:
                            comp = message["Completion"]
                            records.append({
                                "participant": participant,
                                "condition": condition,
                                "actor": comp.get("Actor"),
                                "duration_ms": comp.get("Duration")
                            })

    df = pd.DataFrame(records)
    df["duration_s"] = df["duration_ms"] / 1000.0
    return df

step_df = extract_step_times(DATA_DIR)
step_df
--------
human_steps = step_df[step_df["actor"] == "Human"]

# Mean step time per participant
participant_means = (
    human_steps
    .groupby("participant")["duration_s"]
    .mean()
)
---------
t_stat, p_value = ttest_1samp(
    participant_means,
    PILOT_MEAN
)

print(f"H4 one-sample t-test: t = {t_stat:.3f}, p = {p_value:.4f}")
--------
import numpy as np
from scipy.stats import wilcoxon

# ------------------------------------------------------------
# Pilot study reference value (median per assembly step)
# ------------------------------------------------------------
PILOT_MEDIAN = 5.3  # seconds (from pilot study)

# ------------------------------------------------------------
# participant_means:
# array of mean completion time per assembly step per participant
# Example:
# participant_means = np.array([...])
# ------------------------------------------------------------

participant_means = np.array(participant_means)

# ------------------------------------------------------------
# One-sample Wilcoxon signed-rank test
# ------------------------------------------------------------
w_stat, p_wilcoxon = wilcoxon(participant_means - PILOT_MEDIAN)

print(f"H4 Wilcoxon test: W = {w_stat:.3f}, p = {p_wilcoxon:.4f}")

# ------------------------------------------------------------
# Interpretation
# ------------------------------------------------------------
alpha = 0.05
if p_wilcoxon < alpha:
    print("Reject H0: Completion times differ significantly from the pilot study.")
else:
    print("Fail to reject H0: Completion times are comparable to the pilot study.")
-----------
import numpy as np
from scipy.stats import rankdata

# Differences from pilot median
diffs = participant_means - PILOT_MEDIAN

# Remove zeros
diffs_nonzero = diffs[diffs != 0]

# Absolute ranks
ranks = rankdata(np.abs(diffs_nonzero))

# Sum of ranks for positive and negative differences
W_pos = ranks[diffs_nonzero > 0].sum()
W_neg = ranks[diffs_nonzero < 0].sum()

# Rank-biserial correlation
r_rb = (W_pos - W_neg) / ranks.sum()

print(f"H4 rank-biserial correlation: r_rb = {r_rb:.3f}")
------------
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Example: participant_means is a pandas Series or array of mean step times
participant_means = np.array(participant_means)  # from your data
pilot_median = PILOT_MEDIAN

# Create a dataframe suitable for plotting
plot_df = pd.DataFrame({
    "Participant": [f"P{i+1}" for i in range(len(participant_means))],
    "StepTime": participant_means
})

# ------------------------------------------------------------
# Boxplot with pilot median overlay
# ------------------------------------------------------------
plt.figure(figsize=(6,4))
sns.boxplot(y="StepTime", data=plot_df, color="skyblue", width=0.4)
sns.stripplot(y="StepTime", data=plot_df, color="darkblue", size=6, jitter=True)
plt.axhline(pilot_median, color="red", linestyle="--", label=f"Pilot median = {pilot_median}s")
plt.ylabel("Mean Completion Time per Step (s)")
plt.title("Participant Step Times vs Pilot Study")
plt.legend()
sns.despine(trim=True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Violin plot with pilot median overlay
# ------------------------------------------------------------
plt.figure(figsize=(6,4))
sns.violinplot(y="StepTime", data=plot_df, inner="quartile", color="lightgreen")
sns.stripplot(y="StepTime", data=plot_df, color="darkgreen", size=6, jitter=True)
plt.axhline(pilot_median, color="red", linestyle="--", label=f"Pilot median = {pilot_median}s")
plt.ylabel("Mean Completion Time per Step (s)")
plt.title("Participant Step Times vs Pilot Study (Violin Plot)")
plt.legend()
sns.despine(trim=True)
plt.tight_layout()
plt.show()









# ------------------------------------------------------------
# 3. Compute mean subjective fluency per participant per condition
# ------------------------------------------------------------
mean_fluency = ratings_df.groupby(['participant', 'condition'])['selected'].mean().reset_index()
print(mean_fluency)

# ------------------------------------------------------------
# 4. Pivot table to have columns for static and dynamic
# ------------------------------------------------------------
fluency_pivot = mean_fluency.pivot(index='participant', columns='condition', values='selected').reset_index()
print(fluency_pivot)

# ------------------------------------------------------------
# 5. Hypothesis testing: paired t-test
# H1: Dynamic > Static (one-tailed)
# ------------------------------------------------------------
# scipy ttest_rel is two-tailed, so we can convert manually
t_stat, p_two_tailed = ttest_rel(fluency_pivot['dynamic'], fluency_pivot['static'])

# Convert to one-tailed p-value for dynamic > static
if t_stat > 0:
    p_one_tailed = p_two_tailed / 2
else:
    p_one_tailed = 1 - (p_two_tailed / 2)

print(f"Paired t-test: t = {t_stat:.3f}, one-tailed p = {p_one_tailed:.4f}")

# Interpretation
if p_one_tailed < 0.05:
    print("H1 supported: Dynamic scheduling increases subjective fluency.")
else:
    print("H1 not supported.")
# ------------------------------------------------------------
# 6. Compute paired Cohen's d (effect size)
# ------------------------------------------------------------
diff = fluency_pivot['dynamic'] - fluency_pivot['static']
mean_diff = np.mean(diff)
std_diff = np.std(diff, ddof=1)  # use ddof=1 for sample standard deviation

cohens_d = mean_diff / std_diff
print(f"Paired Cohen's d = {cohens_d:.3f}")

# Interpretation guideline:
# small ~ 0.2, medium ~ 0.5, large ~ 0.8
if cohens_d < 0.2:
    size_desc = "small"
elif cohens_d < 0.5:
    size_desc = "small-to-medium"
elif cohens_d < 0.8:
    size_desc = "medium-to-large"
else:
    size_desc = "large"

print(f"Effect size is {size_desc}.")


















# ------------------------------------------------------------
# STEP 1 — Extract NASA-TLX ratings
# ------------------------------------------------------------
def extract_nasa_tlx(df):
    rows = []
    for _, row in df.iterrows():
        msg = row["message"]
        if isinstance(msg, dict) and msg.get("type") == "NASA TLX ratings":
            rows.append({
                "participant": row["participant_folder"],
                "condition": row["condition_folder"],
                **msg["value"]
            })
    return pd.DataFrame(rows)

tlx = extract_nasa_tlx(df)
tlx.head(100)
--------------------------------
#Compute overall workload score (mean TLX)
--------------------------------
tlx["workload"] = tlx[
    ["Mental Demand", "Physical Demand", "Temporal Demand",
     "Performance", "Effort", "Frustration"]
].mean(axis=1)
tlx.head(10)
--------------------------------
#Aggregate per participant × condition
--------------------------------
workload_summary = (
    tlx.groupby(["participant", "condition"])["workload"]
    .mean()
    .reset_index()
)
workload_summary
--------------------------------
#3.4 Convert to paired (wide) format
--------------------------------
wide_workload = workload_summary.pivot(
    index="participant",
    columns="condition",
    values="workload"
)
wide_workload
--------------------------------
#3.5 Paired t-test for workload (H1c)
--------------------------------
wide_workload = workload_summary.pivot(
    index="participant",
    columns="condition",
    values="workload"
).dropna()

wide_workload.columns = ["static", "dynamic"]
wide_workload
--------------------------------
#3.5 Statistical test (paired t-test)
--------------------------------
t_w, p_w = ttest_rel(
    wide_workload["static"],
    wide_workload["dynamic"],
    alternative="greater"   # static > dynamic
)

print(f"Workload: t = {t_w:.3f}, p = {p_w:.4f}")
--------------------------------
from scipy.stats import ttest_rel

t, p = ttest_rel(workload_static, workload_dynamic)
print(t, p)
---------------------------------
--------------------------------
--------------------------------
--------------------------------
from scipy.stats import ttest_rel

t1, p1 = ttest_rel(
    wide_workload["static"],
    wide_workload["dynamic"],
    alternative="two-sided"
)

print("=== H1: Perceived Workload ===")
print(f"t = {t1:.3f}")
print(f"p = {p1:.4f}")
---------------------------------   
#Direction of the effect (very important) ###If Static > Dynamic → dynamic reduces workload (supports H1) If Dynamic > Static → opposite effect
---------------------------------
mean_static = wide_workload["static"].mean()
mean_dynamic = wide_workload["dynamic"].mean()

print(f"Mean workload (Static): {mean_static:.2f}")
print(f"Mean workload (Dynamic): {mean_dynamic:.2f}")
print(f"Difference (Static − Dynamic): {mean_static - mean_dynamic:.2f}")
---------------------------------
#Effect size (Cohen’s d for paired samples) ###This tells you how large the effect is (not just whether it exists).
####Interpretation (standard): |d| ≈ 0.2 → small |d| ≈ 0.5 → medium |d| ≥ 0.8 → large Sign tells direction: Positive d → static workload higher Negative d → dynamic workload higher
---------------------------------
def cohens_d_paired(x, y):
    diff = x - y
    return diff.mean() / diff.std(ddof=1)

d1 = cohens_d_paired(
    wide_workload["workload_static"],
    wide_workload["workload_dynamic"]
)

print(f"Cohen’s d (Workload) = {d1:.2f}")







--------------------------
#Paired t-test for workload (H1c)
t1, p1 = ttest_rel(
    wide_workload["static"],
    wide_workload["dynamic"],
    alternative="greater"
)
print(f"t = {t1:.3f}, p = {p1:.4f}")
###NOTE: p-value (key point) The p-value is much larger than 0.05: 𝑝 = 0.9704 ≫ 0.05 p=0.9704≫0.05 This means the observed difference in workload between static and dynamic scheduling is not statistically significant. Hypothesis context (H1 – workload part) H1 states that: Dynamic scheduling reduces perceived workload compared to static scheduling. Since the p-value is very high, you fail to reject the null hypothesis. In other words: There is no statistical evidence that dynamic scheduling reduces perceived workload compared to static scheduling. Any observed difference is very likely due to random variation in the sample. t-statistic sign (why it still matters) The negative t-value ( 𝑡 = − 1.996 t=−1.996) indicates that, on average, workload in the dynamic condition was lower than in the static condition in the direction expected by H1. However, because the result is not significant, this directional trend cannot be interpreted as a reliable effect. Why p is so large despite |t| ≈ 2 A p-value close to 1 usually indicates one of the following: The test direction may be mismatched (e.g., one-sided vs two-sided confusion), Or the test was run with an alternative hypothesis opposite to the observed direction, Or the reported p-value corresponds to a different test than the t-statistic shown. (This is something you could briefly acknowledge in an interview as a sanity-check step you performed.) How to state this clearly (paper / interview ready) The paired t-test showed no significant difference in perceived workload between static and dynamic scheduling (t = −1.996, p = 0.97). Although workload tended to be lower in the dynamic condition, this effect was not statistically significant. Therefore, H1 is not supported with respect to perceived workload.
###NOTE:Workload: t = -1.996, p = 0.9704


# ------------------------------------------------------------
# STEP 2 — Extract Key Metrics
# ------------------------------------------------------------

# ----- Human operation durations
def extract_human_ops(df):
    ops = []
    for _, row in df.iterrows():
        msg = row["message"]
        if isinstance(msg, dict) and "Completion" in msg:
            c = msg["Completion"]
            if c.get("Actor") == "Human":
                ops.append({
                    "participant": row["participant_folder"],
                    "condition": row["condition_folder"],
                    "duration": c["Duration"]
                })
    return pd.DataFrame(ops)

human_ops = extract_human_ops(df)
human_ops.head(100)


# ----- Interrupt durations
def extract_interrupts(df):
    inters = []
    for _, row in df.iterrows():
        msg = row["message"]
        if isinstance(msg, dict) and "Interrupt Ended" in msg:
            dur = msg["Interrupt Ended"]["Duration"]
            inters.append({
                "participant": row["participant_folder"],
                "condition": row["condition_folder"],
                "interrupt_duration": dur
            })
    return pd.DataFrame(inters)

interrupts = extract_interrupts(df)
interrupts.iloc[0:10]

# ------------------------------------------------------------
# STEP 3 — Aggregate per participant × condition
# ------------------------------------------------------------

mean_human = (
    human_ops
    .groupby(["participant", "condition"])["duration"]
    .mean()
    .reset_index()
    .rename(columns={"duration": "mean_human_duration"})
)

total_interrupt = (
    interrupts
    .groupby(["participant", "condition"])["interrupt_duration"]
    .sum()
    .reset_index()
    .rename(columns={"interrupt_duration": "total_interrupt_duration"})
)

summary = pd.merge(mean_human, total_interrupt, on=["participant","condition"], how="outer").fillna(0)
summary

# ------------------------------------------------------------
# STEP 4 — Pivot into Static vs Dynamic
# ------------------------------------------------------------

wide = summary.pivot(index="participant", columns="condition", values=["mean_human_duration", "total_interrupt_duration"])
wide.columns = ['_'.join(col).strip() for col in wide.columns.values]
wide = wide.dropna()
print("\nWide paired dataset:")
print(wide)



---------------
from scipy.stats import ttest_rel

# Example: Paired t-test for human operation duration (H1a)
t1, p1 = ttest_rel(wide["mean_human_duration_static"], wide["mean_human_duration_dynamic"])

# Paired t-test for total interrupt duration (H1b)
t2, p2 = ttest_rel(wide["total_interrupt_duration_static"], wide["total_interrupt_duration_dynamic"])


# ------------------------------------------------------------
# STEP 5 — Paired T-tests (H1 analysis)
# ------------------------------------------------------------

# Workload proxy: human operation durations
t1, p1 = ttest_rel(
    wide["mean_human_duration_static"],
    wide["mean_human_duration_dynamic"],
    alternative="greater"   # static > dynamic (dynamic is lower workload)
)

# Fluency proxy: interruptions
t2, p2 = ttest_rel(
    wide["total_interrupt_duration_static"],
    wide["total_interrupt_duration_dynamic"],
    alternative="greater"   # static > dynamic (dynamic is more fluent)
)

print("\n=== H1 Statistical Tests ===")

print("\nH1a: Human Operation Duration (Workload)")
print(f"t = {t1:.3f}, p = {p1:.4f}")
if p1 < 0.05:
    print("→ Significant: Dynamic reduces workload.")
else:
    print("→ Not significant: No reduction detected.")

print("\nH1b: Interrupt Duration (Fluency)")
print(f"t = {t2:.3f}, p = {p2:.4f}")
if p2 < 0.05:
    print("→ Significant: Dynamic increases fluency.")
else:
    print("→ Not significant: No fluency improvement detected.")


# ------------------------------------------------------------
# STEP 6 — Effect Sizes (Cohen’s d)
# ------------------------------------------------------------

def cohens_d(x, y):
    diff = x - y
    return diff.mean() / diff.std(ddof=1)

d1 = cohens_d(wide["mean_human_duration_static"], wide["mean_human_duration_dynamic"])
d2 = cohens_d(wide["total_interrupt_duration_static"], wide["total_interrupt_duration_dynamic"])

print("\nEffect Sizes:")
print(f"Cohen’s d (Workload) = {d1:.2f}")
print(f"Cohen’s d (Fluency)  = {d2:.2f}")

--------------------------------
from scipy.stats import shapiro
import matplotlib.pyplot as plt

# Differences
diff_human = wide["mean_human_duration_static"] - wide["mean_human_duration_dynamic"]
diff_interrupt = wide["total_interrupt_duration_static"] - wide["total_interrupt_duration_dynamic"]

# Shapiro-Wilk tests
shapiro_human = shapiro(diff_human)
shapiro_interrupt = shapiro(diff_interrupt)

print("Shapiro-Wilk Test (Workload differences):", shapiro_human)
print("Shapiro-Wilk Test (Fluency differences):", shapiro_interrupt)

# Optional: visualize
plt.hist(diff_human, bins=8, alpha=0.7, label="Workload differences")
plt.hist(diff_interrupt, bins=8, alpha=0.7, label="Fluency differences")
plt.legend()
plt.show()

-------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Use seaborn style for nicer aesthetics
sns.set(style="whitegrid")

# -------- Workload Differences --------
plt.figure(figsize=(8, 6))
sns.histplot(diff_human, bins=8, kde=True, color='skyblue')
#plt.title("Workload Differences (Static - Dynamic)", fontsize=14)
plt.xlabel("Difference in Human Operation Duration (Seconds)", fontsize=12)
plt.ylabel("Number of Participants", fontsize=12)
plt.axvline(0, color='red', linestyle='--', label='No Difference')
plt.legend()
plt.tight_layout()
plt.show()

# -------- Fluency Differences --------
plt.figure(figsize=(8, 6))
sns.histplot(diff_interrupt, bins=8, kde=True, color='lightgreen')
#plt.title("Fluency Differences (Static - Dynamic)", fontsize=14)
plt.xlabel("Difference in Interrupt Duration (Seconds)", fontsize=12)
plt.ylabel("Number of Participants", fontsize=12)
plt.axvline(0, color='red', linestyle='--', label='No Difference')
plt.legend()
plt.tight_layout()
plt.show()
------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------
# ----- Mental Demand
def extract_mental_demand(df):
    md = []
    for _, row in df.iterrows():
        msg = row["message"]
        if isinstance(msg, dict) and "Mental Demand" in msg:
            md.append({
                "participant": row["participant_folder"],
                "condition": row["condition_folder"],
                "mental_demand": msg["Mental Demand"]
            })
    return pd.DataFrame(md)

mental_demand = extract_mental_demand(df)
mental_demand.head()
--------------------------------
--------------------------------
--------------------------------
Extract NASA TLX Mental Demand
--------------------------------
# ----- NASA TLX Mental Demand extraction
def extract_mental_demand(df):
    md = []
    for _, row in df.iterrows():
        msg = row["message"]
        if isinstance(msg, dict) and msg.get("type") == "NASA TLX ratings":
            # Retrieve the Mental Demand rating
            mental_score = msg["value"].get("Mental Demand", None)

            if mental_score is not None:
                md.append({
                    "participant": row["participant_folder"],
                    "condition": row["condition_folder"],
                    "mental_demand": mental_score
                })
    return pd.DataFrame(md)


mental = extract_mental_demand(df)
print(mental)
--------------------------------
Aggregate Mental Demand per participant × condition
--------------------------------
mean_mental = (
    mental
    .groupby(["participant", "condition"])["mental_demand"]
    .mean()
    .reset_index()
    .rename(columns={"mental_demand": "mean_mental_demand"})
)

summary = pd.merge(summary, mean_mental, on=["participant", "condition"], how="outer")
summary = summary.fillna(0)
summary
# ---------------------------------
# Static vs Dynamic
# ---------------------------------
# Add mental demand to wide format
wide_mental = summary.pivot(
    index="participant",
    columns="condition",
    values="mean_mental_demand"
).dropna()

wide_mental.columns = ["mental_static", "mental_dynamic"]
print("\nWide Mental-Demand Dataset:")
print(wide_mental)
# ---------------------------------
# Paired t-test for Mental Demand (H2)
# ---------------------------------
from scipy.stats import ttest_rel

t3, p3 = ttest_rel(
    wide_mental["mental_static"],
    wide_mental["mental_dynamic"],
    alternative="two-sided"
)

print("\n=== H2 Statistical Test: Mental Demand ===")
print(f"t = {t3:.3f}, p = {p3:.4f}")

if p3 < 0.05:
    print("→ Significant: Mental demand differs between conditions.")
else:
    print("→ Not significant: Dynamic does NOT increase mental demand.")

# -------------------Cohen's d for Mental Demand-------------
# ------------------------------------------------------------
# Cohen's d for paired samples
# ------------------------------------------------------------
def cohens_d(x, y):
    diff = x - y
    return diff.mean() / diff.std(ddof=1)

d3 = cohens_d(
    wide_mental["mental_static"],
    wide_mental["mental_dynamic"]
)

print(f"Cohen's d (Mental Demand) = {d3:.2f}")
# ----------------Normality checks (Shapiro–Wilk)----------------
from scipy.stats import shapiro
diff_mental = wide_mental["mental_static"] - wide_mental["mental_dynamic"]

shapiro_mental = shapiro(diff_mental)

print("\nShapiro-Wilk Test (Mental Demand differences):", shapiro_mental)
# --------------------------------
# Histogram and Q-Q
# --------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot

# Differences: static - dynamic
diff_mental = wide_mental["mental_static"] - wide_mental["mental_dynamic"]

# Set seaborn style for nicer plots
sns.set(style="whitegrid")

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Histogram with KDE
sns.histplot(diff_mental, bins=8, kde=True, color='lightcoral', ax=axes[0])
axes[0].set_title("Histogram of Mental Demand Differences", fontsize=14)
axes[0].set_xlabel("Difference (Static - Dynamic)", fontsize=12)
axes[0].set_ylabel("Number of Participants", fontsize=12)
axes[0].axvline(0, color='red', linestyle='--', label='No Difference')
axes[0].legend()

# Q-Q plot
probplot(diff_mental, dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot of Mental Demand Differences", fontsize=14)

plt.suptitle("Mental Demand Differences: Static vs Dynamic", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

------------------------Plot Mental Demand Differences----------------------------------
plt.hist(diff_mental, bins=8, alpha=0.7, color="violet")
plt.axvline(0, color="black", linestyle="--")
plt.title("Mental Demand Differences (Static - Dynamic)")
plt.xlabel("Difference Score")
plt.ylabel("Participants")
plt.show()









