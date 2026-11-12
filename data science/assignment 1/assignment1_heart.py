import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap




# ================ IDA Step1: Big Picture ================

# Topic: Heart Disease Prediction 
# each row is a patient, each column is a feature (attribute) of the patient, 
# and the last column is the target variable (num) indicating the presence or absence
# of heart disease.

# Question: Can we predict the presence of heart disease based on these 13 features 
# of the patients?




# ================ IDA Step2: Machine-readable ================

# !!!!!!!!!!!!!!!!! ISSUES: 
# File is CSV but in .data format => we use .read_csv() to read it
# no header row => header=None
# columns have no names => we provide names=COLUMNS
# missing values are represented by "?" => we use na_values="?" that converts "?" to NaN (let them be number inputs)


# Load the dataset #
DATA_PATH = "processed.cleveland.data"

COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

# Load the dataset into a pandas DataFrame #
df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS, na_values="?")
# print(df.head())




# ================ IDA Step3: Size and Structure ================

# print(df.shape)

# Number of patients and columns #
# print("shape:", df.shape)

# !!!!!! missing values: ca 299 thal 301 - others 303 !!!!!!
# Data type and non-null count per column #
# df.info()

# For each col : mean, max, min, std and quartiles #
print(df.describe().T.round(2))

# Few unique values -> categorical # 
# print(df.nunique())

# Class distribution of target #
# print(df["num"].value_counts().sort_index())




# ================ IDA Step4: Missing Values ================

# Number of missing values per column #
# print(df.isna().sum())

# Show the incomplete patients #
print("rows with any missing value:", df.isna().any(axis=1).sum())
print(df[df.isna().any(axis=1)])



# Missingness heatmap: black = missing, white = not missing #
# fig, ax = plt.subplots(figsize=(8, 3))
# sns.heatmap(df.isna().T, cbar=False, cmap="Greys", ax=ax)
# ax.set(title="Missing values (black) per patient", xlabel="patient index", xticks=[])
# plt.show()


# OR a colored heatmap: #
missing = df.isna().sum()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1, 3]})

# left: number of missing values per column
missing.plot.barh(ax=ax1, color=["firebrick" if m > 0 else "lightgrey" for m in missing])
ax1.invert_yaxis()
ax1.set(title="Missing values per column", xlabel="count", xlim=(0, missing.max() + 1))
for i, m in enumerate(missing):
    if m > 0:
        ax1.text(m + 0.1, i, str(m), va="center")

# right: position of the missing values (red = missing)
sns.heatmap(df.isna().T, cmap=ListedColormap(["#f0f0f0", "firebrick"]), cbar=False, ax=ax2)
ax2.set(title="Position of missing values (red) per patient", xlabel="patient index")
ax2.set_xticks(range(0, len(df), 50))
ax2.set_xticklabels(range(0, len(df), 50), rotation=0)

plt.suptitle(f"IDA Step 4 – Missing values ({missing.sum()} in "
             f"{df.isna().any(axis=1).sum()} of {len(df)} patients)")
plt.tight_layout()
plt.show()




# ================ IDA Step5: Overview ================

# Lecture tip: log-transform skewed data before plotting.
# Not needed here: skewness is moderate (max ~1.3, chol / oldpeak) and min-max
# scaling keeps all columns comparable; only one extreme chol value (564).

features = [c for c in df.columns if c != "num"]
ordered = df.sort_values("num")                    # Sort patients by diagnosis #

# Optional test: log-transform the two skewed columns (log1p because oldpeak has zeros)
# ordered[["chol", "oldpeak"]] = np.log1p(ordered[["chol", "oldpeak"]])

scaled = (ordered[features] - ordered[features].min()) / \
         (ordered[features].max() - ordered[features].min())   # min-max scaling to [0, 1]

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(scaled.values, cmap="viridis", xticklabels=features, yticklabels=False, ax=ax)
ax.set(title="Overview of all patients (rows sorted by num, 0 at top)", ylabel="patients")
plt.suptitle("IDA Step 5 – Overview")
plt.tight_layout()
plt.show()




# ================ IDA Step6: Univariate Examination ================

# 6a Categorical features + targets: counts plots [in-class: sns.countplot]
cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "num"]
fig, axes = plt.subplots(3, 3, figsize=(12, 9))
for ax, col in zip(axes.ravel(), cat_cols):
    sns.countplot(x=col, data=df, ax=ax, color="steelblue")
    ax.set_title(col)
plt.suptitle("IDA Step 6a – Counts of categorical features and of the target")
plt.tight_layout()
plt.show()

# 6b) continuous features: histograms [in-class: DataFrame.hist(bins=30)]
num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
df[num_cols].hist(bins=30, figsize=(12, 6), layout=(2, 3))
fig = plt.gcf()
fig.suptitle("IDA Step 6b – Histograms of the continuous features", y=1.00)
plt.tight_layout()
plt.show()

# 6c) box plots + skewness to detect outliers / asymmetry [added]
print("skewness:\n", df[num_cols].skew().round(2))
fig, axes = plt.subplots(1, 5, figsize=(16, 3))
for ax, col in zip(axes, num_cols):
    sns.boxplot(x=df[col], ax=ax, color="lightsteelblue")
plt.suptitle("IDA Step 6c – Box plots of continuous features")
plt.tight_layout()
plt.show()

# Observations:
# - about twice as many men as women (206 vs. 97); most patients have cp = 4 (asymptomatic)
# - fbs = 1 is rare (15 %); restecg = 1 occurs only 4 times (almost empty category)
# - target is imbalanced: 164 healthy, classes 2-4 only 36 / 35 / 13 patients
# - age roughly symmetric, thalach slightly left-skewed (-0.54)
# - chol (skew 1.14, one extreme value 564) and oldpeak (skew 1.27, many zeros) right-skewed
# - trestbps has some high values (up to 200); outliers are medically plausible -> kept





# ================ IDA Step7: Understanding the relationships between variables ================
# 7a) correlation matrices, Pearson and Spearman [in-class]
# Pearson = linear relationship; Spearman = rank-based, monotonic relationship,
# robust to outliers and suited for ordinal values -> used for interpretation
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
for ax, method in zip(axes, ["pearson", "spearman"]):
    corr = df.corr(method=method)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1,
                ax=ax, annot_kws={"size": 8})
    ax.set_title(method.capitalize())
plt.suptitle("IDA Step 7a – Correlation matrices (Pearson and Spearman)")
plt.tight_layout()
plt.show()

# 7b) scatter matrix of continuous features, coloured by disease [in-class]
# black = no disease (num = 0), yellow = disease (num > 0)
pd.plotting.scatter_matrix(df[num_cols], c=(df["num"] > 0).astype(int), marker="o", s=20,
                           hist_kwds={"bins": 15}, figsize=(10, 10),
                           cmap=plt.get_cmap("gnuplot"), alpha=0.7)
fig = plt.gcf()
fig.suptitle("IDA Step 7b – Scatter matrix coloured by disease", y=0.995)
plt.show()

# 7c) continuous features per diagnosis class [in-class: box plot per class]
fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
for ax, col in zip(axes, num_cols):
    sns.boxplot(x="num", y=col, data=df, ax=ax, color="lightsteelblue")
    ax.set_title(col)
plt.suptitle("IDA Step 7c – Continuous features per diagnosis class")
plt.tight_layout()
plt.show()

# Observations:
# - features are mostly weakly/moderately correlated; strongest pair oldpeak-slope (0.59),
#   thalach negatively correlated with age, oldpeak, slope, exang (about -0.4) -> no redundant feature
# - strongest association with the target: thal (0.54), ca (0.53), cp (0.49), oldpeak (0.46),
#   exang (0.44), thalach (-0.44); fbs (0.05), chol (0.11), trestbps (0.14) almost unrelated
# - scatter matrix: healthy and diseased overlap in every 2-D projection -> no perfect separation
# - box plots: classes 1-4 overlap strongly -> predicting severity will be much harder
#   than predicting disease yes/no
# - lecture caution: we do not analyse the target relationships further, to avoid
#   circular conclusions / data leakage

