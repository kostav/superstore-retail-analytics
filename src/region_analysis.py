# =============================================================================
# SUPERSTORE — SECTION 1: REGION ROBUSTNESS ANALYSIS
# =============================================================================
# Purpose
# ---------------
# “REGION = Space / Environment”: Measure where the system “works.”
# We assess:
#   • Performance level (Profit, Profit Margin)
#   • Stability/Volatility (std, CV)
#   • Distribution shape (skewness, kurtosis)
#   • Statistical differences across regions (ANOVA / Kruskal, Levene)
#
# Key questions
# -------------
#   Q1) Which regions create/burn value? (profit, margin)
#   Q2) How stable is performance by region? (std, CV)
#   Q3) Are differences in profit across regions statistically significant?
#
# Deliverables
# ------------
#   • region_stats: KPI table per region (sales, profit, margin, n, mean, std, CV)
#   • dist_summary: diagnostics (n, mean, std, skew, kurtosis, normality p-values)
#   • report: compact object with Levene/ANOVA/Kruskal + assumption flags
#   • plots: boxplot, hist+KDE, Q–Q per region
#
# Methodology
# -----------
#   1) Descriptives per region (aggregates + CV)
#   2) Normality checks (Shapiro on ≤500 subsample, D’Agostino-Pearson for whole sample)
#   3) Homogeneity of variances (Levene)
#   4) Choose primary test:
#        • If (normality AND equal variances) → One-way ANOVA (means)
#        • Else → Kruskal–Wallis (ranks/medians)
#
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import shapiro, normaltest, levene, skew, kurtosis, kruskal
import scipy.stats as sps



# Load data
df = pd.read_pickle("clean_superstore.pkl")

# =============================================================================
# 1) As a first thing we calculate the core KPIs per region (Region-level aggregates)
# =============================================================================
region_stats = df.groupby("region").agg(
    sales=("sales", "sum"),
    profit=("profit", "sum"),
    n=("profit", "size"),
    mean_profit=("profit", "mean"),
    std_profit=("profit", "std")
).reset_index()

# Performance & volatility
region_stats["ProfitMargin"] = region_stats["profit"] / region_stats["sales"]
region_stats["CV_profit"] = region_stats["std_profit"] / region_stats["mean_profit"]

print("\n=== Region Stats ===")
print(region_stats)

# =============================================================================
# 2) In order to proceed with the statistical testing we need to create an easy
# to use structure that can provide to us all the corresponding info per region
# on an ordered manner, thus we are creating the following order list:
# =============================================================================
regions = sorted(df["region"].dropna().unique())
# Get all unique region names (no NaN) and sort them for consistent order
groups = {r: df.loc[df["region"] == r, "profit"].dropna() for r in regions}
# Build a dictionary mapping each region to its profit Series (clean, ready for analysis)
group_list = [groups[r] for r in regions]
# Create an ordered list of profit Series to feed directly into statistical tests


# =============================================================================
# 3) We now proceed with normality checks for profit distribution per region using:
# i ) Shapiro–Wilk (parametric approximation to evaluate normality, by comparing
#     the observed order statistics to those theoretically expected under a normal model.)
# ii) D’Agostino–Pearson (evaluates normality by jointly assessing skewness and kurtosis
#     — i.e., the symmetry and tail behavior of the distribution.)
#
# => We will construct a table (i.e. summary_rows) with all needed statistical
#    information about the regions that concern normality
# =============================================================================

summary_rows = []
# This table will store all appropriate info needed for evaluate normality for every region

for r, g in groups.items():
    # r: region name (e.g., "East"), g: profit Series for that region (dropna already applied upstream)
    # Thus we are running the normality checks (Shapiro-Wilk & D’Agostino–Pearson) using the profit values of each region

    # ----------------------------------------------------------------------
    # Shapiro–Wilk on a subset of profit values from *within the same region*
    # ----------------------------------------------------------------------
    # If n is large, Shapiro becomes hyper-sensitive and can even flag small variations as very significant,
    # thus risking to give a fasle evaluation of the normality test of the method.
    # As a result we put an upper limit of 500 observations while on our other normality check
    # (D’Agostino–Pearson) we will be using the full sample.
    n_g = len(g)
    g_for_shapiro = g.sample(500, random_state=42) if n_g > 500 else g

    # ----------------------------------------------------------------------
    # Shapiro–Wilk normality (W → p) on 500 samples
    # H0: normality.  W = (Σ a_i x_(i))² / Σ (x_i − x̄)² (order-statistics alignment).
    # p = P(W ≤ W_obs | H0).  Small p ⇒ reject normality.
    # ----------------------------------------------------------------------
    try:
        sh_p = shapiro(g_for_shapiro).pvalue
    except Exception:
        sh_p = np.nan

    # ----------------------------------------------------------------------
    # D’Agostino–Pearson (K² → p) on the full sample
    # H0: normality.  Uses Z_skew and Z_kurt (standardized) → K² = Z_skew² + Z_kurt² ~ χ²(df=2).
    # p = P(χ² ≥ K²_obs).  Small p ⇒ reject normality due to asymmetry/ heavy tails.
    # ----------------------------------------------------------------------
    try:
        dag_p = normaltest(g).pvalue if n_g >= 8 else np.nan
    except Exception:
        dag_p = np.nan

    # ----------------------------------------------------------------------
    # Bias policy for skew/kurtosis calculation
    # ----------------------------------------------------------------------
    #  => Below we want to calculate the skew and kurtosis and based on the size sample we need to take into
    #     consideration the following rationale:
    #   • For n ≤ 500, sample of that size systematically *underestimate* the true asymmetry and tail heaviness.
    #     (The sample “speaks too much for itself” and sees only a narrow piece of the whole.)
    #     Setting bias=False applies Fisher–Pearson corrections, restoring a fairer reflection of the population shape.
    #   • For n > 500, the estimators are effectively unbiased; keeping bias=True avoids tiny artificial inflation
    #     and preserves numerical stability.
    # ----------------------------------------------------------------------

    if n_g <= 500:
        # small or medium sample → apply correction
        bias_flag = False  # bias=False → Fisher–Pearson correction ON
    else:
        # large sample → keep raw estimators
        bias_flag = True  # bias=True → use raw, asymptotically unbiased estimators


    summary_rows.append({ # Now we pass all the values of the table that concern normality:
        "region": r,
        "n": n_g,

        # Mean and unbiased sample std (ddof=1)
        "mean": float(np.mean(g)) if n_g else np.nan,
        "std": float(np.std(g, ddof=1)) if n_g > 1 else np.nan,

        # Skewness — 3rd central moment / σ³ (direction of tail).
        # bias flag: False (corrected) for n ≤ 500; True (raw) for n > 500.
        "skew": float(skew(g, bias=bias_flag)) if n_g > 2 else np.nan,

        # Kurtosis(excess) — 4th central moment / σ⁴ − 3 (tail weight).
        # fisher=True ⇒ return *excess* kurtosis (i.e. k(excess) = 0 means Normal).
        "kurtosis(excess)": float(kurtosis(g, fisher=True, bias=bias_flag)) if n_g > 3 else np.nan,

        # Normality diagnostics:
        #   • shapiro_p → order-statistics alignment (moderate-n probe)
        #   • dagostino_p → skew/kurtosis deformation (full-n probe)
        "shapiro_p": sh_p,
        "dagostino_p": dag_p
    })

# ----------------------------------------------------------------------
# Combine all regional dictionaries into a single DataFrame
# ----------------------------------------------------------------------
dist_summary = pd.DataFrame(summary_rows).sort_values("region")

print("\n=== Per-Region Distribution Summary ===")
print(dist_summary)

# ----------------------------------------------------------------------
# Conservative normality flag:
# True only if *all available* normality p-values > α.
# ----------------------------------------------------------------------
def _is_normal_row(row, alpha=0.05):               # decide “normal” per row based on p-values
    ps = [p for p in (row["shapiro_p"], row["dagostino_p"]) if not pd.isna(p)]  # collect non-NaN p-values
    if not ps:                                     # no evidence → treat as non-normal conservatively
        return False
    return all(p > alpha for p in ps)              # normal only if every available test does not reject H0

# ---- Apply decision rule and show final table with the flag
dist_summary["normal_flag"] = dist_summary.apply(_is_normal_row, axis=1)  # add boolean normality flag per region
print("\n=== Per-Region Distribution Summary (with normal_flag) ===")
print(dist_summary)


# =============================================================================
# 4) We now need to check on more crucial element for our statistical testings,
#    that is the homogeneity of variances this is important because if the variances
#    of the different regions don't behave on a homogenous way the f-tests of ANOVA
#    tend to give not reliable results.
#    We are achieving this by the Leneve method that assesses the equality of variances
#    by performing an ANOVA on the absolute deviations from each group’s mean
#    (Levene: H0 equal variances)
# =============================================================================
lev_stat, lev_p = levene(*group_list)
print("\n=== Levene’s Test for Equal Variances ===")
print(f"Statistic = {lev_stat:.3f}, p-value = {lev_p:.5f}")

# =============================================================================
# 5) Now that we have calculated all the statistic parameters needed to evaluate
#    which method to use we first calculate both of them (i.e. ANOVA & Kruskal),
#    sth that is useful in order to have a robust view on our sample (since if
#    both results agree regarless of normality and variance assumptions this
#    gives an even more strong case of our findings and our overall decision).
# =============================================================================
anova_f, anova_p = stats.f_oneway(*group_list)
print("\n=== One-way ANOVA (profit ~ region) ===")
print(f"F-statistic = {anova_f:.3f}, p-value = {anova_p:.5f}")

kw_h, kw_p = kruskal(*group_list)
print("\n=== Kruskal–Wallis (profit ~ region) ===")
print(f"H-statistic = {kw_h:.3f}, p-value = {kw_p:.5f}")

# =============================================================================
# 6) Following all the above logic is time to evaluate our tests and based on
#    the assumptions proceed with the appropriate decision. Mainly:
#    i) Normality and    Equal Variance hold        => We proceed with parametric
#                                                      instrument of ANOVA
#    ii)Normality and/or Equal Variance don't hold  => We proceed with non-parametric
#                                                      instrument of Kruskal–Wallis
#
# =>The rationale behind our decision policy lies in the fact that ANOVA compares
#   group means, which must approximately follow a normal distribution and exhibit
#   equal variances. Only under these conditions does the comparison reflect genuine
#   differences in values, rather than random noise or local/extreme fluctuations.
#   In case these assumptions are not met, an appropriate way to get a statistical
#   decent measurement of the differences of the values between the groups is
#   Kruskal–Wallis method, in order to give a raugh but statistically decent estimate
#   of their differences. Where the method is disregarding the means and instead ranks
#   all observations across groups, comparing their average ranks to detect distributional differences.
# =============================================================================
alpha = 0.05
all_normal = bool(dist_summary["normal_flag"].all())
equal_vars = bool(lev_p > alpha)

print("\n=== Assumption Check Summary ===")
print(f"All groups normal? {all_normal}")
print(f"Equal variances (Levene p>{alpha})? {equal_vars}")

if all_normal and equal_vars:
    print("→ Assumptions satisfied. Prefer ANOVA (mean-based).")
else:
    violated = []
    if not all_normal:
        violated.append("non-normality")
    if not equal_vars:
        violated.append("heteroscedasticity")
    print(f"→ Assumptions violated ({', '.join(violated)}). Prefer Kruskal–Wallis.")

# =============================================================================
# 7) To further validate our normality analysis, we complement the statistical
#    tests with visual diagnostics of the profit distribution — including boxplots,
#    histograms with KDE curves, and Q–Q plots for each region.
# =============================================================================
sns.boxplot(x="region", y="profit", data=df)
plt.title("Profit Distribution by Region")
plt.xlabel("Region"); plt.ylabel("Profit")
plt.show()

for r in regions:
    sns.histplot(groups[r], kde=True)
    plt.title(f"Profit Histogram + KDE — {r}")
    plt.xlabel("Profit"); plt.ylabel("Count")
    plt.show()

for r in regions:
    sps.probplot(groups[r], dist="norm", plot=plt)
    plt.title(f"Q–Q Plot — {r}")
    plt.show()

# =============================================================================
# 8) Finally we conclude the section with this structured report dictionary to
#    store all main outputs together
# =============================================================================

report = {
    "region_stats": region_stats,       # aggregated sales/profit/margin table per region
    "dist_summary": dist_summary,       # normality + shape diagnostics (mean, std, skew, kurtosis, p-values)
    "levene": {"stat": lev_stat, "p": lev_p},    # homogeneity of variances test (Levene)
    "anova": {"F": anova_f, "p": anova_p},       # parametric mean-comparison test (One-way ANOVA)
    "kruskal": {"H": kw_h, "p": kw_p},           # nonparametric robust alternative (Kruskal–Wallis)
    "assumptions": {                             # basic conditions controlling test choice
        "all_normal": all_normal,                # True if all groups passed normality checks
        "equal_vars": equal_vars                 # True if Levene p > α (variances not significantly different)
    }
}
