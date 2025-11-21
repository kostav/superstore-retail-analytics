# =============================================================================
# SUPERSTORE — SECTION 3: DISCOUNT CORRELATION ANALYSIS
# =============================================================================
# Purpose
# -----------------------------------------------------------------------------
# We want to measure how discounts are linked to both **Profit** and **Sales**,
# and check whether heavy discounting systematically harms profitability.
# We use Correlation measurements: Where we take as ground-floor that
# **correlation ≠ causation**, which means that discounts and profit move together,
# but that doesn’t mean one directly causes the other. Other drivers could push both.
# That’s why we keep correlation as a *diagnostic lens*, not a final causal claim.
#
# Method
# -----------------------------------------------------------------------------
#
# 1) Two types of correlation:
#    A)  Pearson correlation (r): measures **linear association** between continuous
#        variables. Sensitive to outliers.
#    B)  Spearman correlation (ρ):* measures **monotonic association** using ranks.
#        Robust to outliers, non-normality, and non-linear but consistent trends.
#
#    → Since the discount-profit link is *non-linear* we are using both measurments
#      in order to avoid underestimation of the link (sth that would happen if we
#      were using a linear assiciation measuremnt in order to express a non linear link)
#
#    →Interpretation rules:
#     - Both r and ρ ∈ [–1, 1]. Negative = higher discounts → lower profit.
#     - Benchmarks: |0.1–0.3| = weak, |0.3–0.5| = moderate, >0.5 = strong.
#
#    →Statistical significance (H₀ testing):
#     p-values tell us if the observed association is unlikely to happen by chance
#     if the true relationship were zero. With thousands of rows, even tiny correlations
#     will show as “significant.” That’s why we never stop at p — it only rejects H₀,
#     it doesn’t tell us *how important* the effect is.
#
# 2) Confidence intervals:
#    A single r is just one sample estimate. CIs wrap it with a range of plausible values
#    for the “true” correlation. Narrow CIs mean we can trust the stability of the result;
#    wide CIs mean more uncertainty. Example: r = –0.22, CI [–0.24, –0.20] says: “we’re
#    very confident the real effect is moderately negative, not zero and not weakly positive.”
#    This communicates both *strength* and *reliability*.
#
# 3) Grouped analysis:
#  - Recompute r and ρ by Category to uncover segment-specific dynamics.
#  - Guard against degenerate cases (std = 0 → correlation undefined).
#
# 4) Robustness to outliers:
#    Trimming outliers (99th percentile sales) shows the pattern isn’t a mirage
#    caused by a few giant orders. That strengthens *business trust*: the
#    discount–profit link is structural.
#
# 5) Visualizations:
#  - Scatter + OLS trend line → shows Pearson-style linear slope.
#  - Binned means (avg profit per discount bracket) → reveals Spearman-style
#    threshold effect.
#  - Heatmap of correlation matrix (discount, profit, sales)
#



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import pearsonr, spearmanr, norm
import seaborn as sns

df = pd.read_pickle("clean_superstore.pkl")


# =============================================================================
# 1) Standard Pearson and Spearman Correlation Calculation
# =============================================================================

# ----------------------------------------------------------------------
# A) Pearson (linear association)
# ----------------------------------------------------------------------
r_p, p_p = pearsonr(df["discount"], df["profit"])
r_p1, p_p1 = pearsonr(df["discount"], df["sales"])
# ----------------------------------------------------------------------
# B) Spearman (monotonic / robust to outliers)
# ----------------------------------------------------------------------
r_s, p_s = spearmanr(df["discount"], df["profit"])
r_s1, p_s1 = spearmanr(df["discount"], df["sales"])

print(f"Pearson r(Discount, Profit) = {r_p:.3f}, p = {p_p:.3g}")
print(f"Spearman ρ(Discount, Profit) = {r_s:.3f}, p = {p_s:.3g}")

print(f"Pearson r(Discount, Sales) = {r_p1:.3f}, p = {p_p1:.3g}")
print(f"Spearman ρ(Discount, Sales) = {r_s1:.3f}, p = {p_s1:.3g}")


# =============================================================================
# 2) Pearson CI calculation via Fisher Z-transformation
# =============================================================================
n = df.shape[0]                     # Get number of rows

z  = np.arctanh(r_p)                 # Fisher z inverse hyperbolic tangent to the Pearson correlation coefficient.
z1 = np.arctanh(r_p1)

se = 1/np.sqrt(n-3)                 # Calculates the standard error of Fisher z
zcrit = norm.ppf(0.975)             # Get 97.5th percentile of the standard normal distribution (95% two-sided)

lo  = np.tanh(z - zcrit*se)          # Get the min value for correlation: tan(z-z*XSE(z))
lo1 = np.tanh(z1 - zcrit*se)

hi = np.tanh(z + zcrit*se)          # Get the max value for correlation: tan(z+z*XSE(z))
hi1 = np.tanh(z1 + zcrit*se)


print(f"Pearson r(Discount, Profit) = {r_p:.3f}, 95% CI [{lo:.3f}, {hi:.3f}]")
print(f"Pearson r(Discount, Sales)  = {r_p1:.3f}, 95% CI [{lo1:.3f}, {hi1:.3f}]")


# =============================================================================
# 3) Per group Correlation Calculation
# =============================================================================

# keep only needed cols and drop rows with missing values
sub = df[["category", "discount", "profit", "sales" ]].dropna().copy()

# Function to assure that we calculate correlation only when standard deviation is non-zero
def safe_pearson(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    return pearsonr(x, y)

def safe_spearman(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    return spearmanr(x, y)


by_cat = ( sub.groupby("category", observed=True) # For each Category
       .apply(lambda g: pd.Series({               # Call local function that calculates and returns
           "n": len(g),                           # number of products of category
           "pearson_r_profit": safe_pearson(g["discount"], g["profit"])[0], # Correlation between discount and profit
           "p_value_profit":   safe_pearson(g["discount"], g["profit"])[1], # p-value of corresponding correlation
           "pearson_r_sales": safe_pearson(g["discount"], g["sales"])[0],   # Correlation between discount and sales
           "p_value_sales":   safe_pearson(g["discount"], g["sales"])[1],   # p-value of corresponding correlation


           "spearman_r_profit": safe_spearman(g["discount"], g["profit"])[0], # Correlation between discount and profit
           "p_value_profit":   safe_spearman(g["discount"], g["profit"])[1], # p-value of corresponding correlation
           "spearman_r_sales": safe_spearman(g["discount"], g["sales"])[0],   # Correlation between discount and sales
           "p_value_sales":   safe_spearman(g["discount"], g["sales"])[1]    # p-value of corresponding correlation
       }))
       .reset_index()
       .sort_values("pearson_r_sales")
)

print(by_cat)

#Calulcation correlation between discount-sales-profit altogether
block = df[["sales", "discount", "profit"]].dropna().copy()
corr_mat = block.corr(method="pearson")   # linear associations
print(corr_mat.round(3))

# =============================================================================
# 4) Trim extreme Sales (≤ 99th percentile) and recompute Pearson r(Discount, Profit)
# =============================================================================

# 99th percentile cutoff for Sales
p99 = sub["sales"].quantile(0.99)

# Trim dataset to reduce leverage from huge Sales
df_trim = sub[sub["sales"] <= p99].copy()

# Pearson correlation on trimmed data
r_trim, p_trim = pearsonr(df_trim["discount"], df_trim["profit"])
print(f"[Trimmed @ Sales ≤ 99th pct] Pearson r(Discount, Profit) = {r_trim:.3f}, p = {p_trim:.3g}")
print(f"Rows kept: {len(df_trim)} / {len(sub)}")

# =============================================================================
# 5) Vizualizations
# =============================================================================

# ----------------------------------------------------------------------
# A) Scatter and Trend Line
# ----------------------------------------------------------------------
# x = Discount, y = Profit
x = df["discount"].to_numpy()
y = df["profit"].to_numpy()

# Calculate b1,b0 using simple linear fit: y = b1*x + b0
b1, b0 = np.polyfit(x, y, 1)

# Define scatter plot figure parameters
plt.figure(figsize=(7, 5))
plt.scatter(x, y, s=8, alpha=0.5)

# Generate 200 equally spaced Discount values between min and max, in order to draw the fitted linear trend line continuously.
xx = np.linspace(np.nanmin(x), np.nanmax(x), 200)
plt.plot(xx, b1*xx + b0)


# Put title, labels and plot figure
plt.xlabel("Discount")
plt.ylabel("Profit")
plt.title(f"Discount vs Profit (Pearson r = {r_p:.2f})")
plt.ylim(-1000, 1000)   # zoom in to see slope more clearly
plt.show()

# ----------------------------------------------------------------------
# B) Bar chart of average profit per discount bracket
# ----------------------------------------------------------------------

# Define intervals and labels for discount
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8]
labels = ["0–10%", "10–20%", "20–30%", "30–40%", "40–50%", "50–80%"]

#Calculate and pass to df the avg profit be discount bin
avg_profit_per_bin = (df.groupby(pd.cut(df["discount"], bins=bins, labels=labels))["profit"].mean())

#Plot the corresponding barchart with the calculated values
plt.figure(figsize=(8,5))
avg_profit_per_bin.plot(kind="bar", color=["green" if v > 0 else "red" for v in avg_profit_per_bin])
plt.axhline(0, color="black", linewidth=1)
plt.title("Average Profit per Discount Bracket")
plt.ylabel("Average Profit")
plt.xlabel("Discount Bracket")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# C) Heatmap of correlation matrix (discount, profit, sales)
# ----------------------------------------------------------------------
plt.figure(figsize=(6,4))
sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="RdBu_r", center=0, cbar=False)
plt.title("Correlation Matrix (Discount, Profit, Sales)")
plt.tight_layout()
plt.show()


