# ----------------------------------------------------------------------
# A) Prep + thresholds
# ----------------------------------------------------------------------
df_seg = product_metrics.copy()
CV_THRESHOLD       = 1.0   # stable vs volatile (tunable)
LOSS_PCT_THRESHOLD = 0.30  # low vs high loss frequency (30%)

# Guardrails for CV/mean
df_seg["cv_order_profit"]   = df_seg["cv_order_profit"].replace([np.inf, -np.inf], np.nan)
df_seg["mean_order_profit"] = df_seg["mean_order_profit"].fillna(0)
df_seg["cv_order_profit"]   = df_seg["cv_order_profit"].fillna(np.inf)  # unknown → treat as high volatility
df_seg["pct_orders_loss"]   = df_seg["pct_orders_loss"].fillna(0)

# ----------------------------------------------------------------------
# B) 6-way segmentation
# ----------------------------------------------------------------------
df_seg["segment"] = np.select(
    [
        # Profitable + Stable
        (df_seg["mean_order_profit"] > 0) & (df_seg["cv_order_profit"] <= CV_THRESHOLD) & (df_seg["pct_orders_loss"] <= LOSS_PCT_THRESHOLD),
        (df_seg["mean_order_profit"] > 0) & (df_seg["cv_order_profit"] <= CV_THRESHOLD) & (df_seg["pct_orders_loss"] >  LOSS_PCT_THRESHOLD),

        # Profitable + Volatile
        (df_seg["mean_order_profit"] > 0) & (df_seg["cv_order_profit"] >  CV_THRESHOLD) & (df_seg["pct_orders_loss"] <= LOSS_PCT_THRESHOLD),
        (df_seg["mean_order_profit"] > 0) & (df_seg["cv_order_profit"] >  CV_THRESHOLD) & (df_seg["pct_orders_loss"] >  LOSS_PCT_THRESHOLD),

        # Unprofitable
        (df_seg["mean_order_profit"] <= 0) & (df_seg["cv_order_profit"] <= CV_THRESHOLD),
        (df_seg["mean_order_profit"] <= 0) & (df_seg["cv_order_profit"] >  CV_THRESHOLD),
    ],
    [
        "Hero",               # profitable, stable, low-loss
        "Shaky Hero",         # profitable, stable, high-loss
        "Wildcard",           # profitable, volatile, low-loss
        "Dangerous Wildcard", # profitable, volatile, high-loss
        "Consistent Loser",   # unprofitable, stable
        "Disaster",           # unprofitable, volatile
    ],
    default="Unclassified"
)
# ----------------------------------------------------------------------
# C) Composition table (counts / %)
# ----------------------------------------------------------------------
order = ["Hero","Shaky Hero","Wildcard","Dangerous Wildcard","Consistent Loser","Disaster","Unclassified"]
summary = (
    df_seg["segment"]
    .value_counts()
    .reindex(order, fill_value=0)
    .rename_axis("Segment")
    .reset_index(name="Count")
)
summary["Percent"] = (summary["Count"] / summary["Count"].sum() * 100).round(1)
print(summary)

# ----------------------------------------------------------------------
# D) Segment Scorecard (visual table; CV used internally for stability only)
#     – Keeps the original StabilityScore: (1 - 0.6*CV_norm - 0.4*loss_norm)*100
# ----------------------------------------------------------------------
from IPython.display import display, HTML

order = ["Hero","Shaky Hero","Wildcard","Dangerous Wildcard","Consistent Loser","Disaster","Unclassified"]

total_products        = len(df_seg)
total_positive_profit = df_seg["total_profit"].clip(lower=0).sum()
total_losses_abs      = (-df_seg["total_profit"].clip(upper=0)).sum()
net_profit            = df_seg["total_profit"].sum()

grp = df_seg.groupby("segment", dropna=False)
agg = grp.agg(
    TotalProfit       = ("total_profit", "sum"),
    MedianLossRate    = ("pct_orders_loss", "median"),
    MedianMeanOrdProf = ("mean_order_profit", "median"),
    MedianCV          = ("cv_order_profit", "median"),   # internal use only
)
agg.insert(0, "Count", grp.size())
agg = agg.reindex(order, fill_value=0)

def safe_div(num, den):
    return np.where(den == 0, np.nan, num / den)

agg["% of Products"]          = (agg["Count"] / total_products) * 100
agg["% of Positives (share)"] = safe_div(agg["TotalProfit"].clip(lower=0), total_positive_profit) * 100
agg["% of Losses (share)"]    = safe_div((-agg["TotalProfit"].clip(upper=0)), total_losses_abs) * 100
agg["% of NET (impact)"]      = safe_div(agg["TotalProfit"], net_profit) * 100

# StabilityScore (original composite) → normalize CV to [0,1] via clip 0..3
cv_clip   = agg["MedianCV"].clip(lower=0, upper=3) / 3.0
loss_clip = agg["MedianLossRate"].clip(lower=0, upper=1)
agg["StabilityScore(0-100)"] = (1 - 0.6*cv_clip - 0.4*loss_clip) * 100

# Hide MedianCV from the visible output
agg = agg.drop(columns=["MedianCV"])

# Clean display
pct_cols = ["% of Products","% of Positives (share)","% of Losses (share)","% of NET (impact)","StabilityScore(0-100)"]
agg[pct_cols] = agg[pct_cols].round(1)
agg[["MedianLossRate","MedianMeanOrdProf","TotalProfit"]] = agg[["MedianLossRate","MedianMeanOrdProf","TotalProfit"]].round(2)

segment_table = agg.reset_index().rename(columns={"segment":"Segment"})

headline = (
    f"<b>Segment Scorecard for {total_products} Products</b><br>"
    f"<i>Net company profit: {net_profit:,.2f}</i><br>"
    "‘% of Positives’ ≈ 100% across profitable products; "
    "‘% of Losses’ ≈ 100% across loss-makers; "
    "‘% of NET’ shows each segment’s impact on the bottom line.<br>"
    "StabilityScore (0–100) merges volatility (CV) and loss frequency into one intuitive index."
)

styled = (
    segment_table
    .style
    .format({
        "% of Products": "{:.1f}%",
        "% of Positives (share)": "{:.1f}%",
        "% of Losses (share)": "{:.1f}%",
        "% of NET (impact)": "{:.1f}%",
        "StabilityScore(0-100)": "{:.1f}",
        "MedianLossRate": "{:.2%}",
        "MedianMeanOrdProf": "{:.2f}",
        "TotalProfit": "{:,.2f}",
    })
    .set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#1f77b4'),
                                     ('color', 'white'),
                                     ('text-align', 'center'),
                                     ('font-weight', 'bold')]},
        {'selector': 'td', 'props': [('text-align', 'center')]}
    ])
    .background_gradient(subset=["% of NET (impact)"], cmap="Greens")
    .background_gradient(subset=["StabilityScore(0-100)"], cmap="Blues")
)
