# =============================================================================
# SUPERSTORE — SECTION 3: CUSTOMER VALUE EVOLUTION
# =============================================================================
# Purpose
# -----------------------------------------------------------------------------
# This section investigates how customer relationships evolve through time
# and how these behavioral dynamics translate into sales growth,
# retention, and long-term profitability.
#
# In essence, we model the *evolutionary mechanics* of the customer base:
# how value is generated, retained, and concentrated across different
# generations of customers.
#
# The analysis unfolds in three interlinked pillars:
#
# 1) Cohort Foundations — Behavioral timeline and retention structure
#    • Establishes the core time-based framework (customer × month panel)
#      that tracks how each customer cohort behaves and sustains activity.
#    • Measures Retention Rate and repurchase frequency as fundamental
#      indicators of customer longevity and organic revenue growth.
#
# 2) Comparative Value Base — Lifetime potential and value concentration
#    • Introduces Observed Lifetime Value over H months -LTV(Η)- as a standardized metric
#      to quantify cumulative profit generation within a defined horizon.
#    • Combines LTV analysis with Pareto concentration curves to benchmark
#      both the *magnitude* (total value) and *distribution* (inequality)
#      of profitability across customers and cohorts.
#
# 3) Micro-Analysis of Retention and Value Behavior — The CX dynamic layer
#    • Translates customer transactions into behavioral stages (Funnel framework)
#      to examine where value creation accelerates or leaks along the lifecycle.
#    • Links behavioral patterns (activation, loyalty, high-value transition)
#      to economic outcomes (profit concentration, lifetime depth).
#
# Overall goal:
# To transform static sales data into a dynamic, time-based understanding of
# *how customers evolve, retain, and generate value* — connecting behavioral
# patterns with measurable financial impact.





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import pearsonr, spearmanr, norm
import seaborn as sns


df = pd.read_pickle("clean_superstore.pkl") # We load data
print("Columns:", df.columns.tolist())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


########################################################################################################################
##############################~1) Cohort Foundations — Historical sales tracking ~######################################
########################################################################################################################
# Goal:
#   • Create a solid base where the overall sales-period of superstore is divided in montlhy cohorts, where each cohort
#   • shows how many new customers arrived on that month and for how long they continued purchasing.
# Why:
#   • Gives a way for us to have a first level insight of the impact of new and returning customers
#   • Observe the retention rate of our customers overall the systemic sales period and in specific time-frames
# How to read:
#   • Yearly New vs Returning: Comparison between new and returning customers impact on profits.
#   • Retention heatmap: Overall display of sales distribution among time divided on cohort-time frame.
#   • Average retention decay curve: Mean of sales continuation for all cxs across time.




################################################################################
# i) Functions for calculation of all nessecary info for the cohort analysis:  #
#    [prepare_types ; add_first_purchase ; new_vs_returning_yearly ;           #
#     cohort_retention ; summarize_cohort_results]                             #
################################################################################


# =============================================================================
# A) prepare_types: Is ensuring that all date columns are transformed to
#   corresponding panda date-time format in order to proceed with the analysis
#   without any issues
# =============================================================================
def prepare_types(df, date_col):
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        # We check if the date column is already on the correct panda format, in case it's not we proceed as below:
        df = df.copy() # We first create a copy of the original dataframe in order to avoid changes in the original data
        df[date_col] = pd.to_datetime(df[date_col]) # We transform the column to the correct panda format
    return df

# =============================================================================
# B) add_first_purchase: It takes raw data from superstore and creates
# for each customer the corresponding data to build a cohort analysis such as
# first purchase date, cohort year, order year and the relative timing (index)
# of all subsequent orders.
# =============================================================================

def add_first_purchase(df, cust_col, date_col):
    df = df.copy() # We first create a copy of the original dataframe in order to avoid changes in the original data
    firsts = (     # we calculate a small table "firsts" with each customer and their corresponding first puchase date
        df.groupby(cust_col, as_index=False)[date_col]
          .min()
          .rename(columns={date_col: "FirstPurchaseDate"})
    )
    df = df.merge(firsts, on=cust_col, how="left") # We insert to the df the corresponding first purchase date for each customer
    df["CohortYear"] = df["FirstPurchaseDate"].dt.year # We insert also the Cohort Year for when customer first bought
    df["OrderYear"]  = df[date_col].dt.year # We insert the year of current order (this is useful info in order to be able
                                            # to trace between new (CohortYear == Order Year) and returning customer (CohortYear != Order Year)
    df["CohortIndex"] = ( # We create a new column that calculates the distance (in months) between
                          # each order and the customer's first purchase (cohort month).
                          # We achieve this by converting both dates to monthly periods (Period[M]),
                          # so that their difference can be measured in months.
        (df[date_col].dt.to_period("M").view("i8")
         - df["FirstPurchaseDate"].dt.to_period("M").view("i8")).astype(int)
    )
    return df

# =============================================================================
# C) new_vs_returning_yearly: Returns a df and its corresponding pivot table of all the basic KPIs needed for
#    the cohort analysis
#    => [Year || Group (New/Returning) || # Customers || Total Profit || Total Sales || Avg Profit per Customer || Avg Sales per Customer]
# =============================================================================
def new_vs_returning_yearly(df, cust_col, date_col, sales_col, profit_col):
    df = prepare_types(df, date_col) # we make sure that the dateframe are in the correct panda format
    df = add_first_purchase(df, cust_col, date_col) # We create for each customer basic cohort information as explained above
                                                    # we have new columns: [FirstPurchaseDate, CohortYear, OrderYear, CohortIndex]

    df["IsNewOrderYear"] = (df["OrderYear"] == df["CohortYear"]) # Creation of new column of boolean type to show new/returning customers

    # cust_year: Gives for each customer the total sales and profits for the period between cohort and current year
    # giving also info about if customer is returning or new
    cust_year = (
        df.groupby([cust_col, "CohortYear", "OrderYear", "IsNewOrderYear"], as_index=False) #We achieve this by grouping by the above columns, disactivating the index
          .agg({sales_col: "sum", profit_col: "sum"}) # and aggregating for profit and sales column san
    )
    cust_year["Group"] = np.where(cust_year["IsNewOrderYear"], "New", "Returning") # We then add the Column "Group" and assign the appropriate value of new or returning customer

    # yearly: DataFrame containing all yearly KPIs from the cohort analysis,  for each year, it shows the number of new and returning customers along with
    # their total and average sales and profits.
    yearly = (
        cust_year.groupby(["OrderYear", "Group"], as_index=False)
                 .agg(Customers=(cust_col, "nunique"),
                      TotalSales=(sales_col, "sum"),
                      TotalProfit=(profit_col, "sum"))
    )
    yearly["AvgSales_perCustomer"]  = yearly["TotalSales"]  / yearly["Customers"].replace(0, np.nan)
    yearly["AvgProfit_perCustomer"] = yearly["TotalProfit"] / yearly["Customers"].replace(0, np.nan)

    # pivot: Transform the yearly table into an Excel-style pivot view (Years as rows, New/Returning as columns, metrics side-by-side)
    pivot = (
        yearly.pivot(index="OrderYear", columns="Group",
                     values=["Customers", "TotalSales", "TotalProfit",
                             "AvgSales_perCustomer", "AvgProfit_perCustomer"])
              .sort_index()
    )
    # === Customer flags (New vs Returning) per order for regression analysis ===
    cust = df[['order_id', 'customer_id', 'order_date']].drop_duplicates()
    cust = cust.sort_values(['customer_id', 'order_date']).copy()
    cust['prev_order'] = cust.groupby('customer_id')['order_date'].shift(1)
    cust['new_returning'] = np.where(cust['prev_order'].isna(), 'New', 'Returning')
    flags_export = cust[['order_id', 'customer_id', 'new_returning']]
    flags_export.to_csv(r"C:\Users\kosta\Documents\DA\Projects\1_Super_store\customer_flags.csv", index=False)
    print('✅ Saved artifacts/customer_flags.csv', flags_export.shape)


    return yearly.sort_values(["OrderYear", "Group"]), pivot

# =============================================================================
# D) cohort retention: Performs all necessary computations to return the
#    corresponding cohort retention matrix, where for each cohort it provides the
#    respective retention rates across the entire cohort index period
#    (i.e. all months following the first purchase of that cohort).
# =============================================================================
def cohort_retention(df, cust_col, date_col):
    # Ensure that the date column is in datetime format
    df = prepare_types(df, date_col)

    # Create a safe copy of the dataframe to avoid modifying the original one
    df = df.copy()

    # Convert each order date into a "monthly period" (e.g., 2022-05)
    df["OrderPeriod"] = df[date_col].dt.to_period("M")

    # Identify the first purchase month (cohort) for every customer
    firsts = df.groupby(cust_col)["OrderPeriod"].min()

    # Map that first purchase month back to each transaction (gives each order its cohort)
    df["Cohort"] = df[cust_col].map(firsts)

    # Calculate how many months have passed since each customer's first order,
    # where CohortIndex = 0 for first month, CohortIndex = 1 for next month, etc.)
    df["CohortIndex"] = (df["OrderPeriod"].view("i8") - df["Cohort"].view("i8")).astype(int)

    #  cohort_data: df that shows for each combination of [cohort & cohortIndex] how many unique active customers exist
    #  (i.e. Cohort || CohortIndex || ActiveCustomer)
    #        2021-01       0              5
    #        2021-01       1              3
    #         ...         ...            ...
    #        2021-02       0              4
    #        2021-02       1              2
    #         ...         ...            ...
    cohort_data = (
        df.groupby(["Cohort", "CohortIndex"])[cust_col]
          .nunique()
          .rename("ActiveCustomers")
          .reset_index()
    )

    # cohort_sizes: Calculates for each cohort at CohortIndex 0 how many active customers exists
    # (i.e. initial number of customers for each cohort), this info is stored in column "Cohortsize".
    cohort_sizes = (
        cohort_data[cohort_data["CohortIndex"] == 0][["Cohort", "ActiveCustomers"]]
          .rename(columns={"ActiveCustomers": "CohortSize"})
    )

    # retention: We proceed with the merger of corhort sizes & corhort data in order to have all the info needed for
    # calculation of retention rate for all corhorts within one data frame.
    # (i.e. CohortSize || Cohort || CohortIndex || ActiveCustomer)
    #            5         2021-01       0              5
    #            5         2021-01       1              3
    #           ...         ...         ...            ...
    #            4         2021-02       0              4
    #            4         2021-02       1              2
    #            ...         ...        ...            ...
    retention = cohort_data.merge(cohort_sizes, on="Cohort", how="left")

    # Now that all the information are sorted and collected we calculate and add Retention Rate Columns as follows:
    # Retention Rate = ActiveCustomer / CohortSize
    retention["RetentionRate"] = retention["ActiveCustomers"] / retention["CohortSize"]

    #retention matrix: Now that we have all the information needed calculated we proceed in transforming them in a new
    # matrix that has a more convenient form for the corhort analysis calculation. More specifically:
    # rows = Cohort (first purchase month) ; columns = Months since first purchase (CohortIndex) ; values = Retention rate
    # | Cohort  | 0    | 1    | 2    | 3    | ...
    # | ------- | ---- | ---- | ---- | ---- | ...
    # | 2021-01 | 1.00 | 0.60 | 0.40 | 0.20 | ...
    # | 2021-02 | 1.00 | 0.80 | 0.60 | 0.40 | ...
    # | ...     | ...  | ...  | ...  | ...  | ...

    retention_matrix = (
        retention.pivot(index="Cohort", columns="CohortIndex", values="RetentionRate")
                 .fillna(0.0)
                 .sort_index()
    )
    # Return the final matrix ready for visualization
    return retention_matrix


# =============================================================================
# E) summarize_cohort_results: It calculates complementary information from the
#    retention matrix used for further analysis. More specifically it creates
#    and returns the following three matrixes:
#    -summary_metrics: Average customer retention over the first
#     3,6,12 months since acquisition
#    -lifetime_by_cohort: Average lifetime/active months per customer
#    -decay_curve: Average retantion rate per months for all cohorts/orders
# =============================================================================

def summarize_cohort_results(retention_matrix):

    #We make sure that h3,h6,h12, will take the corresponding values of [3,6,12 months] only if the cohort analysis
    # arrives so far in time, in case the analysis is smaller than these time framework is will take the max time period
    # of the analysis
    n_cols = retention_matrix.shape[1]
    h3, h6, h12 = min(3,n_cols), min(6,n_cols), min(12,n_cols)

    # Calculate summary_metrics only if the corresponding time intervals exist (h3/h6/h12 > 0)
    # For each interval, we compute the average retention across all cohorts
    # during the first 3, 6, and 12 months respectively.
    summary_metrics = {
        "avg_retention_3m":  retention_matrix.iloc[:, :h3].mean().mean()  if h3>0  else np.nan,
        "avg_retention_6m":  retention_matrix.iloc[:, :h6].mean().mean()  if h6>0  else np.nan,
        "avg_retention_12m": retention_matrix.iloc[:, :h12].mean().mean() if h12>0 else np.nan
    }
    lifetime_by_cohort = retention_matrix.sum(axis=1) # Caclulate the sum of each row in order to take the cumulative retentation rate for each corhort
    decay_curve = retention_matrix.mean(axis=0) # Calculate the mean of each column in order to take the average of each month for all corhots
    return summary_metrics, lifetime_by_cohort, decay_curve


################################################################################
# ii)  Functions for creating Cohort Analysis Plots and Displays:              #
#      [plot_yearly_new_vs_returning ; plot_retention_heatmap ;                #
#       plot_retention_decay_curve ; print_cohort_insight]                     #
################################################################################


# =============================================================================
# A) plot_yearly_new_vs_returning: Produces two aligned line charts that contrast
# New vs Returning customers:
#    ->Top:    Total Profit by calendar year and group (New/Returning).
#    ->Bottom: Average Profit per Customer by year and group.
# =============================================================================
def plot_yearly_new_vs_returning(yearly_tidy):
    # Create a 2-row figure sharing the x-axis (years)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # --- Top chart: Total Profit by Year & Group (New vs Returning)
    for grp, sub in yearly_tidy.groupby("Group"):
        axes[0].plot(sub["OrderYear"], sub["TotalProfit"], marker="o", label=grp)
    axes[0].set_title("Total Profit by Year: New vs Returning")
    axes[0].set_ylabel("Total Profit")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Bottom chart: Avg Profit per Customer by Year (signals unit economics)
    for grp, sub in yearly_tidy.groupby("Group"):
        axes[1].plot(sub["OrderYear"], sub["AvgProfit_perCustomer"], marker="o", label=grp)
    axes[1].set_title("Avg Profit per Customer by Year")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Avg Profit / Customer")
    axes[1].grid(True, alpha=0.3)

    # Improve layout and render
    plt.tight_layout()
    plt.show()


# =============================================================================
# B) plot_retention_heatmap: Renders the cohort retention matrix as a heatmap (% active).
#    Where the heatmap has the following structure:
#    • Y-axis (rows): Cohorts (YYYY-MM of first purchase)
#    • X-axis (cols): Months since first purchase (CohortIndex)
#    • Cell value:    Retention rate (%) for that cohort at that month index
# =============================================================================
def plot_retention_heatmap(retention_matrix):
    # Create a canvas sized to the number of cohorts
    fig, ax = plt.subplots(figsize=(10, max(4, len(retention_matrix)*0.4)))

    # Convert retention (0–1) to percentage scale (0–100) for display
    mat = (retention_matrix.values * 100.0)

    # Draw heatmap (no seaborn to avoid extra deps)
    im = ax.imshow(mat, aspect="auto")

    # Titles and axis labels
    ax.set_title("Cohort Retention Heatmap (% active)")
    ax.set_ylabel("Cohort (YYYY-MM)")
    ax.set_xlabel("Months Since First Purchase")

    # Ticks and tick labels
    ax.set_yticks(range(len(retention_matrix.index)))
    ax.set_yticklabels([str(p) for p in retention_matrix.index])
    ax.set_xticks(range(len(retention_matrix.columns)))
    ax.set_xticklabels(retention_matrix.columns.astype(int))

    # Optional cell annotations for smaller matrices (keeps large ones readable)
    if mat.shape[0]*mat.shape[1] <= 400:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i,j]:.0f}%", ha="center", va="center",
                        fontsize=8, color="white")

    # Colorbar for value reference
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Retention (%)")

    # Layout polish and render
    plt.tight_layout()
    plt.show()


# =============================================================================
# C) plot_retention_decay_curve: Plots the average retention (across cohorts)
# for each month index. Where it shows:
#    -> The “typical” retention decay pattern of your customer base.
#    -> Steeper early drop = on-boarding/product-fit issues; flatter tail = loyalty.
# =============================================================================
def plot_retention_decay_curve(decay_curve):
    # Single-line chart: x = months since first purchase, y = avg retention (%)
    plt.figure(figsize=(8, 4))
    plt.plot(decay_curve.index, decay_curve.values * 100, marker="o")
    plt.title("Average Retention Decay over Time")
    plt.xlabel("Months Since First Purchase")
    plt.ylabel("Average Retention (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# D) print_cohort_insight: Prints a compact, readable executive summary of cohort health.
# =============================================================================
def print_cohort_insight(summary_metrics, lifetime_by_cohort):
    # Average “active months” across cohorts (area under the curve → months)
    avg_lifetime = lifetime_by_cohort.mean()

    # Header
    print("\n" + "="*70)
    print("COHORT ANALYSIS — EXECUTIVE SUMMARY")
    print("="*70)

    # Print 3m/6m/12m average retention windows (if available)
    for k, v in summary_metrics.items():
        if not np.isnan(v):
            print(f"• {k.replace('_',' ').title()}: {v*100:5.1f}%")

    # Core lifetime signal (interpretable “months”)
    print(f"• Average Customer Lifetime: {avg_lifetime:4.2f} months")
    print("------------------------------------------------------")

    # Best cohort by lifetime (robust with try/except)
    try:
        best = lifetime_by_cohort.idxmax()
        val  = lifetime_by_cohort.max()
        print(f"• Max-Lifetime Cohort: {best} ({val:.2f} months)")
    except Exception:
        pass

    # Footer
    print("="*70)


################################################################################
# iii) End-to-End Functions of Cohort Analysis Runner                          #
################################################################################

# =============================================================================
# A) Calculate all corresponding data needed for Cohort Analysis
# =============================================================================
yearly_tidy, yearly_pivot = new_vs_returning_yearly(df,
                                                    "customer_id",
                                                    "order_date",
                                                    "sales",
                                                    "profit"
                                                    )

retention_matrix = cohort_retention(df, "customer_id", "order_date")
summary_metrics, lifetime_by_cohort, decay_curve = summarize_cohort_results(retention_matrix)

# =============================================================================
# B) Plot and Display corresponding Results
# =============================================================================

plot_yearly_new_vs_returning(yearly_tidy)
plot_retention_heatmap(retention_matrix)
plot_retention_decay_curve(decay_curve)
print_cohort_insight(summary_metrics, lifetime_by_cohort)


########################################################################################################################
######################## 2) Comparative Value Base — LTV(H) & Value Concentration (Pareto by CohortYear) ###############
########################################################################################################################
# Goal:
#   • Provide a standardized, apples-to-apples basis for comparing customer economic potential.
#   • Measure both the magnitude (Observed LTV over H months) and the distribution (Pareto concentration) of value.
# Why:
#   • LTV(H) converts behavior into comparable profit within a fixed horizon.
#   • Pareto (within CohortYear) shows how value concentrates inside each “generation” of customers, avoiding seasonality bias.
# How to read:
#   • LTV(H) by Cohort → which acquisition months/years create more value in H months.
#   • Pareto by CohortYear → whether each generation is whale-driven or broadly distributed.
#   • Overlay (LTV vs Historical) → does future concentration intensify or diffuse relative to that cohort’s history?


################################################################################
# i) Build the time-aligned panel and compute Observed LTV(H)                  #
#    [make_customer_month_panel ; compute_observed_ltv ;                       #
#     plot_ltv_by_cohort ; plot_avg_profit_by_monthindex]                                           #
################################################################################

# =============================================================================
# A) make_customer_month_panel: Convert transactions into (customer × month) cashflows
# Why:
#   • Aligns all customers on a common “relationship age” axis (MonthIndex).
#   • Forms the basis for Observed LTV(H) and month-by-month diagnostics.
# What it computes:
#   • It creates and returns the table "panel", that is the foundation for all
#     calculation of this section. More specifically it contains for each customer
#     a set of rows, where each row corresponds to a month within the cohort
#     where the customer has placed any orders. Where each individual row
#     contains:
#   - YearMonth (Period[M]), Cohort (first purchase month), MonthIndex = 0,1,2,...
#   - Aggregated Orders/Sales/Profit per (customer, month). Here is a small
#   - numerical example of the table:
#     | customer_id | Cohort  | YearMonth | MonthIndex | Orders | Sales | Profit | IsActive |
#     | ----------- | ------- | --------- | ---------- | ------ | ----- | ------ | -------- |
#     | C1          | 2021-01 | 2021-01   | 0          | 1      | 100   | 20     | 1        |
#     | C1          | 2021-01 | 2021-03   | 2          | 2      | 250   | 50     | 1        |
#     | C2          | 2021-01 | 2021-01   | 0          | 1      | 150   | 30     | 1        |
#     | C2          | 2021-01 | 2021-02   | 1          | 1      | 120   | 25     | 1        |
# =============================================================================
def make_customer_month_panel(df,
                              cust_col="customer_id",
                              date_col="order_date",
                              sales_col="sales",
                              profit_col="profit"):
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d["YearMonth"] = d[date_col].dt.to_period("M")

    # First purchase per customer → defines the Cohort (birth month)
    firsts = (d.groupby(cust_col, as_index=False)[date_col]
                .min()
                .rename(columns={date_col: "FirstPurchaseDate"}))
    d = d.merge(firsts, on=cust_col, how="left")
    d["Cohort"] = d["FirstPurchaseDate"].dt.to_period("M")

    # MonthIndex = months since first purchase (relationship age)
    d["MonthIndex"] = (d["YearMonth"].view("i8") - d["Cohort"].view("i8")).astype(int)

    # Aggregate to (customer × month)
    # --- Aggregation logic (adaptive to dataset structure)
    # The following line dynamically selects how to count monthly "Orders":
    #   • If an 'order_id' column exists → count unique order_ids per customer-month.
    #     This ensures we don’t double-count multiple items within the same order.
    #   • If 'order_id' is missing → fall back to counting the number of rows
    #     (each transaction line = one order occurrence).
    # This single-line conditional avoids writing two separate groupby blocks.
    orders_agg = ("order_id", "nunique") if "order_id" in d.columns else (date_col, "count")
    # Aggregate monthly performance per customer (aligned by Cohort & MonthIndex)
    panel = (d.groupby([cust_col, "Cohort", "YearMonth", "MonthIndex"], as_index=False)
               .agg(Orders=orders_agg,
                    Sales=(sales_col, "sum"),
                    Profit=(profit_col, "sum")))
    # Mark each (customer, month) as active if at least one order exists
    panel["IsActive"] = (panel["Orders"] > 0).astype(int)
    return panel


# =============================================================================
# B) compute_observed_ltv: Observed Lifetime Value over horizon H (months)
# Why:
#   • Translates behavior into comparable profit over a fixed window (e.g., 12m).
# What it computes:
#   • LTV_H per customer = sum of monthly Profit for MonthIndex 0..H-1 (optionally NPV).
#   • Cohort-level summary (mean/median LTV_H and customer counts).
# Notes:
#   • CAC is intentionally excluded (no marketing-cost inputs in this dataset).
#   • For NPV, pass an annual discount rate; otherwise leave None.
# =============================================================================
def compute_observed_ltv(panel, H=12, discount_rate_annual=None):
    p = panel.copy()
    # Keep only records within the observed lifetime window (MonthIndex 0..H−1)
    pH = p[p["MonthIndex"].between(0, H-1)].copy()

    # Optional NPV discounting (annual → monthly)
    if discount_rate_annual is not None and discount_rate_annual > 0:
        r_m = (1 + discount_rate_annual)**(1/12) - 1
        pH["DiscFactor"] = (1 + r_m) ** (-pH["MonthIndex"])
        pH["ProfitAdj"] = pH["Profit"] * pH["DiscFactor"]
    else:
        pH["ProfitAdj"] = pH["Profit"]

    # LTV_H per customer
    ltv_cust = (pH.groupby(["customer_id", "Cohort"], as_index=False)["ProfitAdj"].sum()
                  .rename(columns={"ProfitAdj": "LTV_H"}))

    # Cohort-level summary (manager view)
    cohort_ltv = (ltv_cust.groupby("Cohort")
                  .agg(Customers=("customer_id", "nunique"),
                       Avg_LTV_H=("LTV_H", "mean"),
                       Med_LTV_H=("LTV_H", "median"))
                  .reset_index())
    return ltv_cust, cohort_ltv


# =============================================================================
# C) plot_ltv_by_cohort: Mean Observed LTV(H) per Cohort (YYYY-MM)
# Read:
#   • Upward trend → improved onboarding/mix/retention.
#   • Downward trend → check MonthIndex 0→1 drop, discounts policy, product mix.
# =============================================================================
def plot_ltv_by_cohort(cohort_ltv):
    plt.figure(figsize=(9, 4))
    plt.plot(cohort_ltv["Cohort"].astype(str), cohort_ltv["Avg_LTV_H"], marker="o")
    plt.title("Observed LTV(H) by Cohort")
    plt.xlabel("Cohort (YYYY-MM)")
    plt.ylabel("Average LTV(H)")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# D) plot_avg_profit_by_monthindex: Average monthly profit vs relationship age
# Read:
#   • Sharp drop after MonthIndex 0 → fix time-to-second-order.
#   • Flat/low tail → weak loyalty mechanics; explore frequency & bundles.
# =============================================================================
def plot_avg_profit_by_monthindex(panel, max_H=12):
    p = panel.copy()
    p = p[p["MonthIndex"].between(0, max_H-1)]
    curve = (p.groupby("MonthIndex")["Profit"].mean()
               .reindex(range(0, max_H), fill_value=0.0))

    plt.figure(figsize=(8, 4))
    plt.plot(curve.index, curve.values, marker="o")
    plt.title(f"Average Monthly Profit by MonthIndex (0..{max_H-1})")
    plt.xlabel("MonthIndex (relationship age)")
    plt.ylabel("Avg Profit per Customer-Month")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


################################################################################
# ii) Concentration diagnostics (Pareto) BY CohortYear — completeness-aware    #
#    [get_customer_first_purchase_year ; mark_complete_cohort_years ;          #
#     ltv_pareto_by_year ; historical_profit_pareto_by_year ;                  #
#     plot_ltv_pareto_by_year ; plot_hist_pareto_by_year ;                     #
#     plot_overlay_pareto_by_year]                                             #
################################################################################

# =============================================================================
# A) get_customer_first_purchase_year: Map each customer to CohortYear (YYYY)
# =============================================================================
def get_customer_first_purchase_year(df,
                                     cust_col="customer_id",
                                     date_col="order_date"):
    d = df[[cust_col, date_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col])
    firsts = (d.groupby(cust_col, as_index=False)[date_col]
                .min()
                .rename(columns={date_col: "FirstPurchaseDate"}))
    firsts["CohortYear"] = firsts["FirstPurchaseDate"].dt.year.astype(int)
    return firsts[[cust_col, "CohortYear"]]


# =============================================================================
# B) mark_complete_cohort_years: Flag CohortYears as COMPLETE/PARTIAL for H
# Logic:
#   • Year Y is COMPLETE if the dataset’s last YearMonth ≥ Period(f"{Y}-12") + (H-1).
#   • Else PARTIAL (Dec starters of Y lack full H months).
# =============================================================================
def get_data_last_period(df, date_col="order_date"):
    d = df[[date_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col])
    return d[date_col].dt.to_period("M").max()

def mark_complete_cohort_years(df, H=12, date_col="order_date"):
    last_p = get_data_last_period(df, date_col=date_col)
    years = pd.to_datetime(df[date_col]).dt.year.unique()
    flags = []
    for y in sorted(years):
        latest_start = pd.Period(f"{int(y)}-12", freq="M")
        needed = latest_start + (H - 1)
        is_complete = (last_p is not pd.NaT) and (last_p >= needed)
        flags.append({"CohortYear": int(y), "is_complete_H": bool(is_complete),
                      "last_period": last_p, "needed_until": needed})
    return pd.DataFrame(flags)


# =============================================================================
# C) ltv_pareto_by_year: Pareto-style table of LTV_H within each CohortYear
# Output: CohortYear | customer_id | LTV_H | CumLTV | CumLTVPct | CustomerPct
# =============================================================================
def ltv_pareto_by_year(ltv_cust, cust_year_map):
    # Keep only customer_id + LTV_H and merge with their CohortYear
    t = (ltv_cust[["customer_id", "LTV_H"]]
         .merge(cust_year_map, on="customer_id", how="left"))

    # Clean any infinite or missing LTV/CohortYear values
    t = t.replace([np.inf, -np.inf], np.nan).dropna(subset=["LTV_H", "CohortYear"])

    out = []  # will store one DataFrame per CohortYear

    # Loop through each cohort year (yr) and its subset of customers (sub)
    for yr, sub in t.groupby("CohortYear"):
        g = sub[["customer_id", "LTV_H"]].copy()  # copy relevant columns
        g = g.sort_values("LTV_H", ascending=False).reset_index(drop=True)  # sort by LTV descending

        g["CumLTV"] = g["LTV_H"].cumsum()  # cumulative LTV per cohort
        total = float(g["LTV_H"].sum())  # total LTV for normalization
        denom = total if total != 0.0 else 1.0  # avoid division by zero
        g["CumLTVPct"] = 100.0 * g["CumLTV"] / denom  # cumulative % of total LTV

        n = len(g)
        g["CustomerPct"] = 100.0 * (np.arange(1, n + 1) / n) if n > 0 else 0.0  # cumulative % of customers

        g.insert(0, "CohortYear", int(yr))  # insert CohortYear as first column
        out.append(g)  # save this cohort’s table into list

    # Combine all yearly results into one DataFrame, or return empty one if none exist
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["CohortYear", "customer_id", "LTV_H", "CumLTV", "CumLTVPct", "CustomerPct"]
    )


# =============================================================================
# D) historical_profit_pareto_by_year: Total-to-date profit Pareto by CohortYear
# Output: CohortYear | customer_id | TotalProfit | CumProfit | CumProfitPct | CustomerPct
# =============================================================================
def historical_profit_pareto_by_year(df,
                                     cust_year_map,
                                     cust_col="customer_id",
                                     profit_col="profit"):
    # Aggregate total (historical) profit per customer across all time
    cp = (df.groupby(cust_col, as_index=False)[profit_col].sum()
          .rename(columns={profit_col: "TotalProfit"}))

    # Attach CohortYear to each customer via mapping table
    t = cp.merge(cust_year_map, on=cust_col, how="left")

    # Clean invalid values and replace missing profit with 0
    t = t.replace([np.inf, -np.inf], np.nan).fillna({"TotalProfit": 0.0})
    t = t.dropna(subset=["CohortYear"])  # drop customers with unknown cohort

    out = []  # collect one table per CohortYear

    # Iterate over each CohortYear and its subset of customers
    for yr, sub in t.groupby("CohortYear"):
        g = sub[[cust_col, "TotalProfit"]].copy()  # copy relevant fields
        g = g.sort_values("TotalProfit", ascending=False).reset_index(drop=True)  # sort by profit descending

        g["CumProfit"] = g["TotalProfit"].cumsum()  # cumulative profit within cohort
        total = float(g["TotalProfit"].sum())  # total profit for normalization
        denom = total if total != 0.0 else 1.0  # avoid division by zero
        g["CumProfitPct"] = 100.0 * g["CumProfit"] / denom  # cumulative % of total profit

        n = len(g)
        g["CustomerPct"] = 100.0 * (np.arange(1, n + 1) / n) if n > 0 else 0.0  # cumulative % of customers

        g.insert(0, "CohortYear", int(yr))  # insert year as first column
        out.append(g)  # store this cohort’s result

    # Concatenate all cohort-year tables or return empty DataFrame if none
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["CohortYear", "customer_id", "TotalProfit", "CumProfit", "CumProfitPct", "CustomerPct"]
    )


# =============================================================================
# I) Plots — completeness-aware labels (“(partial)” shown where applicable)
# =============================================================================
def plot_ltv_pareto_by_year(ltv_par_year, completeness_df, H, years_to_show=None, max_years=4):
    d = ltv_par_year.copy()
    uniq_years = sorted(d["CohortYear"].unique())
    if years_to_show is None:
        years_to_show = uniq_years[-max_years:]
    comp = completeness_df.set_index("CohortYear") if completeness_df is not None else None

    plt.figure(figsize=(9, 5))
    for yr in years_to_show:
        sub = d[d["CohortYear"] == yr]
        if sub.empty:
            continue
        label = str(yr)
        if comp is not None and yr in comp.index and (not comp.loc[yr, "is_complete_H"]):
            label = f"{yr} (partial)"
        plt.plot(sub["CustomerPct"], sub["CumLTVPct"], marker="o", label=label)

    plt.axhline(80, linestyle="--", alpha=0.4)
    plt.axvline(20, linestyle="--", alpha=0.4)
    plt.title(f"LTV(H={H}) Pareto by CohortYear")
    plt.xlabel("% Customers (sorted within cohort)")
    plt.ylabel("Cumulative % of LTV(H)")
    plt.legend(title="CohortYear")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_hist_pareto_by_year(hist_par_year, completeness_df, H, years_to_show=None, max_years=4):
    d = hist_par_year.copy()
    uniq_years = sorted(d["CohortYear"].unique())
    if years_to_show is None:
        years_to_show = uniq_years[-max_years:]
    comp = completeness_df.set_index("CohortYear") if completeness_df is not None else None

    plt.figure(figsize=(9, 5))
    for yr in years_to_show:
        sub = d[d["CohortYear"] == yr]
        if sub.empty:
            continue
        label = str(yr)
        if comp is not None and yr in comp.index and (not comp.loc[yr, "is_complete_H"]):
            label = f"{yr} (partial)"
        plt.plot(sub["CustomerPct"], sub["CumProfitPct"], marker="o", label=label)

    plt.axhline(80, linestyle="--", alpha=0.4)
    plt.axvline(20, linestyle="--", alpha=0.4)
    plt.title("Historical Profit Pareto by CohortYear")
    plt.xlabel("% Customers (sorted within cohort)")
    plt.ylabel("Cumulative % of Historical Profit")
    plt.legend(title="CohortYear")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_overlay_pareto_by_year(ltv_par_year, hist_par_year, completeness_df, H,
                                years_to_show=None, max_years=3):
    yrs_ltv = set(ltv_par_year["CohortYear"].unique())
    yrs_hist = set(hist_par_year["CohortYear"].unique())
    common_years = sorted(list(yrs_ltv.intersection(yrs_hist)))
    if not common_years:
        return
    if years_to_show is None:
        years_to_show = common_years[-max_years:]
    comp = completeness_df.set_index("CohortYear") if completeness_df is not None else None

    n = len(years_to_show)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, yr in zip(axes, years_to_show):
        lt = ltv_par_year[ltv_par_year["CohortYear"] == yr]
        ht = hist_par_year[hist_par_year["CohortYear"] == yr]
        if lt.empty or ht.empty:
            ax.set_title(f"{yr} (insufficient data)")
            continue

        ax.plot(lt["CustomerPct"], lt["CumLTVPct"], marker="o", label="LTV(H)")
        ax.plot(ht["CustomerPct"], ht["CumProfitPct"], marker="o", label="Historical")
        ax.axhline(80, linestyle="--", alpha=0.4)
        ax.axvline(20, linestyle="--", alpha=0.4)

        title = f"CohortYear {yr}"
        if comp is not None and yr in comp.index and (not comp.loc[yr, "is_complete_H"]):
            title += " (partial)"
        ax.set_title(title)

        ax.set_xlabel("% Customers")
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("Cumulative %")
        ax.legend()

    plt.tight_layout()
    plt.show()


################################################################################
# iii) End-to-End Runner for Section 2                                         #
################################################################################
# Steps:
#   1) Build panel and compute LTV(H)
#   2) Plot LTV(H) by Cohort and Avg Profit vs MonthIndex
#   3) Build CohortYear mappings + completeness flags
#   4) Build Pareto tables and visualize (LTV vs Historical) by CohortYear

# 1) Panel + LTV(H)
panel = make_customer_month_panel(df,
                                  cust_col="customer_id",
                                  date_col="order_date",
                                  sales_col="sales",
                                  profit_col="profit")

H = 12  # choose 12 or 24 according to data depth
ltv_cust, cohort_ltv = compute_observed_ltv(panel, H=H, discount_rate_annual=None)

# 2) Manager plots (value trajectory + age curve)
plot_ltv_by_cohort(cohort_ltv)
plot_avg_profit_by_monthindex(panel, max_H=H)

# 3) CohortYear mapping + completeness flags
cust_year_map  = get_customer_first_purchase_year(df, cust_col="customer_id", date_col="order_date")
complete_flags = mark_complete_cohort_years(df, H=H, date_col="order_date")

# 4) Pareto tables + visual comparisons
ltv_par_year  = ltv_pareto_by_year(ltv_cust, cust_year_map)
hist_par_year = historical_profit_pareto_by_year(df, cust_year_map,
                                                 cust_col="customer_id",
                                                 profit_col="profit")

plot_ltv_pareto_by_year(ltv_par_year, complete_flags, H, years_to_show=None, max_years=4)
plot_hist_pareto_by_year(hist_par_year, complete_flags, H, years_to_show=None, max_years=4)
plot_overlay_pareto_by_year(ltv_par_year, hist_par_year, complete_flags, H,
                            years_to_show=None, max_years=3)





########################################################################################################################
###############################~4) Funnel Analysis — Behavioral transformation~#########################################
########################################################################################################################

################################################################################
# i) Functions to build stages, summarize funnel KPIs, and visualize:          #
#    [build_funnel_stages ; funnel_counts_rates ; plot_funnel_counts ;         #
#     plot_funnel_rates ; stage_transition_matrix]                             #
################################################################################

# =============================================================================
# A) build_funnel_stages: Assigns each customer to a lifecycle stage based on
#    purchase behavior (order count & second-order timing) and value (LTV_H).
# Why:
#   • Translates raw behavior into interpretable stages (activation → loyalty).
#   • Connects behavior (orders) with value (LTV_H) to identify High-Value cohorts.
# What it computes:
#   • TotalOrders per customer, SecondOrderMonth (MonthIndex of 2nd order).
#   • Cohort-level thresholds (median & p-high percentile of LTV_H).
#   • Stage label per customer:
#       S1_FirstPurchase   : TotalOrders == 1
#       S2_SecondPurchase  : TotalOrders >= 2 and 2nd order within N months
#       S3_Active          : TotalOrders >= 3
#       S4_Loyal           : S3 and LTV_H >= cohort median
#       S5_HighValue       : S3 and LTV_H >= cohort p-high (e.g., 90th pct)
# Notes:
#   • Priority of assignment is top-down (S5 → S4 → S3 → S2 → S1).
# =============================================================================
def build_funnel_stages(panel,
                        ltv_cust,
                        second_order_max_months=6,
                        high_value_percentile=0.90):
    p = panel.copy()

    # --- Per-customer totals (orders and relationship span)
    cust_totals = (p.groupby("customer_id")
                     .agg(TotalOrders=("Orders", "sum"),
                          MaxMonth=("MonthIndex", "max"))
                     .reset_index())

    # --- MonthIndex of the second order (if exists)
    #     We define "good onboarding" when 2nd order happens within N months.
    second_orders = (p[p["Orders"] > 0]
                     .sort_values(["customer_id", "MonthIndex"])
                     .groupby("customer_id")
                     .apply(lambda g: g["MonthIndex"].iloc[1] if len(g) >= 2 else np.nan)
                     .reset_index(name="SecondOrderMonth"))

    cust_totals = cust_totals.merge(second_orders, on="customer_id", how="left")

    # --- Join LTV_H and Cohort (from ltv_cust) to apply value thresholds by cohort
    cust_val = ltv_cust[["customer_id", "Cohort", "LTV_H"]].copy()
    cust_totals = cust_totals.merge(cust_val, on="customer_id", how="left")

    # --- Cohort-level thresholds for Loyal (median) and High-Value (percentile)
    cohort_median = (ltv_cust.groupby("Cohort")["LTV_H"].median()
                     .rename("CohortMedLTV_H")).reset_index()
    cohort_phigh  = (ltv_cust.groupby("Cohort")["LTV_H"].quantile(high_value_percentile)
                     .rename(f"CohortP{int(high_value_percentile*100)}LTV_H")).reset_index()

    cust_totals = cust_totals.merge(cohort_median, on="Cohort", how="left")
    cust_totals = cust_totals.merge(cohort_phigh,  on="Cohort", how="left")

    # --- Stage rules (priority order: HighValue → Loyal → Active → SecondPurchase → FirstPurchase)
    cond_high   = (cust_totals["TotalOrders"] >= 3) & (cust_totals["LTV_H"] >= cust_totals[cohort_phigh.columns[-1]])
    cond_loyal  = (cust_totals["TotalOrders"] >= 3) & (cust_totals["LTV_H"] >= cust_totals["CohortMedLTV_H"])
    cond_active = (cust_totals["TotalOrders"] >= 3)
    cond_s2     = (cust_totals["TotalOrders"] >= 2) & (cust_totals["SecondOrderMonth"] <= second_order_max_months)
    cond_s1     = (cust_totals["TotalOrders"] == 1)

    cust_totals["Stage"] = np.select(
        [cond_high,          cond_loyal,       cond_active,         cond_s2,              cond_s1],
        ["S5_HighValue",     "S4_Loyal",       "S3_Active",         "S2_SecondPurchase",  "S1_FirstPurchase"],
        default="S0_Prospect"  # not expected for buyers-only datasets
    )

    # Keep tidy output
    cols = ["customer_id", "Cohort", "TotalOrders", "SecondOrderMonth", "LTV_H", "Stage"]
    return cust_totals[cols]


# =============================================================================
# B) funnel_counts_rates: Summarizes funnel population and conversion ladder.
# Why:
#   • Provides a clear snapshot of how customers distribute across stages
#     (S1 → S5) and how many reach each milestone of engagement or loyalty.
#   • Avoids misleading % > 100 by using all purchasers as the reference base.
#
# What it computes:
#   • Counts of customers per lifecycle stage (S1–S5).
#   • Overall percentage of all purchasers in each final stage.
#   • Stepwise “conversion ladder” — % of all purchasers who reached each
#     key behavioral milestone (2nd order, ≥3 orders, loyal, high-value).
#
# How to read:
#   • The first table (“stages_table”) shows the final composition of the base.
#   • The second table (“ladder”) shows progressive retention/conversion rates.
#   • Example: If only 40% reach 2+ orders but 15% reach Loyal (S4+),
#     onboarding likely underperforms, but loyalty mechanics convert well.
# =============================================================================
def funnel_counts_rates(stage_df):
    # Canonical order of lifecycle stages
    order = ["S1_FirstPurchase", "S2_SecondPurchase", "S3_Active", "S4_Loyal", "S5_HighValue"]

    # --- Raw counts of customers per stage
    counts_raw = stage_df["Stage"].value_counts()
    counts = pd.Series({k: int(counts_raw.get(k, 0)) for k in order})

    # --- Define base population: all purchasers with at least one order
    base_all = int(counts.sum()) if counts.sum() > 0 else 1

    # --- Overall percentage (each final stage as % of all purchasers)
    overall_pct = (counts / base_all * 100).round(1)

    # --- Stepwise milestones (monotonic ladder)
    reached_2nd = counts["S2_SecondPurchase"] + counts["S3_Active"] + counts["S4_Loyal"] + counts["S5_HighValue"]
    reached_3rd = counts["S3_Active"] + counts["S4_Loyal"] + counts["S5_HighValue"]
    loyal       = counts["S4_Loyal"] + counts["S5_HighValue"]
    high_value  = counts["S5_HighValue"]

    # --- Build ladder DataFrame
    ladder = pd.DataFrame({
        "Milestone": [
            "Any purchase (S1–S5)",
            "2nd order within N months (S2+)",
            "≥3 orders (S3+)",
            "Loyal (≥ median LTV, S4+)",
            "HighValue (≥ p90 LTV, S5)"
        ],
        "Customers": [base_all, reached_2nd, reached_3rd, loyal, high_value],
    })
    ladder["Rate_of_All_%"] = (ladder["Customers"] / base_all * 100).round(1)

    # --- Build stage summary table
    stages_table = pd.DataFrame({
        "Stage": counts.index,
        "Customers": counts.values,
        "Rate_of_All_%": overall_pct.values
    })

    return stages_table, ladder


def plot_funnel_counts_stages(stages_table):
    """Bar chart of customers per final stage (S1..S5)."""
    plt.figure(figsize=(9, 5))
    plt.bar(stages_table["Stage"], stages_table["Customers"])
    plt.title("Customer Funnel — Counts per Final Stage (S1..S5)")
    plt.xlabel("Stage")
    plt.ylabel("# Customers")
    plt.xticks(rotation=15)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_funnel_ladder_rates(ladder):
    """Line plot of conversion ladder rates (as % of all purchasers)."""
    plt.figure(figsize=(10, 4))
    plt.plot(ladder["Milestone"], ladder["Rate_of_All_%"], marker="o")
    plt.title("Customer Funnel — Conversion Ladder (% of all purchasers)")
    plt.xlabel("Milestone")
    plt.ylabel("% of All Purchasers")
    plt.xticks(rotation=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# E) stage_transition_matrix: Early→Horizon transition diagnostic.
# Why:
#   • Tests whether early behavior (within a small MonthIndex window) predicts
#     end-of-horizon outcomes (S3/S4/S5).
# What it computes:
#   • Crosstab of EarlyStage vs FinalStage, normalized by row.
# How to read:
#   • If Early “2+ orders” strongly maps to S4/S5, invest in onboarding nudges.
# Notes:
#   • EarlyStage here is simplified: 0–1 orders → S1, 2+ orders → S2.
# =============================================================================
def stage_transition_matrix(panel, stage_df, early_max_month_index=1):
    # Early activity within MonthIndex <= early_max_month_index
    early = (panel[panel["MonthIndex"] <= early_max_month_index]
             .groupby("customer_id")["Orders"].sum().reset_index())
    early["EarlyStage"] = np.where(early["Orders"] >= 2, "Early_2plus", "Early_0to1")

    final = stage_df[["customer_id", "Stage"]].rename(columns={"Stage": "FinalStage"})
    trans = early.merge(final, on="customer_id", how="right")

    M = pd.crosstab(trans["EarlyStage"], trans["FinalStage"], normalize="index").round(3)
    return M


################################################################################
#                 ii) End-to-End Funnel Runner (build → summarize → plot)      #
################################################################################

# A) Build stage labels (behavior + value thresholds by cohort)
stages = build_funnel_stages(panel,
                             ltv_cust,
                             second_order_max_months=6,
                             high_value_percentile=0.90)

# B) Summarize funnel
#    • stages_table: final composition by Stage (S1..S5), % of all purchasers
#    • ladder:       progressive milestones (2nd order, ≥3 orders, Loyal, HighValue)
stages_table, ladder = funnel_counts_rates(stages)

print("\nFunnel — Final composition by Stage (of all purchasers):\n", stages_table)
print("\nFunnel — Conversion ladder (milestones as % of all purchasers):\n", ladder)

# C) Visualize funnel counts and rates
plot_funnel_counts_stages(stages_table)
plot_funnel_ladder_rates(ladder)

# D) Early→Horizon transition check (diagnostic)
T = stage_transition_matrix(panel, stages, early_max_month_index=1)
print("\nTransition matrix (Early activity → Final stage):\n", T)

# We need columns: customer_id, LTV
ltv_export = ltv_cust[['customer_id','LTV_H']].rename(columns={'LTV_H':'LTV'})
ltv_export.to_csv("C:/Users/kosta/Documents/DA/Projects/1_Super_store/ltv.csv", index=False)
print('✅ Saved artifacts/ltv.csv', ltv_export.shape)
