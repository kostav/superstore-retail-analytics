# =============================================================================
# SUPERSTORE — SECTION 0: DATA CLEANING AND PROCESSINGS
# =============================================================================
# This step takes the raw Superstore export (already screened in SQL for nulls,
# duplicates, and invalid ranges) and prepares it for analysis:
# • Load CSV into pandas.
# • Convert date fields to datetime and numeric fields to floats/ints, coercing
#   invalid entries to NaT/NaN so they don’t silently break analysis.
# • Run boundary checks (sales must be >0; profits can be negative but flag absurd values).
# • Normalize column names (lowercase, underscores) for consistency across code.
#
# Output: a clean DataFrame `df` with reliable dtypes and standardized schema.
# -----------------------------------------------------------------------------



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import pearsonr, spearmanr, norm
import seaborn as sns

pd.set_option('display.max_columns', None)   # show all columns
pd.set_option('display.width', None)         # don't wrap badly


# Load CSV
df = pd.read_csv(r"C:\Users\kosta\Documents\DA\Projects\1_Super_store\Sample - Superstore.csv", encoding='ISO-8859-1')

#1) Since we already checked via our SQL filtering that there are no duplicates or categorial/grammar issues we only need to proceed with
#  Data Type conversions in order to be able to proceed with data analysis(dates & numerics)

#1a) Dates
for c in ["order_date","ship_date"]:
    if c in df.columns: # safety check: only try to convert if that column exists in the dataframe (avoids errors if it doesn’t).
        df[c] = pd.to_datetime(df[c], errors="coerce")
        pd.to_datetime()
        #takes whatever’s in that column(strings like "01/01/2019") and converts to pandas’ datetime type.
        #errors = "coerce" → if a value can’t be converted (e.g."abc"), it becomes NaT (Not a Time, i.e.missing value for dates).

#1b) Numerics
for c in ["sales","profit","discount","quantity","postal_code"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        #If the column had "12.34"(string) or "1,000"(with comma), pandas tries to convert to float / int.
        #If a value can’t be parsed, it becomes NaN.

#1c) Numerical boundaries check (sales should be > 0; profit can be negative but check absurd values)
print("Rows with sales <= 0:", int((df['Sales'] <= 0).sum()))
print("Rows with |profit| > 100000:", int((df['Profit'].abs() > 100000).sum()))

#1d) Normalize columns names to drop capital letters etc and have all data names in a unified way
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(" ", "_")
      .str.replace("-", "_")
)

print("Dataset Column Names: ", df.columns.tolist())
df_clean = df.copy()
df_clean.to_pickle("clean_superstore.pkl")
