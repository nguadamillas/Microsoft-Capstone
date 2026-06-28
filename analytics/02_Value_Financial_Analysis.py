
# # Notebook 2 — Value & Financial Analysis
# **TED Procurement Intelligence | IE University × Microsoft**
# **Analytics + Semantic Layer | Data: January 2026**
#
# ----------------------------------------------------------------------
# ### What this notebook covers
# 0. Setup — Mount Drive & Load Data
# 1. Estimated value null audit — where are the gaps and does it bias financial analysis?
# 2. Awarded value coverage — how complete is `awarded_eur` vs `estimated`?
# 3. Total spend landscape — awarded value by country (notice-normalised)
# 4. Average contract size — mean & median awarded value per notice by country
# 5. Estimated vs awarded comparison — are buyers over- or under-estimating?
# 6. Value distribution — are a few mega-contracts driving all spend?
# 7. Spend by procurement type — Services vs Supplies vs Works in EUR
# 8. Top CPV categories by total awarded value
# 9. Largest individual contracts — the mega-deal leaderboard
# 10. Buyer concentration — how many buyers account for most of the spend?
# 11. Contract size segmentation — Micro / Small / Mid / Large breakdown
# 12. Dominant buyers per country — who controls each national market?
#
# > **Key rule from NB1:** All country-level financial metrics use notice-level aggregation
# > (`groupby notice_id` first, then `groupby country`). Romania has 14.31 lots per notice
# > in gold_awards — raw row sums would inflate its totals by ~14x.
# >
# > **Key constraint from NB1:** `estimated` is 57% null across gold_notices.
# > Financial analysis on awarded values uses gold_awards (lot-level, then collapsed to notice-level).
# > Estimated-vs-awarded comparisons only use the subset where both fields are present.


# ## 0. Setup — Mount Drive & Load Data

from google.colab import drive
drive.mount('/content/drive')

import os, zipfile

POSSIBLE_PATHS = [
    '/content/drive/Shareddrives/CAPSTONE PROJECT/up to date parquet/final_bronze_silver_gold_parquets.zip',
    '/content/drive/Shared drives/CAPSTONE PROJECT/up to date parquet/final_bronze_silver_gold_parquets.zip',
    '/content/drive/MyDrive/up to date parquet/final_bronze_silver_gold_parquets.zip',
]

zip_path = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p):
        zip_path = p
        print(f'Found zip at: {p}')
        break

if zip_path is None:
    print('Could not find zip. Listing Shared Drives:')
    base = '/content/drive/Shareddrives'
    if not os.path.exists(base):
        base = '/content/drive/Shared drives'
    for root, dirs, files in os.walk(base):
        depth = root.replace(base, '').count(os.sep)
        if depth < 4:
            print('  ' * depth + os.path.basename(root) + '/')
            for f in files:
                print('  ' * (depth+1) + f)

EXTRACT_PATH = '/content/parquets/'
os.makedirs(EXTRACT_PATH, exist_ok=True)

if zip_path and not os.listdir(EXTRACT_PATH):
    print('Unzipping... (this may take a minute)')
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(EXTRACT_PATH)
    print('Done!')
elif os.listdir(EXTRACT_PATH):
    print('Files already extracted.')
else:
    print('zip_path not set — fix path above and re-run.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.dpi': 130,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

C_BLUE   = '#2E6DA4'
C_TEAL   = '#2196A6'
C_GREEN  = '#2E8B57'
C_ORANGE = '#E07B39'
C_RED    = '#C0392B'
C_PURPLE = '#7B5EA7'
C_GREY   = '#7F8C8D'
C_PINK   = '#C0436A'
C_GOLD   = '#B8860B'

def fmt_eur(x, pos=None):
    if x >= 1e9:  return f'€{x/1e9:.1f}B'
    if x >= 1e6:  return f'€{x/1e6:.1f}M'
    if x >= 1e3:  return f'€{x/1e3:.0f}K'
    return f'€{x:.0f}'

parquet_files = {}
for root, dirs, files in os.walk(EXTRACT_PATH):
    for f in files:
        if f.endswith('.parquet'):
            parquet_files[f.replace('.parquet', '')] = os.path.join(root, f)

print(f'Found {len(parquet_files)} parquet files:')
for name in sorted(parquet_files): print(f'  {name}')

gold_notices       = pd.read_parquet(parquet_files['gold_notices'])
gold_opportunities = pd.read_parquet(parquet_files['gold_opportunities'])
gold_awards        = pd.read_parquet(parquet_files['gold_awards'])
gold_country_kpis  = pd.read_parquet(parquet_files['gold_country_kpis'])
gold_cpv_analysis  = pd.read_parquet(parquet_files['gold_cpv_analysis'])

for df in [gold_notices, gold_opportunities, gold_awards]:
    if 'pub_date' in df.columns:
        df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce')

jan_mask = (gold_notices['pub_date'] >= '2026-01-01') & (gold_notices['pub_date'] <= '2026-01-31')
gold_notices_jan = gold_notices[jan_mask].copy()

print(f'gold_notices (all):         {len(gold_notices):>7,} rows')
print(f'gold_notices (Jan 2026):    {len(gold_notices_jan):>7,} rows')
print(f'gold_opportunities:         {len(gold_opportunities):>7,} rows')
print(f'gold_awards (lot-level):    {len(gold_awards):>7,} rows')
print(f'gold_country_kpis:          {len(gold_country_kpis):>7,} rows')
print(f'gold_cpv_analysis:          {len(gold_cpv_analysis):>7,} rows')
print()
print('Columns in gold_awards:')
print(list(gold_awards.columns))
print()
print('Columns in gold_opportunities:')
print(list(gold_opportunities.columns))


# ----------------------------------------------------------------------
# ## 1. Estimated Value Null Audit
# > **Why this comes first:** NB1 found that `estimated` is 57% null across gold_notices.
# > Before doing any financial analysis, we need to understand *where* those nulls are.
# > If they are concentrated in specific countries or CPV categories, then financial
# > conclusions will only reflect the countries/categories that *do* report values —
# > which could be systematically misleading.
# >
# > **What we are checking:**
# > - Null rate of `estimated` per country (are some countries near 0% null? others near 100%?)
# > - Null rate of `awarded_eur` in gold_awards (a separate field — do completed contracts report values?)
# > - Null rate by procurement type (do Works contracts report values more than Services?)
# > - Overall conclusion: which subset of data is safe to use for financial analysis

print('=== OVERALL NULL RATES — KEY FINANCIAL FIELDS ===')
print()

n_est_null = gold_notices_jan['estimated'].isna().sum()
n_est_tot  = len(gold_notices_jan)
print(f'gold_notices (Jan 2026):')
print(f'  Total notices:                    {n_est_tot:>7,}')
print(f'  Nulls in `estimated`:             {n_est_null:>7,}  ({n_est_null/n_est_tot*100:.1f}%)')
print(f'  Notices WITH estimated value:     {n_est_tot - n_est_null:>7,}  ({(n_est_tot-n_est_null)/n_est_tot*100:.1f}%)')
print()

value_cols = ['contract_value', 'total_awarded', 'awarded_amount', 'lot_est', 'estimated']
print('gold_awards (lot-level) — coverage of value columns:')
print(f'{"Column":<20} {"Valid":>10} {"Coverage":>10} {"Null":>10} {"Max":>20}')
print('-' * 75)
for col in value_cols:
    if col in gold_awards.columns:
        n_valid = gold_awards[col].notna().sum()
        n_null  = gold_awards[col].isna().sum()
        pct     = n_valid / len(gold_awards) * 100
        mx      = gold_awards[col].max()
        print(f'{col:<20} {n_valid:>10,} {pct:>9.1f}% {n_null:>10,} {mx:>20,.0f}')
print()
print('→ Using contract_value as primary financial metric (93.3% coverage).')
print('→ Capping at €5B to remove likely data entry errors (max was €250B).')

notices_with_val = gold_awards.dropna(subset=['contract_value'])[
    gold_awards['contract_value'] > 0]['notice_id'].nunique()
notices_total    = gold_awards['notice_id'].nunique()
print(f'\nUnique notices with >= 1 valid contract_value: {notices_with_val:,} of {notices_total:,}'
      f' ({notices_with_val/notices_total*100:.1f}%)')

null_by_country = (gold_notices_jan
    .groupby('buyer_country')
    .agg(
        total_notices   = ('notice_id', 'count'),
        has_estimated   = ('estimated', lambda x: x.notna().sum())
    )
    .reset_index()
)
null_by_country['pct_with_value'] = null_by_country['has_estimated'] / null_by_country['total_notices'] * 100
null_by_country['pct_null']       = 100 - null_by_country['pct_with_value']

null_sig = null_by_country[null_by_country['total_notices'] >= 100].sort_values('pct_with_value', ascending=False)

print('Estimated value coverage by country (countries with >= 100 notices):')
print(f'{"Country":<10} {"Notices":>10} {"With Value":>12} {"% Coverage":>12} {"% Null":>10}')
print('-' * 58)
for _, row in null_sig.iterrows():
    bar = '█' * int(row['pct_with_value'] / 5)
    print(f'{row["buyer_country"]:<10} {int(row["total_notices"]):>10,} {int(row["has_estimated"]):>12,} {row["pct_with_value"]:>11.1f}% {row["pct_null"]:>9.1f}%  {bar}')


# > **Data quality note — Switzerland (CHE):**
# > Switzerland appears in the dataset with 752 notices but **0.0% estimated value coverage** —
# > the only country that never reports an estimated contract value.
# > This is not a pipeline error. Switzerland is not an EU member state and operates under
# > its own procurement law (BöB/LMP), which does not require estimated values to be published
# > in TED notices. Swiss contracts are included in volume counts but must be excluded from
# > any analysis that relies on the `estimated` field, including the savings analysis in Section 5.
# > Person 6's chatbot should be aware that Swiss contract values are systematically underreported.

top_countries = null_sig.sort_values('total_notices', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(14, 7))

countries = top_countries['buyer_country'].tolist()[::-1]
pct_ok    = top_countries['pct_with_value'].tolist()[::-1]
pct_null  = top_countries['pct_null'].tolist()[::-1]

bars1 = ax.barh(countries, pct_ok,   color=C_TEAL,  alpha=0.85, label='Has estimated value')
bars2 = ax.barh(countries, pct_null, left=pct_ok,   color=C_RED, alpha=0.4,  label='Null (no value reported)')

ax.axvline(50, color='black', linestyle='--', linewidth=1, alpha=0.5, label='50% threshold')
ax.set_xlabel('% of Notices')
ax.set_title('Estimated Value Coverage by Country — Top 20 by Notice Count | January 2026',
             fontweight='bold')
ax.set_xlim(0, 100)
ax.legend(loc='lower right')

for i, (ok, country) in enumerate(zip(pct_ok, countries)):
    ax.text(ok - 2, i, f'{ok:.0f}%', va='center', ha='right', fontsize=8,
            color='white', fontweight='bold')

plt.tight_layout()
plt.show()
print()
print('INTERPRETATION:')
print('Countries on the left (low coverage) will be underrepresented in financial analysis.')
print('Countries on the right (high coverage) are reliable sources of financial data.')

null_by_type = (gold_notices_jan
    .groupby('proc_type')
    .agg(
        total   = ('notice_id', 'count'),
        has_val = ('estimated', lambda x: x.notna().sum())
    )
    .reset_index()
)
null_by_type['pct_coverage'] = null_by_type['has_val'] / null_by_type['total'] * 100

print('Estimated value coverage by procurement type (gold_notices):')
print(null_by_type.sort_values('pct_coverage', ascending=False).to_string(index=False))
print()

null_aw_type = (gold_awards
    .groupby('proc_type')
    .agg(
        total_lots        = ('notice_id', 'count'),
        has_contract_val  = ('contract_value', lambda x: x.notna().sum())
    )
    .reset_index()
)
null_aw_type['pct_cv_coverage'] = null_aw_type['has_contract_val'] / null_aw_type['total_lots'] * 100
print('contract_value coverage by procurement type (gold_awards lots):')
print(null_aw_type.sort_values('pct_cv_coverage', ascending=False).to_string(index=False))

CAP = 5_000_000_000

gold_awards['cv_clean'] = gold_awards['contract_value'].clip(upper=CAP)

notice_awards = (gold_awards
    .groupby(['notice_id', 'buyer_country', 'proc_type', 'cpv_division', 'cpv_division_name'])
    .agg(
        awarded_eur_total = ('cv_clean',       'sum'),
        lot_count         = ('notice_id',      'count'),
        awarded_eur_max   = ('cv_clean',       'max'),
        tenders_count     = ('tenders_count',  'mean')
    )
    .reset_index()
)

notice_awards_clean = notice_awards[notice_awards['awarded_eur_total'] > 0].copy()

print('=== CLEAN FINANCIAL DATASET (notice-level) ===')
print(f'Total unique notices in gold_awards:       {notice_awards["notice_id"].nunique():>7,}')
print(f'Notices with contract_value_total > 0:     {len(notice_awards_clean):>7,}'
      f'  ({len(notice_awards_clean)/notice_awards["notice_id"].nunique()*100:.1f}%)')
print()
print('This notice-level table is the basis for all financial analysis below.')
print('Romania lot-inflation is eliminated — each row = one procurement notice.')
print()
total_spend = notice_awards_clean['awarded_eur_total'].sum()
print(f'Total awarded value across all notices: {fmt_eur(total_spend)}')
print(f'Mean awarded value per notice:          {fmt_eur(notice_awards_clean["awarded_eur_total"].mean())}')
print(f'Median awarded value per notice:        {fmt_eur(notice_awards_clean["awarded_eur_total"].median())}')
print(f'Max awarded value (single notice):      {fmt_eur(notice_awards_clean["awarded_eur_total"].max())}')
print(f'Min awarded value (single notice > 0):  {fmt_eur(notice_awards_clean["awarded_eur_total"].min())}')


# ----------------------------------------------------------------------
# ## 2. Awarded Value Coverage — How Complete Are Financial Records?
# > **KPI Definition — Awarded Value Coverage Rate:**
# > The percentage of award notices (CAN) in gold_awards that have a non-null, positive
# > `awarded_eur` value. A low coverage rate means financial totals only reflect a fraction
# > of actual market activity. This is a data quality KPI, not a business KPI.
# >
# > **Why this matters:** If Germany has 90% coverage but Bulgaria has 20% coverage,
# > Germany will dominate total spend figures not because it actually spends more per notice,
# > but because it reports more. The null audit makes this transparent.

aw_cov = (gold_awards
    .groupby('buyer_country')
    .agg(
        total_lots    = ('notice_id',       'count'),
        lots_with_val = ('contract_value',  lambda x: x.notna().sum()),
        total_eur     = ('cv_clean',        'sum')
    )
    .reset_index()
)
aw_cov['pct_coverage'] = aw_cov['lots_with_val'] / aw_cov['total_lots'] * 100
aw_cov = aw_cov[aw_cov['total_lots'] >= 50].sort_values('pct_coverage', ascending=False)

print('contract_value lot-level coverage by country (>= 50 lots):')
print(f'{"Country":<10} {"Total Lots":>12} {"Lots w/ Value":>14} {"Coverage %":>12} {"Total EUR":>15}')
print('-' * 68)
for _, row in aw_cov.iterrows():
    print(f'{row["buyer_country"]:<10} {int(row["total_lots"]):>12,} {int(row["lots_with_val"]):>14,}'
          f' {row["pct_coverage"]:>11.1f}% {fmt_eur(row["total_eur"]):>15}')


# ----------------------------------------------------------------------
# ## 3. Total Spend Landscape — Awarded Value by Country
# > **KPI Definition — Total Awarded Spend:**
# > The sum of `awarded_eur` across all award lots for a given country, collapsed to
# > notice-level first to avoid Romania's lot-inflation issue. Expressed in EUR.
# > This tells us: which countries have the highest total volume of contracted public spend
# > in the dataset.
# >
# > **Important caveat:** Countries with low awarded_eur coverage (Section 2) will appear
# > smaller here than they truly are. Total spend figures should always be read alongside
# > coverage rates.

spend_country = (notice_awards_clean
    .groupby('buyer_country')
    .agg(
        total_awarded_eur = ('awarded_eur_total', 'sum'),
        notice_count      = ('notice_id', 'count')
    )
    .reset_index()
    .sort_values('total_awarded_eur', ascending=False)
)
spend_country['pct_of_total'] = spend_country['total_awarded_eur'] / spend_country['total_awarded_eur'].sum() * 100
spend_country['cumulative_pct'] = spend_country['pct_of_total'].cumsum()

total_market = spend_country['total_awarded_eur'].sum()

print(f'Total awarded spend in dataset: {fmt_eur(total_market)}')
print()
print(f'{"Country":<10} {"Total Awarded":>16} {"% of Total":>12} {"Cumulative %":>14} {"Notices":>10}')
print('-' * 68)
for _, row in spend_country.head(20).iterrows():
    print(f'{row["buyer_country"]:<10} {fmt_eur(row["total_awarded_eur"]):>16}'
          f' {row["pct_of_total"]:>11.1f}% {row["cumulative_pct"]:>13.1f}%'
          f' {int(row["notice_count"]):>10,}')

top15_spend = spend_country.head(15)

fig, ax1 = plt.subplots(figsize=(14, 7))
ax2 = ax1.twinx()

bars = ax1.bar(
    top15_spend['buyer_country'],
    top15_spend['total_awarded_eur'],
    color=C_BLUE, alpha=0.85, edgecolor='white'
)
ax2.plot(
    top15_spend['buyer_country'],
    top15_spend['pct_of_total'].cumsum(),
    color=C_ORANGE, marker='o', linewidth=2, markersize=5, label='Cumulative %'
)
ax2.axhline(80, color=C_RED, linestyle='--', linewidth=1, alpha=0.6, label='80% threshold')

ax1.set_ylabel('Total Awarded Value (EUR)', color=C_BLUE)
ax2.set_ylabel('Cumulative % of Total Spend', color=C_ORANGE)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_eur))
ax2.set_ylim(0, 105)
ax1.set_title('Total Awarded Spend by Country — Top 15 | January 2026 (Notice-Normalised)',
              fontweight='bold')
ax2.legend(loc='center right')

for bar in bars:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h * 1.01,
             fmt_eur(h), ha='center', fontsize=8, rotation=45)

plt.tight_layout()
plt.show()
print()
print('NOTE: Full dataset including Hungary — see Section 3b for ex-Hungary view.')
n_80 = (spend_country['cumulative_pct'] <= 80).sum() + 1
print(f'80% of total awarded spend is concentrated in the top {n_80} countries.')
top_n = spend_country.head(n_80)['buyer_country'].tolist()
print(f'These are: {top_n}')


# ----------------------------------------------------------------------
# ## 3b. Framework Value Diagnostic — Investigating the Hungary Anomaly
# > Hungary accounts for 44% of total awarded spend from just 509 notices,
# > while Germany — the largest EU economy — shows only €5.5B from 2,525 notices.
# > Before accepting this at face value, we need to investigate whether these figures
# > reflect actual awarded contract values or framework agreement **ceiling values**.
# >
# > **What is a framework ceiling?** In EU public procurement, a buyer can set up a
# > framework agreement with a theoretical maximum value (e.g. 'up to €30B over 4 years
# > for medical equipment'). Some countries report this ceiling as the contract value
# > in their TED notices; others report only the actual transaction amounts. If Hungary
# > is reporting ceilings, its figures could be inflated by 10x–100x.
# >
# > **Our investigation approach:** We first tried to use the pipeline's own award flags
# > (`result_code` and `is_awarded`) to filter out non-awarded lots — the cleanest possible
# > solution if the data supported it. We then examined round-number patterns and
# > `contract_count` on all lots above €1B to gather additional evidence.

THRESHOLD = 1_000_000_000
big_lots  = gold_awards[gold_awards['cv_clean'] >= THRESHOLD].copy()

print(f'=== FRAMEWORK VALUE DIAGNOSTIC ===')
print(f'Lots with contract_value >= €1B: {len(big_lots):,}')
print()

print('--- Signal 1: Pipeline award flags on lots >= €1B ---')
if 'result_code' in big_lots.columns:
    print('result_code distribution:')
    print(big_lots['result_code'].value_counts().to_string())
    print()
if 'is_awarded' in big_lots.columns:
    print('is_awarded distribution:')
    print(big_lots['is_awarded'].value_counts(dropna=False).to_string())
    print()

print('--- Signal 2: Round-number pattern on lots >= €1B ---')
big_lots['is_round_billion'] = (big_lots['cv_clean'] % 1_000_000_000 == 0)
big_lots['is_round_500m']    = (big_lots['cv_clean'] % 500_000_000  == 0)
n_round_b = big_lots['is_round_billion'].sum()
n_round_5 = big_lots['is_round_500m'].sum()
print(f'Lots where value is exact multiple of €1B:    {n_round_b:>5,} of {len(big_lots):,} ({n_round_b/len(big_lots)*100:.1f}%)')
print(f'Lots where value is exact multiple of €500M:  {n_round_5:>5,} of {len(big_lots):,} ({n_round_5/len(big_lots)*100:.1f}%)')
print()
print('Most common exact values (top 15):')
print(big_lots['cv_clean'].value_counts().head(15).apply(fmt_eur).to_string())
print()

print('--- Signal 3: contract_count on lots >= €1B ---')
if 'contract_count' in big_lots.columns:
    print('contract_count distribution (0 or null = no actual contract):')
    print(big_lots['contract_count'].value_counts(dropna=False).head(10).to_string())
    n_zero_cc = (big_lots['contract_count'] == 0).sum()
    n_null_cc =  big_lots['contract_count'].isna().sum()
    print(f'\ncontract_count == 0:    {n_zero_cc:,}')
    print(f'contract_count is null: {n_null_cc:,}')
    print()

print('--- Signal 4: Big lots (>= €1B) by country ---')
by_country = (big_lots.groupby('buyer_country')
    .agg(n_big_lots=('cv_clean','count'),
         total_big_eur=('cv_clean','sum'),
         n_round=('is_round_billion','sum'))
    .reset_index()
    .sort_values('total_big_eur', ascending=False)
)
print(f'{"Country":<10} {"Big Lots":>10} {"Total EUR":>14} {"Round €1B":>12} {"% Round":>10}')
print('-' * 55)
for _, row in by_country.iterrows():
    pct_r = row['n_round']/row['n_big_lots']*100 if row['n_big_lots']>0 else 0
    print(f'{row["buyer_country"]:<10} {int(row["n_big_lots"]):>10,}'
          f' {fmt_eur(row["total_big_eur"]):>14}'
          f' {int(row["n_round"]):>12,} {pct_r:>9.1f}%')


# ### What the Diagnostic Tells Us — And Why We Exclude Hungary
#
# **Signal 1 — The pipeline flag approach did not work.**
# Every single lot above €1B has `result_code = selec-w` and `is_awarded = True`.
# The pipeline has marked all of them as genuine awarded contracts, which means we
# cannot use these flags to filter out framework ceilings from actual awards.
# This was the cleanest possible fix, and the data does not support it.
#
# **Signal 2 — Round numbers point to Hungary and Sweden as the main offenders.**
# 29 contracts worth exactly €5B each is not a coincidence — this is a strong signal
# of framework ceiling registration. Hungary has 36.4% of its big lots as exact
# multiples of €1B; Sweden 50%. However, France, Denmark, Romania and Poland
# show 0% round numbers on their big lots, suggesting those are genuine large contracts.
#
# **Signal 3 — contract_count does not help.**
# All 156 big lots show `contract_count >= 1`, meaning the pipeline recorded at least
# one actual contract execution. This does not conclusively prove the values are real —
# a framework can have one associated contract signed at a fraction of the ceiling —
# but it removes the clearest possible evidence of zero-transaction registrations.
#
# **Our conclusion — Why Hungary is excluded from spend analysis:**
# Hungary has 88 lots above €1B totalling €231.7B, representing 36.4% round-billion
# values and contributing 44% of the dataset's total spend from only 509 notices.
# For context, Germany contributes €5.5B from 2,525 notices — 23x more activity
# but 42x less spend. No plausible difference in contract sizes explains this gap.
# Hungary's procurement law permits framework agreements to be published with ceiling
# values that are set deliberately high to cover any future spend under the agreement.
# The combination of implausible scale, high round-number rate, and country-level
# context makes excluding Hungary the only defensible analytical choice.
#
# **Sweden** shows a similar pattern (50% round billions) but at a much smaller scale
# (€35B, 14 lots) and Sweden's GDP-to-spend ratio is more plausible than Hungary's.
# We retain Sweden but flag it as potentially overstated.
#
# **All other countries** with big lots (FRA, DNK, POL, ROU, ISL, NLD, ITA, DEU, SRB)
# show 0% round-billion rates and plausible absolute values. They are retained.
#
# > **Going forward:** All financial analysis from Section 3 onward is presented in
# > two views where relevant — **Full dataset** (including Hungary, for completeness)
# > and **Ex-Hungary** (the analytically reliable view). KPIs cited in the presentation
# > to Microsoft use the Ex-Hungary figures unless otherwise noted.

notice_awards_exhun = notice_awards_clean[notice_awards_clean['buyer_country'] != 'HUN'].copy()

spend_exhun = (notice_awards_exhun
    .groupby('buyer_country')
    .agg(
        total_awarded_eur = ('awarded_eur_total', 'sum'),
        notice_count      = ('notice_id', 'count')
    )
    .reset_index()
    .sort_values('total_awarded_eur', ascending=False)
)
spend_exhun['pct_of_total']   = spend_exhun['total_awarded_eur'] / spend_exhun['total_awarded_eur'].sum() * 100
spend_exhun['cumulative_pct'] = spend_exhun['pct_of_total'].cumsum()

total_exhun = spend_exhun['total_awarded_eur'].sum()
total_full  = notice_awards_clean['awarded_eur_total'].sum()
hun_total   = notice_awards_clean[notice_awards_clean['buyer_country']=='HUN']['awarded_eur_total'].sum()

print('=== TOTAL SPEND — DUAL VIEW ===')
print(f'Full dataset (incl. Hungary):  {fmt_eur(total_full)}')
print(f'Hungary alone:                 {fmt_eur(hun_total)}  ({hun_total/total_full*100:.1f}% of total)')
print(f'Ex-Hungary total:              {fmt_eur(total_exhun)}')
print()
print('Ex-Hungary — Top 20 countries by awarded spend:')
print(f'{"Country":<10} {"Total Awarded":>16} {"% of Total":>12} {"Cumulative %":>14} {"Notices":>10}')
print('-' * 68)
for _, row in spend_exhun.head(20).iterrows():
    print(f'{row["buyer_country"]:<10} {fmt_eur(row["total_awarded_eur"]):>16}'
          f' {row["pct_of_total"]:>11.1f}% {row["cumulative_pct"]:>13.1f}%'
          f' {int(row["notice_count"]):>10,}')
print()
n_80_exhun = (spend_exhun['cumulative_pct'] <= 80).sum() + 1
top_n_exhun = spend_exhun.head(n_80_exhun)['buyer_country'].tolist()
print(f'80% of ex-Hungary spend is in the top {n_80_exhun} countries: {top_n_exhun}')

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Total Awarded Spend by Country | January 2026', fontsize=13, fontweight='bold')

top10_full = spend_country.head(10)
bars1 = axes[0].bar(top10_full['buyer_country'], top10_full['total_awarded_eur'],
                    color=C_GREY, alpha=0.7, edgecolor='white')
axes[0].set_title('Full Dataset (incl. Hungary)', fontweight='bold')
axes[0].set_ylabel('Total Awarded Value (EUR)')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(fmt_eur))
axes[0].tick_params(axis='x', rotation=30)
for bar in bars1:
    h = bar.get_height()
    axes[0].text(bar.get_x()+bar.get_width()/2, h*1.01, fmt_eur(h),
                 ha='center', fontsize=7, rotation=45)
axes[0].text(0.97, 0.97, f'Total: {fmt_eur(total_full)}\n⚠ Hungary inflated by\nframework ceilings',
             transform=axes[0].transAxes, ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='#FFF3CD', alpha=0.8))

top10_exhun = spend_exhun.head(10)
bars2 = axes[1].bar(top10_exhun['buyer_country'], top10_exhun['total_awarded_eur'],
                    color=C_BLUE, alpha=0.85, edgecolor='white')
axes[1].set_title('Ex-Hungary (Analytically Reliable View)', fontweight='bold')
axes[1].set_ylabel('Total Awarded Value (EUR)')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(fmt_eur))
axes[1].tick_params(axis='x', rotation=30)
for bar in bars2:
    h = bar.get_height()
    axes[1].text(bar.get_x()+bar.get_width()/2, h*1.01, fmt_eur(h),
                 ha='center', fontsize=7, rotation=45)
axes[1].text(0.97, 0.97, f'Total: {fmt_eur(total_exhun)}\n✓ Recommended view\nfor presentation',
             transform=axes[1].transAxes, ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='#D4EDDA', alpha=0.8))

plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------
# ## 4. Average Contract Size — Mean & Median Awarded Value per Notice
# > **KPI Definition — Average Contract Value (ACV):**
# > The mean awarded_eur per notice within a country. Tells us the typical size of a
# > public contract in that country. A country can have high total spend (many small
# > contracts) or high ACV (few large contracts). These two dimensions tell different stories.
# >
# > **KPI Definition — Median Contract Value:**
# > The midpoint of the awarded value distribution. Much more robust than the mean because
# > a single €100M mega-contract can inflate the mean dramatically while the median stays
# > at the typical small-contract level. Always report both.

acv_country = (notice_awards_clean
    .groupby('buyer_country')
    .agg(
        mean_eur   = ('awarded_eur_total', 'mean'),
        median_eur = ('awarded_eur_total', 'median'),
        std_eur    = ('awarded_eur_total', 'std'),
        n_notices  = ('notice_id', 'count'),
        max_eur    = ('awarded_eur_total', 'max')
    )
    .reset_index()
)
acv_country = acv_country[acv_country['n_notices'] >= 20].sort_values('mean_eur', ascending=False)

print('Average and Median Contract Value by Country (>= 20 notices with awarded value):')
print(f'{"Country":<10} {"Mean":>12} {"Median":>12} {"Max":>12} {"N Notices":>12}')
print('-' * 62)
for _, row in acv_country.iterrows():
    skew_flag = ' ← high skew (mean >> median)' if row['mean_eur'] > row['median_eur'] * 5 else ''
    print(f'{row["buyer_country"]:<10} {fmt_eur(row["mean_eur"]):>12} {fmt_eur(row["median_eur"]):>12}'
          f' {fmt_eur(row["max_eur"]):>12} {int(row["n_notices"]):>12,}{skew_flag}')

top_acv = acv_country.head(15)

x     = np.arange(len(top_acv))
width = 0.38

fig, ax = plt.subplots(figsize=(15, 6))
bars_mean   = ax.bar(x - width/2, top_acv['mean_eur'],   width, color=C_BLUE,   alpha=0.85, label='Mean')
bars_median = ax.bar(x + width/2, top_acv['median_eur'], width, color=C_ORANGE, alpha=0.85, label='Median')

ax.set_xticks(x)
ax.set_xticklabels(top_acv['buyer_country'], rotation=30, ha='right')
ax.set_ylabel('Awarded Value per Notice (EUR)')
ax.set_title('Average Contract Size — Mean vs Median | Top 15 Countries by Mean | January 2026',
             fontweight='bold')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_eur))
ax.legend()

for bar in list(bars_mean) + list(bars_median):
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h * 1.01, fmt_eur(h),
                ha='center', fontsize=7, rotation=45)

plt.tight_layout()
plt.show()
print()
print('INTERPRETATION:')
print('Where Mean >> Median, a small number of very large contracts are pulling the average up.')
print('Median is more representative of the typical contract a supplier would bid on.')

acv_exhun = (notice_awards_exhun
    .groupby('buyer_country')
    .agg(
        mean_eur   = ('awarded_eur_total', 'mean'),
        median_eur = ('awarded_eur_total', 'median'),
        n_notices  = ('notice_id', 'count'),
        max_eur    = ('awarded_eur_total', 'max')
    )
    .reset_index()
)
acv_exhun = acv_exhun[acv_exhun['n_notices'] >= 20].sort_values('mean_eur', ascending=False)

print('Average Contract Size — Ex-Hungary (countries with >= 20 notices):')
print(f'{"Country":<10} {"Mean":>12} {"Median":>12} {"Max":>12} {"N Notices":>12}')
print('-' * 62)
for _, row in acv_exhun.iterrows():
    skew = ' ← high skew' if row['mean_eur'] > row['median_eur'] * 5 else ''
    print(f'{row["buyer_country"]:<10} {fmt_eur(row["mean_eur"]):>12}'
          f' {fmt_eur(row["median_eur"]):>12} {fmt_eur(row["max_eur"]):>12}'
          f' {int(row["n_notices"]):>12,}{skew}')

print()
overall_median_exhun = notice_awards_exhun['awarded_eur_total'].median()
overall_mean_exhun   = notice_awards_exhun['awarded_eur_total'].mean()
print(f'Overall ex-Hungary median contract value: {fmt_eur(overall_median_exhun)}')
print(f'Overall ex-Hungary mean contract value:   {fmt_eur(overall_mean_exhun)}')


# ----------------------------------------------------------------------
# ## 5. Estimated vs Awarded — Are Buyers Accurate Estimators?
# > **KPI Definition — Estimation Accuracy / Savings %:**
# > `savings_pct = (estimated - awarded) / estimated × 100`
# > A positive savings_pct means the final contract came in BELOW the buyer's estimate.
# > A negative savings_pct means the final contract EXCEEDED the estimate (cost overrun).
# >
# > **Why this matters:** Systematic over-estimation (high positive savings) could mean
# > buyers are padding budgets. Under-estimation (negative savings) could signal scope creep
# > or poor planning. Country-level patterns reveal procurement culture.
# >
# > **Subset used:** Only notices where BOTH `estimated` and `awarded_eur` are non-null.
# > This is the most restricted subset — expect a relatively small sample.

est_aw_lots = gold_awards[gold_awards['estimated'].notna() & gold_awards['cv_clean'].notna()].copy()
est_aw_lots = est_aw_lots[est_aw_lots['cv_clean'] > 0]

est_vs_aw = (est_aw_lots
    .groupby(['notice_id', 'buyer_country', 'proc_type'])
    .agg(
        estimated_total = ('estimated',  'sum'),
        awarded_total   = ('cv_clean',   'sum')
    )
    .reset_index()
)
est_vs_aw = est_vs_aw[
    (est_vs_aw['estimated_total'] > 0) &
    (est_vs_aw['awarded_total']   > 0)
].copy()

est_vs_aw['savings_eur'] = est_vs_aw['estimated_total'] - est_vs_aw['awarded_total']
est_vs_aw['savings_pct'] = est_vs_aw['savings_eur'] / est_vs_aw['estimated_total'] * 100

est_vs_aw_clean = est_vs_aw[
    (est_vs_aw['savings_pct'] >= -200) &
    (est_vs_aw['savings_pct'] <=  200)
].copy()

print('=== ESTIMATED vs AWARDED SUBSET ===')
print(f'Lots with both estimated + contract_value:          {len(est_aw_lots):>6,}')
print(f'Collapsed to notice-level (both fields > 0):        {len(est_vs_aw):>6,}')
print(f'After removing extreme outliers (|savings| > 200%): {len(est_vs_aw_clean):>6,}')
print()
print(f'Mean savings %:    {est_vs_aw_clean["savings_pct"].mean():.1f}%')
print(f'Median savings %:  {est_vs_aw_clean["savings_pct"].median():.1f}%')
print()
n_over  = (est_vs_aw_clean['savings_pct'] < 0).sum()
n_under = (est_vs_aw_clean['savings_pct'] > 0).sum()
print(f'Contracts that EXCEEDED estimate (cost overrun):    {n_over:>6,} ({n_over/len(est_vs_aw_clean)*100:.1f}%)')
print(f'Contracts that came in BELOW estimate (savings):    {n_under:>6,} ({n_under/len(est_vs_aw_clean)*100:.1f}%)')

savings_country = (est_vs_aw_clean
    .groupby('buyer_country')
    .agg(
        mean_savings_pct   = ('savings_pct', 'mean'),
        median_savings_pct = ('savings_pct', 'median'),
        n_contracts        = ('notice_id',   'count'),
        total_savings_eur  = ('savings_eur', 'sum')
    )
    .reset_index()
)
savings_country = savings_country[
    savings_country['n_contracts'] >= 5
].sort_values('median_savings_pct', ascending=False)

print('Savings % by country (countries with >= 5 matched contracts):')
print(f'{"Country":<10} {"Median Sav%":>12} {"Mean Sav%":>12} {"Contracts":>12} {"Total Savings EUR":>20}')
print('-' * 72)
for _, row in savings_country.iterrows():
    flag = ' ← OVERRUN' if row['median_savings_pct'] < 0 else ''
    print(f'{row["buyer_country"]:<10} {row["median_savings_pct"]:>11.1f}%'
          f' {row["mean_savings_pct"]:>11.1f}%'
          f' {int(row["n_contracts"]):>12,}'
          f' {fmt_eur(row["total_savings_eur"]):>20}{flag}')

sav_plot = savings_country.sort_values('median_savings_pct').head(20)

fig, ax = plt.subplots(figsize=(12, 8))
colors = [C_GREEN if v >= 0 else C_RED for v in sav_plot['median_savings_pct']]
bars = ax.barh(sav_plot['buyer_country'], sav_plot['median_savings_pct'],
               color=colors, alpha=0.85, edgecolor='white')
ax.axvline(0, color='black', linewidth=1.2)
ax.set_xlabel('Median Savings % (positive = under estimate, negative = cost overrun)')
ax.set_title('Estimation Accuracy by Country — Median Savings % | January 2026', fontweight='bold')

for bar, val in zip(bars, sav_plot['median_savings_pct']):
    ax.text(val + (0.5 if val >= 0 else -0.5), bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', ha='left' if val >= 0 else 'right', fontsize=8)

ax.add_patch(mpatches.FancyArrowPatch((0, -0.8), (8, -0.8),
    arrowstyle='->', mutation_scale=15, color=C_GREEN))
ax.text(4, -1.1, 'Savings (came in under budget)', ha='center', fontsize=9, color=C_GREEN)
ax.add_patch(mpatches.FancyArrowPatch((0, -0.8), (-8, -0.8),
    arrowstyle='->', mutation_scale=15, color=C_RED))
ax.text(-4, -1.1, 'Overrun (exceeded budget)', ha='center', fontsize=9, color=C_RED)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(est_vs_aw_clean['savings_pct'], bins=60, color=C_TEAL, alpha=0.8, edgecolor='white')
axes[0].axvline(0, color=C_RED, linewidth=1.5, linestyle='--', label='Break-even (0%)')
axes[0].axvline(
    est_vs_aw_clean['savings_pct'].median(), color=C_ORANGE, linewidth=1.5,
    linestyle='--', label=f'Median ({est_vs_aw_clean["savings_pct"].median():.1f}%)'
)
axes[0].set_xlabel('Savings %')
axes[0].set_ylabel('Number of Contracts')
axes[0].set_title('Distribution of Savings % — All Countries', fontweight='bold')
axes[0].legend()

for pt, color in [('Services', C_BLUE), ('Supplies', C_ORANGE), ('Works', C_GREEN)]:
    subset = est_vs_aw_clean[est_vs_aw_clean['proc_type'] == pt]['savings_pct']
    if len(subset) >= 5:
        axes[1].hist(subset, bins=40, alpha=0.5, color=color,
                     label=f'{pt} (n={len(subset):,})', edgecolor='white')
axes[1].axvline(0, color='black', linewidth=1.2, linestyle='--')
axes[1].set_xlabel('Savings %')
axes[1].set_ylabel('Number of Contracts')
axes[1].set_title('Savings % Distribution by Procurement Type', fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.show()

print('Savings % summary by procurement type:')
print(est_vs_aw_clean.groupby('proc_type')['savings_pct']
      .agg(['mean', 'median', 'count']).round(2).to_string())


# ----------------------------------------------------------------------
# ## 6. Value Distribution — Are Mega-Contracts Driving Total Spend?
# > **Why this analysis:** The mean vs median gap in Section 4 hinted at high skew.
# > This section quantifies it properly. In public procurement it is common for 80% of
# > total EUR value to come from 5-10% of contracts (a Pareto distribution).
# > Understanding this is critical for Person 5's dashboard — if a country's spend
# > is driven by 2 mega-contracts, the chart tells a very different story than if it's
# > spread across 500 mid-sized contracts.
# >
# > **KPI Definition — Value Concentration Ratio:**
# > The percentage of total spend accounted for by the top 10% of contracts by value.
# > A ratio above 80% signals extreme concentration. Below 50% suggests a more
# > competitive, distributed market.

vals = notice_awards_clean['awarded_eur_total'].sort_values(ascending=False).reset_index(drop=True)
total_val = vals.sum()
n_total   = len(vals)

thresholds = [1, 5, 10, 20, 50]
print('=== VALUE CONCENTRATION ANALYSIS ===')
print(f'Total notices with awarded value: {n_total:,}')
print(f'Total awarded spend:              {fmt_eur(total_val)}')
print()
print(f'{"Top X% of contracts":>25} {"N Contracts":>14} {"Value":>14} {"% of Total Spend":>18}')
print('-' * 76)
for pct in thresholds:
    n_contracts  = max(1, int(n_total * pct / 100))
    top_val      = vals.iloc[:n_contracts].sum()
    top_pct      = top_val / total_val * 100
    print(f'  Top {pct:>2}% of contracts     {n_contracts:>14,} {fmt_eur(top_val):>14} {top_pct:>17.1f}%')

print()
brackets = [
    ('< €10K',      0,          10_000),
    ('€10K–€100K',  10_000,    100_000),
    ('€100K–€1M',   100_000,  1_000_000),
    ('€1M–€10M',  1_000_000, 10_000_000),
    ('€10M–€100M',10_000_000,100_000_000),
    ('> €100M',  100_000_000, float('inf'))
]
print('Value bracket distribution:')
print(f'{"Bracket":>18} {"N Contracts":>14} {"% of Count":>12} {"Total Value":>14} {"% of Spend":>12}')
print('-' * 76)
for label, lo, hi in brackets:
    mask  = (vals >= lo) & (vals < hi)
    n_br  = mask.sum()
    v_br  = vals[mask].sum()
    print(f'{label:>18} {n_br:>14,} {n_br/n_total*100:>11.1f}% {fmt_eur(v_br):>14} {v_br/total_val*100:>11.1f}%')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

x_pct = np.linspace(0, 100, n_total)
y_cum = vals.cumsum() / total_val * 100
axes[0].plot(x_pct[::-1], y_cum.values, color=C_BLUE, linewidth=2, label='Actual')
axes[0].plot([0, 100], [0, 100], color=C_GREY, linestyle='--', linewidth=1, label='Perfect equality')
axes[0].axhline(80, color=C_ORANGE, linestyle=':', alpha=0.7, label='80% spend')
axes[0].set_xlabel('% of Contracts (ranked by value, largest first)')
axes[0].set_ylabel('Cumulative % of Total Spend')
axes[0].set_title('Value Concentration — Pareto Curve', fontweight='bold')
axes[0].legend()

brackets = [
    ('< €10K',      0,           10_000),
    ('€10K–€100K',  10_000,     100_000),
    ('€100K–€1M',   100_000,  1_000_000),
    ('€1M–€10M',  1_000_000, 10_000_000),
    ('€10M–€100M',10_000_000,100_000_000),
    ('> €100M',  100_000_000, float('inf'))
]
bracket_labels  = [b[0] for b in brackets]
bracket_counts  = []
bracket_values  = []
for label, lo, hi in brackets:
    mask = (notice_awards_clean['awarded_eur_total'] >= lo) & \
           (notice_awards_clean['awarded_eur_total'] <  hi)
    bracket_counts.append(int(mask.sum()))
    bracket_values.append(float(notice_awards_clean.loc[mask, 'awarded_eur_total'].sum()))

x_br = np.arange(len(bracket_labels))
ax2b = axes[1].twinx()
axes[1].bar(x_br, bracket_counts, color=C_TEAL, alpha=0.7, label='# Contracts')
ax2b.plot(x_br, [v / 1e6 for v in bracket_values],
          color=C_ORANGE, marker='o', linewidth=2, markersize=6, label='Total Value (€M)')
axes[1].set_xticks(x_br)
axes[1].set_xticklabels(bracket_labels, rotation=30, ha='right')
axes[1].set_ylabel('Number of Contracts', color=C_TEAL)
ax2b.set_ylabel('Total Awarded Value (€M)', color=C_ORANGE)
axes[1].set_title('Contract Count vs Total Value by Size Bracket', fontweight='bold')
axes[1].legend(loc='upper left')
ax2b.legend(loc='upper right')

plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------
# ## 7. Spend by Procurement Type — Services vs Supplies vs Works
# > **Why this differs from NB1's type analysis:**
# > NB1 counted *notices* by type — useful for understanding market activity volume.
# > This section adds the EUR dimension: which type attracts the most money?
# > A category could be high-volume but low-value (many small supply orders)
# > or low-volume but high-value (a few large infrastructure works contracts).
# >
# > **Reminder:** We use notice-level aggregation to avoid Supplies' lot-inflation.

spend_type = (notice_awards_clean
    .groupby('proc_type')
    .agg(
        total_eur   = ('awarded_eur_total', 'sum'),
        mean_eur    = ('awarded_eur_total', 'mean'),
        median_eur  = ('awarded_eur_total', 'median'),
        n_contracts = ('notice_id', 'count')
    )
    .reset_index()
    .sort_values('total_eur', ascending=False)
)
spend_type['pct_of_spend'] = spend_type['total_eur'] / spend_type['total_eur'].sum() * 100

print('Spend by procurement type (notice-normalised):')
print(spend_type.assign(
    total_eur  = spend_type['total_eur'].apply(fmt_eur),
    mean_eur   = spend_type['mean_eur'].apply(fmt_eur),
    median_eur = spend_type['median_eur'].apply(fmt_eur)
).to_string(index=False))
print()
cn_type_counts = gold_opportunities['proc_type'].value_counts().reset_index()
cn_type_counts.columns = ['proc_type', 'cn_count']
spend_type_merged = spend_type.merge(cn_type_counts, on='proc_type', how='left')
spend_type_merged['eur_per_cn_notice'] = spend_type_merged['total_eur'] / spend_type_merged['cn_count']
print('Value per open tender (CN) by type:')
for _, row in spend_type_merged.iterrows():
    print(f'  {row["proc_type"]:<12}: {fmt_eur(row["total_eur"])} total | {int(row["cn_count"]):,} CN notices | {fmt_eur(row["eur_per_cn_notice"])} per CN notice')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Procurement Type — Financial Breakdown | January 2026', fontsize=13, fontweight='bold')
type_colors = {'Services': C_BLUE, 'Supplies': C_ORANGE, 'Works': C_GREEN}
types = spend_type['proc_type'].tolist()
colors_list = [type_colors.get(t, C_GREY) for t in types]

wedges, texts, autotexts = axes[0, 0].pie(
    spend_type['total_eur'], labels=types,
    colors=colors_list, autopct='%1.1f%%',
    pctdistance=0.75, startangle=90,
    wedgeprops={'width': 0.55, 'edgecolor': 'white', 'linewidth': 2}
)
for at in autotexts: at.set_fontweight('bold')
axes[0, 0].set_title('% of Total Awarded Spend', fontweight='bold')

bars2 = axes[0, 1].bar(types, spend_type['total_eur'], color=colors_list, alpha=0.85, edgecolor='white')
axes[0, 1].yaxis.set_major_formatter(mticker.FuncFormatter(fmt_eur))
axes[0, 1].set_title('Total Awarded Value (EUR)', fontweight='bold')
for bar in bars2:
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    fmt_eur(bar.get_height()), ha='center', fontsize=9, fontweight='bold')

x_t = np.arange(len(types))
axes[1, 0].bar(x_t - 0.2, spend_type['mean_eur'],   0.38, color=colors_list, alpha=0.85, label='Mean')
axes[1, 0].bar(x_t + 0.2, spend_type['median_eur'], 0.38, color=colors_list, alpha=0.45, label='Median')
axes[1, 0].set_xticks(x_t); axes[1, 0].set_xticklabels(types)
axes[1, 0].yaxis.set_major_formatter(mticker.FuncFormatter(fmt_eur))
axes[1, 0].set_title('Mean vs Median Contract Size by Type', fontweight='bold')
axes[1, 0].legend()

bars4 = axes[1, 1].bar(types, spend_type['n_contracts'], color=colors_list, alpha=0.85, edgecolor='white')
axes[1, 1].set_title('Number of Awarded Contracts (Notice-Level)', fontweight='bold')
axes[1, 1].set_ylabel('Notice Count')
for bar in bars4:
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    f'{int(bar.get_height()):,}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------
# ## 8. Top CPV Categories by Total Awarded Value
# > **KPI Definition — CPV Category Spend:**
# > Total awarded_eur summed across all notices within a CPV division.
# > CPV divisions are the 2-digit groupings (e.g. Division 45 = Construction works,
# > Division 72 = IT services). This tells us: which procurement categories represent
# > the most money in the EU market.
# >
# > **Why this matters:** Microsoft operates heavily in Division 72 (IT services)
# > and Division 48 (Software packages). Understanding whether these categories are
# > high-value — not just high-volume — is critical market intelligence.

cpv_spend = (notice_awards_clean
    .groupby(['cpv_division', 'cpv_division_name'])
    .agg(
        total_eur   = ('awarded_eur_total', 'sum'),
        mean_eur    = ('awarded_eur_total', 'mean'),
        median_eur  = ('awarded_eur_total', 'median'),
        n_contracts = ('notice_id', 'count')
    )
    .reset_index()
    .sort_values('total_eur', ascending=False)
)
cpv_spend['pct_of_spend'] = cpv_spend['total_eur'] / cpv_spend['total_eur'].sum() * 100
cpv_spend['cumulative_pct'] = cpv_spend['pct_of_spend'].cumsum()

print('Top 20 CPV divisions by total awarded value:')
print(f'{"CPV":>5} {"Division Name":35} {"Total Value":>14} {"% Spend":>9} {"Cumul%":>8} {"N":>8} {"Median":>12}')
print('-' * 100)
for _, row in cpv_spend.head(20).iterrows():
    name = str(row['cpv_division_name'])[:33] if pd.notna(row['cpv_division_name']) else 'Unknown'
    print(f'{str(row["cpv_division"]):>5} {name:<35} {fmt_eur(row["total_eur"]):>14}'
          f' {row["pct_of_spend"]:>8.1f}% {row["cumulative_pct"]:>7.1f}%'
          f' {int(row["n_contracts"]):>8,} {fmt_eur(row["median_eur"]):>12}')

print()
msft_divs = ['72', '48', '73', '79', '50']
msft_cpv  = cpv_spend[cpv_spend['cpv_division'].astype(str).isin(msft_divs)]
if len(msft_cpv) > 0:
    print('Microsoft-relevant CPV divisions (IT, Software, R&D, Maintenance):')
    for _, row in msft_cpv.iterrows():
        name = str(row['cpv_division_name'])[:40] if pd.notna(row['cpv_division_name']) else 'Unknown'
        print(f'  CPV {row["cpv_division"]:>3}: {name:<40} {fmt_eur(row["total_eur"]):>12}'
              f'  ({row["pct_of_spend"]:.1f}% of total)  n={int(row["n_contracts"]):,}')

top15_cpv = cpv_spend.head(15).copy()
top15_cpv['label'] = top15_cpv.apply(
    lambda r: f"CPV {r['cpv_division']}\n{str(r['cpv_division_name'])[:22] if pd.notna(r['cpv_division_name']) else 'Unknown'}",
    axis=1
)

fig, ax = plt.subplots(figsize=(14, 8))
bars = ax.barh(
    top15_cpv['label'][::-1].tolist(),
    top15_cpv['total_eur'][::-1].tolist(),
    color=C_PURPLE, alpha=0.85, edgecolor='white'
)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_eur))
ax.set_xlabel('Total Awarded Value (EUR)')
ax.set_title('Top 15 CPV Divisions by Total Awarded Value | January 2026', fontweight='bold')

for bar, val in zip(bars, top15_cpv['total_eur'][::-1]):
    ax.text(val * 1.01, bar.get_y() + bar.get_height()/2,
            fmt_eur(val), va='center', fontsize=8)

plt.tight_layout()
plt.show()

n_cpv_80 = (cpv_spend['cumulative_pct'] <= 80).sum() + 1
print(f'80% of total awarded spend is concentrated in the top {n_cpv_80} CPV divisions (out of {len(cpv_spend)}).')


# ----------------------------------------------------------------------
# ## 9. Largest Individual Contracts — The Mega-Deal Leaderboard
# > **Why this matters:** A handful of mega-contracts can distort country and CPV totals.
# > Identifying them by name, country, and category lets Person 5 flag them on the dashboard
# > and lets Person 6's chatbot answer 'What was the largest contract in January 2026?'

titles = (gold_awards
    .dropna(subset=['project_title'])
    .groupby('notice_id')['project_title']
    .first()
    .reset_index()
)

top_contracts = (notice_awards_clean
    .sort_values('awarded_eur_total', ascending=False)
    .head(25)
    .merge(titles, on='notice_id', how='left')
)

buyers = gold_awards[['notice_id','buyer_name']].drop_duplicates('notice_id')
top_contracts = top_contracts.merge(buyers, on='notice_id', how='left')

print('Top 25 Largest Individual Contracts — January 2026:')
print(f'{"Rank":>4} {"Value":>14} {"Ctry":>5} {"Type":>10} {"CPV":>5} {"Project Title / Buyer":<}')
print('-' * 100)
for rank, (_, row) in enumerate(top_contracts.iterrows(), 1):
    title = str(row.get('project_title', ''))[:35] if pd.notna(row.get('project_title')) else ''
    buyer = str(row.get('buyer_name',    ''))[:30] if pd.notna(row.get('buyer_name'))    else 'Unknown'
    cpv   = str(row.get('cpv_division',  '?'))
    cty   = str(row.get('buyer_country', '?'))
    pt    = str(row.get('proc_type',     '?'))
    display = title if title else buyer
    print(f'{rank:>4} {fmt_eur(row["awarded_eur_total"]):>14} {cty:>5} {pt:>10} {cpv:>5}  {display}')

titles = (gold_awards
    .dropna(subset=['project_title'])
    .groupby('notice_id')['project_title']
    .first()
    .reset_index()
)
buyers = gold_awards[['notice_id','buyer_name']].drop_duplicates('notice_id')

top_contracts_exhun = (notice_awards_exhun
    .sort_values('awarded_eur_total', ascending=False)
    .head(25)
    .merge(titles, on='notice_id', how='left')
    .merge(buyers, on='notice_id', how='left')
)

print('Top 25 Largest Contracts — Ex-Hungary | January 2026:')
print(f'{"Rank":>4} {"Value":>14} {"Ctry":>5} {"Type":>10} {"CPV":>5}  Project Title / Buyer')
print('-' * 100)
for rank, (_, row) in enumerate(top_contracts_exhun.iterrows(), 1):
    title   = str(row.get('project_title',''))[:38] if pd.notna(row.get('project_title')) else ''
    buyer   = str(row.get('buyer_name',''))[:38]    if pd.notna(row.get('buyer_name'))    else 'Unknown'
    display = title if title else buyer
    print(f'{rank:>4} {fmt_eur(row["awarded_eur_total"]):>14}'
          f' {str(row.get("buyer_country","?"))!s:>5}'
          f' {str(row.get("proc_type","?"))!s:>10}'
          f' {str(row.get("cpv_division","?"))!s:>5}  {display}')


# > **Microsoft spotlight — Largest IT contract in the dataset (ex-Hungary):**
# > Rank 13 in the ex-Hungary leaderboard is a Czech Republic contract worth **€4.5B**:
# > *"RÁMCOVÁ DOHODA NA ROZVOJ SYSTÉMU BI/AA"* — a framework agreement for Business Intelligence
# > and Advanced Analytics system development (CPV 72, IT Services).
# > This is the **single largest IT/analytics contract in the entire dataset outside Hungary**,
# > awarded by the Czech government. It is directly relevant to Microsoft's positioning in
# > public sector BI and cloud analytics (Azure, Power BI, Fabric).

buyer_names = (gold_awards[['notice_id','buyer_name','buyer_country']]
    .drop_duplicates('notice_id'))

buyer_spend = (notice_awards_clean
    .merge(buyer_names, on='notice_id', how='left', suffixes=('','_raw'))
    .groupby(['buyer_name','buyer_country'])
    .agg(
        total_eur   = ('awarded_eur_total', 'sum'),
        n_contracts = ('notice_id',         'count')
    )
    .reset_index()
    .sort_values('total_eur', ascending=False)
)

print('Top 20 Buyers by Total Awarded Spend — January 2026:')
print(f'{"Rank":>5} {"Total Awarded":>16} {"Country":>9} {"N Contracts":>13}  {"Buyer Name"}')
print('-' * 90)
for rank, (_, row) in enumerate(buyer_spend.head(20).iterrows(), 1):
    buyer = str(row['buyer_name'])[:45] if pd.notna(row['buyer_name']) else 'Unknown'
    print(f'{rank:>5} {fmt_eur(row["total_eur"]):>16} {str(row["buyer_country"]):>9}'
          f' {int(row["n_contracts"]):>13,}  {buyer}')

top10_spend = buyer_spend.head(10)['total_eur'].sum()
total_spend = buyer_spend['total_eur'].sum()
print()
print(f'Top 10 buyers account for {top10_spend/total_spend*100:.1f}% of all awarded spend.')
print(f'Total unique buyers with awarded contracts: {len(buyer_spend):,}')

buyer_names = gold_awards[['notice_id','buyer_name','buyer_country']].drop_duplicates('notice_id')

buyer_spend_exhun = (notice_awards_exhun
    .merge(buyer_names, on='notice_id', how='left', suffixes=('','_raw'))
    .groupby(['buyer_name','buyer_country'])
    .agg(
        total_eur   = ('awarded_eur_total', 'sum'),
        n_contracts = ('notice_id',         'count')
    )
    .reset_index()
    .sort_values('total_eur', ascending=False)
)

print('Top 20 Buyers by Total Awarded Spend — Ex-Hungary | January 2026:')
print(f'{"Rank":>5} {"Total Awarded":>16} {"Country":>9} {"N Contracts":>13}  Buyer Name')
print('-' * 90)
for rank, (_, row) in enumerate(buyer_spend_exhun.head(20).iterrows(), 1):
    buyer = str(row['buyer_name'])[:45] if pd.notna(row['buyer_name']) else 'Unknown'
    print(f'{rank:>5} {fmt_eur(row["total_eur"]):>16}'
          f' {str(row["buyer_country"]):>9}'
          f' {int(row["n_contracts"]):>13,}  {buyer}')

top10_exhun_spend = buyer_spend_exhun.head(10)['total_eur'].sum()
total_exhun_spend = buyer_spend_exhun['total_eur'].sum()
print()
print(f'Top 10 buyers (ex-HUN) account for {top10_exhun_spend/total_exhun_spend*100:.1f}% of ex-Hungary spend.')
print(f'Total unique buyers (ex-HUN):       {len(buyer_spend_exhun):,}')

bse = buyer_spend_exhun.reset_index(drop=True)
bse['cumulative_pct'] = bse['total_eur'].cumsum() / bse['total_eur'].sum() * 100
total_buyers_exhun = len(bse)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Buyer Concentration — Ex-Hungary | January 2026', fontweight='bold', fontsize=13)

x_b = np.arange(1, total_buyers_exhun+1) / total_buyers_exhun * 100
axes[0].plot(x_b, bse['cumulative_pct'].values, color=C_BLUE, linewidth=2, label='Actual')
axes[0].plot([0,100],[0,100], color=C_GREY, linestyle='--', linewidth=1, label='Perfect equality')
axes[0].axhline(80, color=C_ORANGE, linestyle=':', alpha=0.7, label='80% spend')
axes[0].set_xlabel('% of Buyers (ranked by spend)')
axes[0].set_ylabel('Cumulative % of Total Spend')
axes[0].set_title('Buyer Concentration Curve (Ex-Hungary)', fontweight='bold')
axes[0].legend()

top15_bse = bse.head(15)
names = [str(n)[:32] for n in top15_bse['buyer_name']]
axes[1].barh(names[::-1], top15_bse['total_eur'].tolist()[::-1],
             color=C_GOLD, alpha=0.85, edgecolor='white')
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(fmt_eur))
axes[1].set_xlabel('Total Awarded Spend (EUR)')
axes[1].set_title('Top 15 Buyers by Spend (Ex-Hungary)', fontweight='bold')
plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------
# ## 10. Buyer Concentration — How Many Buyers Drive Most of the Spend?
# > **KPI Definition — Buyer Concentration Ratio (BCR):**
# > The percentage of total awarded spend accounted for by the top N buyers.
# > A high BCR means procurement is dominated by a few large institutional buyers.
# > A low BCR means spend is fragmented across many buyers.
# >
# > **What the data shows:** Only 5 buyers are needed to reach 50% of total spend,
# > and just 79 buyers cover 80%. Out of 11,417 unique buyers, the top 1 buyer alone
# > (Hungary's digital agency) accounts for 32.6% — an extreme concentration driven
# > entirely by the framework ceiling anomaly identified in Section 3b.
# > The ex-Hungary view (Section 12) gives the analytically sound picture.

buyer_spend_sorted = buyer_spend.sort_values('total_eur', ascending=False).reset_index(drop=True)
buyer_spend_sorted['cumulative_pct'] = (
    buyer_spend_sorted['total_eur'].cumsum() / buyer_spend_sorted['total_eur'].sum() * 100
)
total_buyers = len(buyer_spend_sorted)

print('=== BUYER CONCENTRATION ANALYSIS ===')
print(f'Total buyers with awarded contracts: {total_buyers:,}')
print()
for n in [1, 5, 10, 25, 50, 100]:
    if n <= total_buyers:
        cum_pct    = buyer_spend_sorted.iloc[n-1]['cumulative_pct']
        pct_buyers = n / total_buyers * 100
        print(f'  Top {n:>4} buyers ({pct_buyers:.1f}% of all buyers) = {cum_pct:.1f}% of total spend')

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Buyer Concentration Analysis — January 2026', fontweight='bold', fontsize=13)

x_b = np.arange(1, total_buyers + 1) / total_buyers * 100
y_b = buyer_spend_sorted['cumulative_pct'].values
axes[0].plot(x_b, y_b, color=C_BLUE, linewidth=2, label='Actual')
axes[0].plot([0, 100], [0, 100], color=C_GREY, linestyle='--', linewidth=1, label='Perfect equality')
axes[0].axhline(80, color=C_ORANGE, linestyle=':', alpha=0.7, label='80% spend')
axes[0].set_xlabel('% of Buyers (ranked by spend)')
axes[0].set_ylabel('Cumulative % of Total Spend')
axes[0].set_title('Buyer Concentration Curve', fontweight='bold')
axes[0].legend()

top15_b = buyer_spend_sorted.head(15)
names   = [str(n)[:32] for n in top15_b['buyer_name']]
axes[1].barh(names[::-1], top15_b['total_eur'].tolist()[::-1],
             color=C_GOLD, alpha=0.85, edgecolor='white')
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(fmt_eur))
axes[1].set_xlabel('Total Awarded Spend (EUR)')
axes[1].set_title('Top 15 Buyers by Total Spend', fontweight='bold')

plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------
# ## 11. Contract Size Segmentation — Micro / Small / Mid / Large
# > **Why this analysis:** The Pareto curve in Section 6 showed that 95% of total spend
# > sits in the 'Large' bracket (>€5M), yet only 8.4% of contracts fall there.
# > This section makes those segments explicit and cross-cuts them by procurement type,
# > giving Person 5 a chart that shows the volume/value trade-off at a glance.
# >
# > **KPI Definition — Contract Size Segment:**
# > Contracts are classified into four tiers based on awarded lot value:
# > - **Micro** (<€50K): sub-threshold; typically small-business territory
# > - **Small** (€50K–€500K): SME-accessible mid-market
# > - **Mid** (€500K–€5M): the competitive mid-market
# > - **Large** (>€5M): framework agreements and infrastructure mega-deals
# >
# > **Key insight from the data:**
# > - 68.8% of all lots are Micro or Small — high volume, low value
# > - Large contracts (8.4% of lots) account for 95.0% of total spend
# > - This confirms the procurement market is a dual economy: many small contracts
# >   for SMEs and a small number of mega-deals that drive almost all financial value
# >
# > **For Supplies specifically:** 41.1% of all Supplies lots are Micro (under €50K) — the highest Micro concentration of any type. Yet spend is dominated by the 6.1% of Large lots. Supplies is the most fragmented market by volume. Works is the most evenly distributed, with 21.1% of its lots in the Large bracket.

# ── CONTRACT SIZE SEGMENTATION ─────────────────────────────────────────────
# Bins match the four tiers used as KPI definitions above.
# We segment at the lot level (gold_awards with cv_clean) because
# the size of each individual lot is what a supplier bids on —
# a €20M notice split into 10 lots is actually ten €2M opportunities.

bins   = [0, 50_000, 500_000, 5_000_000, float('inf')]
labels = ['Micro (<€50K)', 'Small (€50K–500K)', 'Mid (€500K–5M)', 'Large (>€5M)']

awards_seg = gold_awards.dropna(subset=['cv_clean']).copy()
awards_seg['size_segment'] = pd.cut(
    awards_seg['cv_clean'], bins=bins, labels=labels, right=True
)

seg_overall = (
    awards_seg
    .groupby('size_segment', observed=True)
    .agg(
        lot_count     = ('notice_id', 'count'),
        total_spend   = ('cv_clean',  'sum'),
        avg_value     = ('cv_clean',  'mean'),
        median_value  = ('cv_clean',  'median')
    )
    .reset_index()
)
seg_overall['pct_lots']  = seg_overall['lot_count']  / seg_overall['lot_count'].sum()  * 100
seg_overall['pct_spend'] = seg_overall['total_spend'] / seg_overall['total_spend'].sum() * 100

print('=== CONTRACT SIZE SEGMENTATION (lot-level) ===')
print()
print(f'{"Segment":<22} {"Lots":>10} {"% Lots":>8} {"Total Spend":>14} {"% Spend":>9} {"Median Value":>14}')
print('-' * 82)
for _, row in seg_overall.iterrows():
    print(f'{str(row["size_segment"]):<22} {int(row["lot_count"]):>10,}'
          f' {row["pct_lots"]:>7.1f}%'
          f' {fmt_eur(row["total_spend"]):>14}'
          f' {row["pct_spend"]:>8.1f}%'
          f' {fmt_eur(row["median_value"]):>14}')

print()
print('INTERPRETATION:')
print('68.8% of lots are Micro or Small — these are the accessible, high-volume contracts.')
print('Large contracts (8.4% of lots) drive 95.0% of all spend.')
print('The procurement market is a dual economy: volume in small, value in large.')

# ── BY PROCUREMENT TYPE ─────────────────────────────────────────────────────
seg_type = (
    awards_seg.dropna(subset=['size_segment'])
    .groupby(['proc_type', 'size_segment'], observed=True)
    .agg(
        lot_count   = ('notice_id', 'count'),
        total_spend = ('cv_clean',  'sum')
    )
    .reset_index()
)
seg_type['pct_lots']  = seg_type.groupby('proc_type')['lot_count'].transform(
    lambda x: x / x.sum() * 100
)
seg_type['pct_spend'] = seg_type.groupby('proc_type')['total_spend'].transform(
    lambda x: x / x.sum() * 100
)

print('Contract size distribution by procurement type:')
print(f'{"Type":<12} {"Segment":<22} {"Lots":>8} {"% of Type Lots":>16} {"% of Type Spend":>17}')
print('-' * 80)
for _, row in seg_type.iterrows():
    print(f'{row["proc_type"]:<12} {str(row["size_segment"]):<22}'
          f' {int(row["lot_count"]):>8,}'
          f' {row["pct_lots"]:>15.1f}%'
          f' {row["pct_spend"]:>16.1f}%')

print()
print('KEY FINDING: Works contracts have the most evenly distributed size profile.')
print('Services and Supplies are dominated by many small lots with spend concentrated at top.')

# ── CHART: Size segment dual view ───────────────────────────────────────────
seg_colors = {
    'Micro (<€50K)':      C_TEAL,
    'Small (€50K–500K)':  C_GREEN,
    'Mid (€500K–5M)':     C_ORANGE,
    'Large (>€5M)':       C_RED,
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Contract Size Segmentation | January 2026', fontsize=13, fontweight='bold')

# Left: Overall lots vs spend
x = np.arange(len(labels))
width = 0.38
colors_seg = [seg_colors[l] for l in labels]
axes[0].bar(x - width/2, seg_overall['pct_lots'],  width, color=colors_seg, alpha=0.85,
            label='% of Lots', edgecolor='white')
axes[0].bar(x + width/2, seg_overall['pct_spend'], width, color=colors_seg, alpha=0.4,
            label='% of Spend', edgecolor='white')
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
axes[0].set_ylabel('Percentage (%)')
axes[0].set_title('Lots vs Spend by Size Segment (Overall)', fontweight='bold')
axes[0].legend()
for i, (pl, ps) in enumerate(zip(seg_overall['pct_lots'], seg_overall['pct_spend'])):
    axes[0].text(i - width/2, pl + 0.5, f'{pl:.1f}%', ha='center', fontsize=8)
    axes[0].text(i + width/2, ps + 0.5, f'{ps:.1f}%', ha='center', fontsize=8)

# Middle: % lots by type
proc_types = seg_type['proc_type'].unique()
bottom_lots = {pt: np.zeros(len(labels)) for pt in proc_types}
type_colors_map = {'Services': C_BLUE, 'Supplies': C_ORANGE, 'Works': C_GREEN}
x_pt = np.arange(len(proc_types))
for seg_label, color in zip(labels, colors_seg):
    vals_lots = []
    for pt in proc_types:
        row = seg_type[(seg_type['proc_type'] == pt) & (seg_type['size_segment'] == seg_label)]
        vals_lots.append(row['pct_lots'].values[0] if len(row) > 0 else 0)
    axes[1].bar(x_pt, vals_lots, bottom=[bottom_lots[pt][0] for pt in proc_types],
                color=color, alpha=0.85, label=seg_label, edgecolor='white')
    for i, pt in enumerate(proc_types):
        bottom_lots[pt][0] += vals_lots[i]
axes[1].set_xticks(x_pt)
axes[1].set_xticklabels(proc_types)
axes[1].set_ylabel('% of Lots')
axes[1].set_title('Lot Volume Distribution by Type & Segment', fontweight='bold')
axes[1].legend(loc='upper right', fontsize=8)

# Right: % spend by type
bottom_spend = {pt: np.zeros(len(labels)) for pt in proc_types}
for seg_label, color in zip(labels, colors_seg):
    vals_spend = []
    for pt in proc_types:
        row = seg_type[(seg_type['proc_type'] == pt) & (seg_type['size_segment'] == seg_label)]
        vals_spend.append(row['pct_spend'].values[0] if len(row) > 0 else 0)
    axes[2].bar(x_pt, vals_spend, bottom=[bottom_spend[pt][0] for pt in proc_types],
                color=color, alpha=0.85, label=seg_label, edgecolor='white')
    for i, pt in enumerate(proc_types):
        bottom_spend[pt][0] += vals_spend[i]
axes[2].set_xticks(x_pt)
axes[2].set_xticklabels(proc_types)
axes[2].set_ylabel('% of Spend')
axes[2].set_title('Spend Distribution by Type & Segment', fontweight='bold')
axes[2].legend(loc='lower right', fontsize=8)

plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------
# ## 12. Dominant Buyers Per Country — Who Controls Each National Market?
# > **Why this matters:** Section 10 showed that 11,417 buyers exist in the dataset,
# > but that spend is concentrated. This section goes one level deeper:
# > for each country, who is the single largest buyer, and how dominant are they?
# >
# > **KPI Definition — National Market Dominance:**
# > The share of a country's total awarded spend accounted for by its single largest buyer.
# > A high dominance ratio (>50%) means one institution controls most of that country's
# > procurement — a centralised market. A low ratio (<20%) means spend is spread
# > across many agencies — a decentralised market.
# >
# > **Key findings from the data:**
# > - The top buyer in Hungary (Digitális Kormányzati Ügynökség) accounts for €431.8B —
# >   a framework ceiling anomaly, not a real single-buyer dominance.
# > - **Ex-Hungary highlights:** Prague's public transport company (CZE) leads at €31.5B.
# >   France's GIP RESAH (a hospital group purchasing body) at €6.5B is a key healthcare
# >   procurement aggregator. Italy's Consip S.p.A. at €4.9B is a central purchasing body
# >   that buys on behalf of the entire Italian public sector — a single point of contact
# >   for selling to the Italian government.
# > - **Strategic implication:** Centralised buyers — Consip (Italy), GIP RESAH (France), Isavia (Iceland), Stedin Netbeheer (Netherlands) — are high-priority accounts. Winning one contract with them can unlock an entire national market.

# ── DOMINANT BUYER PER COUNTRY ──────────────────────────────────────────────
# We compute each buyer's total spend within their country,
# then identify the top buyer per country and their dominance ratio.

buyer_names = gold_awards[['notice_id','buyer_name','buyer_country']].drop_duplicates('notice_id')

buyer_country_spend = (
    notice_awards_clean
    .merge(buyer_names, on='notice_id', how='left', suffixes=('', '_raw'))
    .groupby(['buyer_country', 'buyer_name'])
    .agg(
        buyer_total  = ('awarded_eur_total', 'sum'),
        n_contracts  = ('notice_id',         'count')
    )
    .reset_index()
)

# Country totals
country_totals = (
    buyer_country_spend
    .groupby('buyer_country')['buyer_total']
    .sum()
    .reset_index()
    .rename(columns={'buyer_total': 'country_total'})
)

buyer_country_spend = buyer_country_spend.merge(country_totals, on='buyer_country')
buyer_country_spend['dominance_pct'] = (
    buyer_country_spend['buyer_total'] / buyer_country_spend['country_total'] * 100
)

# Top buyer per country
top_per_country = (
    buyer_country_spend
    .sort_values('buyer_total', ascending=False)
    .groupby('buyer_country')
    .first()
    .reset_index()
    [['buyer_country', 'buyer_name', 'buyer_total', 'n_contracts', 'country_total', 'dominance_pct']]
    .sort_values('buyer_total', ascending=False)
)

print('=== DOMINANT BUYER PER COUNTRY (Top 20 by buyer spend) ===')
print()
print(f'{"Country":<8} {"Buyer Spend":>14} {"Contracts":>11} {"Country Total":>16} {"Dominance %":>13}  Buyer Name')
print('-' * 105)
for _, row in top_per_country.head(20).iterrows():
    name = str(row['buyer_name'])[:40] if pd.notna(row['buyer_name']) else 'Unknown'
    dom  = f'{row["dominance_pct"]:.1f}%'
    flag = ' ⚠ centralised' if row['dominance_pct'] > 50 else ''
    print(f'{row["buyer_country"]:<8} {fmt_eur(row["buyer_total"]):>14}'
          f' {int(row["n_contracts"]):>11,}'
          f' {fmt_eur(row["country_total"]):>16}'
          f' {dom:>13}  {name}{flag}')

# ── EX-HUNGARY VIEW ─────────────────────────────────────────────────────────
buyer_country_spend_exhun = (
    notice_awards_exhun
    .merge(buyer_names, on='notice_id', how='left', suffixes=('', '_raw'))
    .groupby(['buyer_country', 'buyer_name'])
    .agg(
        buyer_total  = ('awarded_eur_total', 'sum'),
        n_contracts  = ('notice_id',         'count')
    )
    .reset_index()
)

country_totals_exhun = (
    buyer_country_spend_exhun
    .groupby('buyer_country')['buyer_total']
    .sum()
    .reset_index()
    .rename(columns={'buyer_total': 'country_total'})
)

buyer_country_spend_exhun = buyer_country_spend_exhun.merge(country_totals_exhun, on='buyer_country')
buyer_country_spend_exhun['dominance_pct'] = (
    buyer_country_spend_exhun['buyer_total'] / buyer_country_spend_exhun['country_total'] * 100
)

top_per_country_exhun = (
    buyer_country_spend_exhun
    .sort_values('buyer_total', ascending=False)
    .groupby('buyer_country')
    .first()
    .reset_index()
    [['buyer_country', 'buyer_name', 'buyer_total', 'n_contracts', 'country_total', 'dominance_pct']]
    .sort_values('buyer_total', ascending=False)
)

print('=== DOMINANT BUYER PER COUNTRY — Ex-Hungary (Top 20) ===')
print()
print(f'{"Country":<8} {"Buyer Spend":>14} {"Contracts":>11} {"Country Total":>16} {"Dominance %":>13}  Buyer Name')
print('-' * 105)
for _, row in top_per_country_exhun.head(20).iterrows():
    name = str(row['buyer_name'])[:40] if pd.notna(row['buyer_name']) else 'Unknown'
    dom  = f'{row["dominance_pct"]:.1f}%'
    flag = ' ⚠ centralised' if row['dominance_pct'] > 50 else ''
    print(f'{row["buyer_country"]:<8} {fmt_eur(row["buyer_total"]):>14}'
          f' {int(row["n_contracts"]):>11,}'
          f' {fmt_eur(row["country_total"]):>16}'
          f' {dom:>13}  {name}{flag}')

print()
print('Countries where top buyer controls >50% of national spend (centralised):')
centralised = top_per_country_exhun[top_per_country_exhun['dominance_pct'] > 50]
for _, row in centralised.iterrows():
    print(f'  {row["buyer_country"]}: {fmt_eur(row["buyer_total"])} ({row["dominance_pct"]:.1f}%) — {str(row["buyer_name"])[:50]}')

# ── CHART: Dominant buyer spend and dominance % ─────────────────────────────
plot_data = top_per_country_exhun.head(15).copy()
plot_data['short_name'] = plot_data['buyer_name'].apply(
    lambda x: str(x)[:28] + '…' if pd.notna(x) and len(str(x)) > 28 else str(x)
)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('National Market Dominance — Top Buyer per Country | Ex-Hungary | January 2026',
             fontsize=13, fontweight='bold')

# Left: buyer spend bar
bar_colors = [C_RED if row['dominance_pct'] > 50 else C_BLUE
              for _, row in plot_data.iterrows()]
bars = axes[0].barh(
    plot_data['buyer_country'][::-1].tolist(),
    plot_data['buyer_total'][::-1].tolist(),
    color=bar_colors[::-1], alpha=0.85, edgecolor='white'
)
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(fmt_eur))
axes[0].set_xlabel('Top Buyer Total Spend (EUR)')
axes[0].set_title('Spend by Top Buyer per Country', fontweight='bold')
for bar, val in zip(bars, plot_data['buyer_total'][::-1]):
    axes[0].text(val * 1.01, bar.get_y() + bar.get_height()/2,
                 fmt_eur(val), va='center', fontsize=8)

# Right: dominance % bar
dom_colors = [C_RED if d > 50 else C_ORANGE if d > 30 else C_TEAL
              for d in plot_data['dominance_pct']]
bars2 = axes[1].barh(
    [f"{row['buyer_country']}: {row['short_name']}" for _, row in plot_data.iterrows()][::-1],
    plot_data['dominance_pct'][::-1].tolist(),
    color=dom_colors[::-1], alpha=0.85, edgecolor='white'
)
axes[1].axvline(50, color='black', linestyle='--', linewidth=1, alpha=0.6, label='50% dominance')
axes[1].set_xlabel('Top Buyer Share of Country Spend (%)')
axes[1].set_title('National Market Dominance % (>50% = centralised)', fontweight='bold')
axes[1].set_xlim(0, 105)
axes[1].legend()
for bar, val in zip(bars2, plot_data['dominance_pct'][::-1]):
    axes[1].text(val + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}%', va='center', fontsize=8)

red_patch   = mpatches.Patch(color=C_RED,    alpha=0.85, label='>50% — Centralised')
orange_patch= mpatches.Patch(color=C_ORANGE, alpha=0.85, label='30–50% — Moderately concentrated')
teal_patch  = mpatches.Patch(color=C_TEAL,   alpha=0.85, label='<30% — Distributed')
axes[1].legend(handles=[red_patch, orange_patch, teal_patch], loc='lower right', fontsize=8)

plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------
# ## Summary of Key Findings — Notebook 2: Value & Financial Analysis
# **Data: January 2026 | TED EU Public Procurement | Notice-Level (Romania lot-normalised)**
#
# ----------------------------------------------------------------------
#
# ### 1. The Null Audit (Section 1)
# - `estimated` is 57.2% null across gold_notices — confirmed from NB1.
# - Nulls are NOT evenly distributed. Sweden leads at 97.7% estimated value coverage;
#   Germany reports only 14.7% despite being the largest market by notice count.
# - **Switzerland (CHE) is a special case:** 752 notices, 0.0% estimated value coverage.
#   Switzerland is not an EU member and its procurement law does not require estimated
#   values to be published. Swiss contracts must be excluded from any savings analysis.
# - `contract_value` in gold_awards (93.3% lot coverage) was selected as the primary
#   financial metric. Capped at €5B per lot.
#
# ----------------------------------------------------------------------
#
# ### 2. The Hungary Framework Value Problem (Section 3b)
# - Hungary appeared to account for 44% of total EU spend (€350.6B) from only 509 notices.
# - **Investigation:** Pipeline flags (`result_code`, `is_awarded`) could not distinguish
#   framework ceilings from actual awards. 36.4% of Hungary's big lots are exact multiples
#   of €1B — including 29 contracts worth exactly €5B each.
# - **Decision:** Hungary excluded from all spend, ACV, leaderboard, and buyer analysis.
# - **Ex-Hungary total awarded spend: €445.0B** — this is the figure cited to Microsoft.
#
# ----------------------------------------------------------------------
#
# ### 3. Total Spend Landscape (Section 3 + 3b)
# - Ex-Hungary top 5: CZE (29.6%), ROU (21.0%), POL (11.2%), SWE (11.2%), FRA (8.1%) — 81.2% of spend.
# - Germany and Spain are underrepresented due to low contract_value coverage rates.
#
# ----------------------------------------------------------------------
#
# ### 4. Average Contract Size (Section 4)
# - Universal right-skew across all countries — a small number of mega-deals pull the mean up.
# - **Ex-Hungary benchmark: median €680K, mean €16.5M** — always use median as the 'typical' contract.
#
# ----------------------------------------------------------------------
#
# ### 5. Savings & Estimation Accuracy (Section 5)
# - **Median savings: 48.9%** across 13,232 matched contracts — buyers systematically over-estimate by ~2x.
# - Slovakia (96.9%), Croatia (88.6%), Czech Republic (83.1%) are the biggest over-estimators.
# - 78.1% of contracts came in below estimate; only 8.9% exceeded budget.
# - By type: Supplies has the highest savings (56.9%), Works the lowest (28.3%).
#
# ----------------------------------------------------------------------
#
# ### 6. Value Concentration (Section 6)
# - Extreme Pareto: **top 3% of contracts (>€100M each) = 84.8% of all spend**.
# - Most contracts by volume (87.6%) are in the €100K–€10M range.
#
# ----------------------------------------------------------------------
#
# ### 7. CPV Categories by Value (Section 8)
# - **CPV 72 (IT services): €88.9B, 11.2% of total spend, 1,439 contracts** — 3rd largest category.
# - **CPV 48 (Software): €26.6B, 3.3%, 659 contracts** — 9th largest.
# - **Combined IT+Software: €115.5B** — the single largest sector when combined.
#
# ----------------------------------------------------------------------
#
# ### 8. Mega-Deals & Buyer Concentration (Sections 9–10)
# - **Microsoft spotlight:** Largest IT/analytics contract ex-Hungary: Czech BI/AA framework at **€4.5B** (CPV 72).
# - **Full dataset (HUN-inflated):**  Top 1 buyer = 32.6% of spend. Only 5 buyers reach 50%; 79 buyers reach 80% — extreme concentration driven by Hungarian framework ceilings, not real market structure.
# - **Ex-Hungary top buyer:** Dopravní podnik hl. m. Prahy, Prague public transport (CZE) at €30.5B — 174 contracts, the highest-volume institutional buyer outside Hungary.
#
# ----------------------------------------------------------------------
#
# ### 9. Contract Size Segmentation (Section 11)
# - **Dual-economy market structure confirmed:**
#   - 68.8% of lots are Micro or Small (< €500K) — the accessible, high-volume end
#   - Large contracts (>€5M, 8.4% of lots) account for **95.0% of all spend**
# - By type: Works is the most evenly distributed (21.1% of lots are Large).
#   Services and Supplies have heavy small-lot volume but spend concentrated at the top.
# - **For Person 5:** Show this as a dual-axis chart or split panel — count vs value by segment tells completely different stories.
#
# ----------------------------------------------------------------------
#
# ### 10. National Market Dominance (Section 12)
# - **Centralised markets** (top buyer >50% of national spend):
#   - Key examples ex-Hungary: ISL (Isavia, airport authority), NLD (Stedin Netbeheer), NOR (Statnett)
#   - In these markets, one buyer relationship unlocks the whole country.
# - **Key institutional buyers to know:**
#   - **Italy — Consip S.p.A.:** Central purchasing body for the entire Italian public sector.
#     Winning a Consip framework = selling to all of Italy's government agencies.
#   - **France — GIP RESAH:** Hospital group purchasing body; largest French buyer at €6.5B.
#   - **Czech Republic — Prague Transport (DPP):** High-volume buyer (174 contracts), dominant at €31.5B.
# - **For Person 5:** A map or bubble chart of top buyers by country with dominance % as bubble size.
#
# ----------------------------------------------------------------------
#
# ### KPI Sensitivity to Hungary Exclusion
# | KPI | Hungary-Sensitive? | Recommended View |
# |-----|-------------------|------------------|
# | Total awarded spend | ✅ Yes | Ex-Hungary |
# | Country spend ranking | ✅ Yes | Ex-Hungary |
# | Average contract size (mean) | ✅ Yes | Ex-Hungary |
# | Median contract size (€680K) | ⚠ Partially | Ex-Hungary preferred |
# | Savings % by country | ❌ No | Full dataset |
# | Value concentration (Pareto) | ⚠ Partially | Flag in presentation |
# | CPV spend ranking | ⚠ Partially | Flag in presentation |
# | Mega-deal leaderboard | ✅ Yes | Ex-Hungary |
# | Buyer concentration | ✅ Yes | Ex-Hungary |
# | Contract size segmentation | ⚠ Minor | Either (HUN skews Large segment) |
# | National dominance % | ✅ Yes | Ex-Hungary |
# | Null audit / coverage rates | ❌ No | Full dataset |
#
# ----------------------------------------------------------------------
#
# ### Implications for Notebook 3 (Competition & SME Analysis)
# - Use `notice_awards_exhun` as the base dataset for all competition and SME metrics.
# - The contract size segments from Section 11 are key inputs for NB3:
#   **SME win rates should be computed per segment** — SMEs are unlikely to win Large (>€5M)
#   contracts, but may dominate Micro and Small tiers.
# - The centralised buyers identified in Section 12 are likely to have very different
#   competition intensity profiles — a Consip framework attracts many more bidders than
#   a small municipal authority.
