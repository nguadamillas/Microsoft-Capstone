
# # Notebook 1 — Volume & Market Structure
# **TED Procurement Intelligence | IE University × Microsoft**
# **Analytics + Semantic Layer | Data: January 2026**
#
# ----------------------------------------------------------------------
# ### What this notebook covers
# 1. Dataset scope (total notices, countries, buyers, CPV divisions)
# 2. CN vs CAN split — open tenders vs awarded contracts
# 3. Notice volume by country — all notices, CN only, CAN only
# 4. Procurement type distribution — Services / Supplies / Works
# 5. Procurement type mix by country
# 6. Notice publication activity within January 2026
# 7. CN-to-CAN ratio — data coverage diagnostic per country
#
# > **Note:** Data covers January 2026 only. All date-based analysis is within this month.
#


# ## 0. Setup — Mount Drive & Load Data
#

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

# Discover all parquet files dynamically
parquet_files = {}
for root, dirs, files in os.walk(EXTRACT_PATH):
    for f in files:
        if f.endswith('.parquet'):
            parquet_files[f.replace('.parquet', '')] = os.path.join(root, f)

print(f'Found {len(parquet_files)} parquet files:')
for name, path in sorted(parquet_files.items()):
    print(f'  {name}')


# Load tables
gold_notices       = pd.read_parquet(parquet_files['gold_notices'])
gold_opportunities = pd.read_parquet(parquet_files['gold_opportunities'])
gold_awards        = pd.read_parquet(parquet_files['gold_awards'])
gold_country_kpis  = pd.read_parquet(parquet_files['gold_country_kpis'])

# Parse dates
for df in [gold_notices, gold_opportunities, gold_awards]:
    if 'pub_date' in df.columns:
        df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce')

print(f'gold_notices:       {len(gold_notices):>7,} rows')
print(f'gold_opportunities: {len(gold_opportunities):>7,} rows')
print(f'gold_awards:        {len(gold_awards):>7,} rows')
print(f'gold_country_kpis:  {len(gold_country_kpis):>7,} rows')
print()
print('Date range in gold_notices:')
print(f'  Min: {gold_notices["pub_date"].min()}')
print(f'  Max: {gold_notices["pub_date"].max()}')



# ----------------------------------------------------------------------
# ## 1. Dataset Scope
#

print('=== DATASET SCOPE ===')
print(f'Total notices (gold_notices):       {len(gold_notices):,}')
print(f'  Unique notice IDs:                {gold_notices["notice_id"].nunique():,}')
print(f'  Unique countries:                 {gold_notices["buyer_country"].nunique():,}')
print(f'  Unique buyers:                    {gold_notices["buyer_name"].nunique():,}')
print(f'  Unique CPV divisions:             {gold_notices["cpv_division"].nunique():,}')
print()
print(f'Open tenders (gold_opportunities):  {len(gold_opportunities):,}')
print(f'  Unique notice IDs:                {gold_opportunities["notice_id"].nunique():,}')
print()
print(f'Awarded contracts (gold_awards):    {len(gold_awards):,} rows (lot-level)')
print(f'  Unique notice IDs:                {gold_awards["notice_id"].nunique():,}')
print(f'  Unique countries:                 {gold_awards["buyer_country"].nunique():,}')
print()
print('Notice types found in gold_notices:')
print(gold_notices['notice_type'].value_counts().to_string())


# Null rates for key fields
key_fields = ['notice_type','pub_date','buyer_country','buyer_name',
              'proc_type','cpv_code','cpv_division_name','estimated']
print('Null rates — gold_notices:')
for col in key_fields:
    if col in gold_notices.columns:
        n   = gold_notices[col].isna().sum()
        pct = gold_notices[col].isna().mean()*100
        print(f'  {col:<25} {n:>7,} null  ({pct:.1f}%)')



# ----------------------------------------------------------------------
# ## 2. CN vs CAN Split
# > `notice_type` column. CN = open tender. CAN = awarded contract.
#

type_counts = gold_notices['notice_type'].value_counts()
print('Notice type counts:')
print(type_counts.to_string())
print()
for t, c in type_counts.items():
    print(f'  {t}: {c:,}  ({c/len(gold_notices)*100:.1f}%)')


# PIN = Prior Information Notice — buyer announces upcoming procurement before publishing the full tender
# These are early market signals, not yet open for bidding
pin_notices = gold_notices[gold_notices['notice_type'] == 'PIN']
print(f'PIN notices: {len(pin_notices):,}')
print(f'Countries using PINs:')
print(pin_notices['buyer_country'].value_counts().head(10).to_string())
print()
print(f'CPV divisions in PINs:')
print(pin_notices['cpv_division_name'].value_counts().head(10).to_string())

colors_map = {'CAN': C_ORANGE, 'CN': C_TEAL}
colors_bar = [colors_map.get(t, C_GREY) for t in type_counts.index]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle('Notice Type Distribution — January 2026', fontsize=13, fontweight='bold')

bars = axes[0].bar(type_counts.index, type_counts.values,
                   color=colors_bar, edgecolor='white', linewidth=1.5)
axes[0].set_ylabel('Number of Notices')
axes[0].set_title('Count by Notice Type')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
for bar, val in zip(bars, type_counts.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+100,
                 f'{val:,}', ha='center', fontsize=10, fontweight='bold')

wedges, texts, autotexts = axes[1].pie(
    type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
    colors=colors_bar, pctdistance=0.75, startangle=90,
    wedgeprops={'width':0.6,'edgecolor':'white','linewidth':2})
for at in autotexts: at.set_fontsize(11); at.set_fontweight('bold')
axes[1].set_title('Percentage Split')
plt.tight_layout()
plt.show()



# ----------------------------------------------------------------------
# ## 3. Notice Volume by Country
# > Using `gold_notices` (notice-level, deduplicated). `gold_awards` is lot-level — one notice can produce many rows there.
#

# Volume from gold_notices — notice level
vol_all = gold_notices['buyer_country'].value_counts()
vol_cn  = gold_notices[gold_notices['notice_type']=='CN']['buyer_country'].value_counts()
vol_can = gold_notices[gold_notices['notice_type']=='CAN']['buyer_country'].value_counts()

print('Top 20 countries — All Notices:')
print(vol_all.head(20).to_string())
print()
print('Top 20 countries — CN (Open Tenders):')
print(vol_cn.head(20).to_string())
print()
print('Top 20 countries — CAN (Awards):')
print(vol_can.head(20).to_string())


fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle('Notice Volume by Country — Top 15 | January 2026', fontsize=13, fontweight='bold')

for ax, (data, title, color) in zip(axes, [
    (vol_all, 'All Notices',              C_BLUE),
    (vol_cn,  'CN — Open Tenders',        C_TEAL),
    (vol_can, 'CAN — Awarded Contracts',  C_ORANGE),
]):
    top15 = data.head(15)
    ax.barh(top15.index[::-1], top15.values[::-1], color=color, alpha=0.85, edgecolor='white')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Number of Notices')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
    for i, val in enumerate(top15.values[::-1]):
        ax.text(val + max(top15.values)*0.01, i, f'{val:,}', va='center', fontsize=8)

plt.tight_layout()
plt.show()


# Cross-check: gold_country_kpis (pre-aggregated by pipeline)
ckpi = gold_country_kpis.sort_values('notice_count', ascending=False)
print('Top 20 countries — gold_country_kpis (pre-aggregated):')
print(ckpi[['buyer_country','notice_count','contract_notice_count','award_notice_count']].head(20).to_string(index=False))


# Lot-level vs notice-level comparison per country
# gold_awards is lot-level — shows how many lots each country's notices contain
lot_vs_notice = (gold_awards.groupby('buyer_country')
                 .agg(award_rows=('notice_id','count'),
                      unique_notices=('notice_id','nunique'))
                 .reset_index())
lot_vs_notice['lots_per_notice'] = lot_vs_notice['award_rows'] / lot_vs_notice['unique_notices']
lot_vs_notice = lot_vs_notice.sort_values('award_rows', ascending=False)

print('Lot-level rows vs unique notices in gold_awards (top 20):')
print(lot_vs_notice.head(20).round(2).to_string(index=False))



# ----------------------------------------------------------------------
# ## 4. Procurement Type Distribution — Services / Supplies / Works
#

proc_all = gold_notices['proc_type'].value_counts()
proc_cn  = gold_opportunities['proc_type'].value_counts()
proc_can = gold_awards['proc_type'].value_counts()

print('Procurement type — All Notices:')
print(proc_all.to_string())
print()
print('Procurement type — CN (Open Tenders):')
print(proc_cn.to_string())
print()
print('Procurement type — CAN (Awards):')
print(proc_can.to_string())

comparison = pd.DataFrame({'All': proc_all, 'CN': proc_cn, 'CAN': proc_can}).fillna(0).astype(int)
print()
print('Side-by-side comparison:')
print(comparison.to_string())


colors_proc = {'Services': C_BLUE, 'Supplies': C_TEAL, 'Works': C_ORANGE}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Procurement Type Distribution — January 2026', fontsize=13, fontweight='bold')

for ax, (data, title) in zip(axes, [
    (proc_all, 'All Notices'),
    (proc_cn,  'CN — Open Tenders'),
    (proc_can, 'CAN — Awarded'),
]):
    clrs = [colors_proc.get(x, C_GREY) for x in data.index]
    wedges, texts, autotexts = ax.pie(
        data.values, labels=data.index, autopct='%1.1f%%', colors=clrs,
        startangle=90, wedgeprops={'width':0.6,'edgecolor':'white','linewidth':2},
        pctdistance=0.75)
    for at in autotexts: at.set_fontsize(10); at.set_fontweight('bold')
    ax.set_title(title, fontweight='bold')

plt.tight_layout()
plt.show()



# ----------------------------------------------------------------------
# ## 5. Procurement Type Mix by Country
# > What each country procures — Services, Supplies, or Works.
#

top15_countries = vol_all.head(15).index
proc_ctry = (gold_notices[gold_notices['buyer_country'].isin(top15_countries)]
             .groupby(['buyer_country','proc_type']).size().unstack(fill_value=0))
proc_ctry_pct = proc_ctry.div(proc_ctry.sum(axis=1), axis=0) * 100
proc_ctry_pct = proc_ctry_pct.loc[vol_all.head(15).index]  # keep top-15 order

print('Procurement type % by country (top 15):')
print(proc_ctry_pct.round(1).to_string())


fig, ax = plt.subplots(figsize=(12, 7))
clrs = [colors_proc.get(c, C_GREY) for c in proc_ctry_pct.columns]
proc_ctry_pct.plot(kind='bar', ax=ax, color=clrs, edgecolor='white', linewidth=0.5, width=0.75)
ax.set_title('Procurement Type Mix by Country (Top 15) — % of Notices | Jan 2026', fontweight='bold')
ax.set_ylabel('% of Notices')
ax.set_ylim(0, 105)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x:.0f}%'))
plt.xticks(rotation=45, ha='right')
plt.legend(title='Type', bbox_to_anchor=(1.01, 1))
plt.tight_layout()
plt.show()


# The flip: Supplies goes from 37.3% in CN to 64.4% in CAN
# This means Supplies contracts are awarded at much higher rates than opened
# Possible explanations: framework agreements (one CN opens many supply lots),
# faster award cycles for supplies vs services
# Let's check: what is the average lots per notice broken down by proc_type?

lots_by_proc = (gold_awards.groupby('proc_type')
                .agg(total_rows=('notice_id','count'),
                     unique_notices=('notice_id','nunique'))
                .reset_index())
lots_by_proc['lots_per_notice'] = lots_by_proc['total_rows'] / lots_by_proc['unique_notices']
print('Lot structure by procurement type:')
print(lots_by_proc.to_string(index=False))
print()
print('This explains the CN→CAN flip:')
print('If Supplies notices have more lots per notice, they generate')
print('disproportionately more CAN rows than CN rows.')


# ----------------------------------------------------------------------
# ## 6. Notice Publication Activity — Within January 2026
# > Daily notice count within January. Shows which days had highest procurement activity.
#

# Filter strictly to January 2026
jan_notices = gold_notices[
    (gold_notices['pub_date'] >= '2026-01-01') &
    (gold_notices['pub_date'] <= '2026-01-31')
].copy()

print(f'Notices in January 2026: {len(jan_notices):,}')
print(f'Unique dates:             {jan_notices["pub_date"].nunique()}')
print(f'Date range:               {jan_notices["pub_date"].min().date()} to {jan_notices["pub_date"].max().date()}')
print()
print('Notices outside January 2026 (if any):')
outside = gold_notices[
    (gold_notices['pub_date'] < '2026-01-01') |
    (gold_notices['pub_date'] > '2026-01-31')
]
print(f'  {len(outside):,} rows — dates: {outside["pub_date"].unique()[:10]}')


# The 2,460 notices outside January 2026
outside_jan = gold_notices[
    (gold_notices['pub_date'] < '2026-01-01') |
    (gold_notices['pub_date'] > '2026-01-31')
]
print('Notices outside January 2026 by year-month:')
print(outside_jan['pub_date'].dt.to_period('M').value_counts().sort_index().to_string())
print()
print('Countries with most notices outside January 2026:')
print(outside_jan['buyer_country'].value_counts().head(10).to_string())

daily = jan_notices.groupby('pub_date').size().reset_index(name='count')
daily_cn  = jan_notices[jan_notices['notice_type']=='CN'].groupby('pub_date').size().reset_index(name='count')
daily_can = jan_notices[jan_notices['notice_type']=='CAN'].groupby('pub_date').size().reset_index(name='count')

print('Daily notice counts — January 2026:')
print(daily.to_string(index=False))


fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle('Daily Notice Publications — January 2026', fontsize=13, fontweight='bold')

for ax, (data, label, color) in zip(axes, [
    (daily,     'All Notices',       C_BLUE),
    (daily_cn,  'CN — Open Tenders', C_TEAL),
    (daily_can, 'CAN — Awards',      C_ORANGE),
]):
    if len(data) > 0:
        ax.bar(data['pub_date'], data['count'], color=color, alpha=0.85, edgecolor='white', width=0.6)
        ax.set_ylabel('Notices / Day')
        ax.set_title(label, fontweight='bold', fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{int(x):,}'))

axes[-1].set_xlabel('Date (January 2026)')
plt.tight_layout()
plt.show()


# Day-of-week pattern within January 2026
jan_notices['day_of_week'] = jan_notices['pub_date'].dt.day_name()
jan_notices['week_num']    = jan_notices['pub_date'].dt.isocalendar().week

dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
dow_counts = jan_notices['day_of_week'].value_counts().reindex(dow_order, fill_value=0)

print('Notices by day of week — January 2026:')
print(dow_counts.to_string())
print()
print('Notices by week number:')
print(jan_notices.groupby('week_num').size().to_string())


fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(dow_counts.index, dow_counts.values, color=C_BLUE, edgecolor='white', alpha=0.85)
ax.set_title('Notices by Day of Week — January 2026', fontweight='bold')
ax.set_ylabel('Number of Notices')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
for i, val in enumerate(dow_counts.values):
    ax.text(i, val + max(dow_counts.values)*0.01, f'{val:,}', ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.show()



# ----------------------------------------------------------------------
# ## 7. CN-to-CAN Ratio — Data Coverage Diagnostic
# > CAN/CN ratio per country. Not a business KPI — a data quality check. Low ratio means fewer awarded contracts were captured relative to open tenders for that country.
#

cn_by  = gold_notices[gold_notices['notice_type']=='CN']['buyer_country'].value_counts()
can_by = gold_notices[gold_notices['notice_type']=='CAN']['buyer_country'].value_counts()

ratio_df = pd.DataFrame({'cn_count': cn_by, 'can_count': can_by}).fillna(0).astype(int)
ratio_df['total'] = ratio_df['cn_count'] + ratio_df['can_count']
ratio_df['can_to_cn_ratio'] = np.where(
    ratio_df['cn_count'] > 0,
    ratio_df['can_count'] / ratio_df['cn_count'],
    np.nan
)
ratio_df['coverage'] = pd.cut(
    ratio_df['can_to_cn_ratio'],
    bins=[-np.inf, 0.3, 0.7, np.inf],
    labels=['Low <0.3', 'Medium 0.3-0.7', 'Good >0.7']
)
ratio_df = ratio_df[ratio_df['total'] >= 5].sort_values('can_to_cn_ratio')

print('CN-to-CAN ratio by country (countries with >= 5 total notices):')
print(ratio_df.round(3).to_string())
print()
print('Coverage band summary:')
print(ratio_df['coverage'].value_counts().to_string())


color_map  = {'Low <0.3': C_RED, 'Medium 0.3-0.7': C_ORANGE, 'Good >0.7': C_GREEN}
bar_colors = [color_map.get(str(f), C_GREY) for f in ratio_df['coverage']]

fig, ax = plt.subplots(figsize=(12, 10))
ax.barh(ratio_df.index, ratio_df['can_to_cn_ratio'], color=bar_colors, edgecolor='white')
ax.axvline(0.3, color=C_RED,   linestyle='--', linewidth=1.2, label='Low threshold (0.3)')
ax.axvline(0.7, color=C_GREEN, linestyle='--', linewidth=1.2, label='Good threshold (0.7)')
ax.axvline(1.0, color='black', linestyle='-',  linewidth=0.8, alpha=0.4, label='Parity (1.0)')
ax.set_xlabel('CAN / CN Ratio')
ax.set_title('CN-to-CAN Coverage Ratio by Country — January 2026', fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()


# Countries with ratio > 1.0 means more CANs than CNs
# This is NOT a data quality issue — it happens because one CN can generate
# multiple CAN notices (one per lot awarded)
# High ratios = countries that use many lots per notice

above_parity = ratio_df[ratio_df['can_to_cn_ratio'] > 1.0].sort_values('can_to_cn_ratio', ascending=False)
print('Countries with CAN > CN (ratio > 1.0) — lot-based procurement:')
print(above_parity[['cn_count','can_count','can_to_cn_ratio']].to_string())
print()
# Cross-check with lots_per_notice from the lot comparison table
lot_vs_notice = (gold_awards.groupby('buyer_country')
                 .agg(award_rows=('notice_id','count'),
                      unique_notices=('notice_id','nunique'))
                 .reset_index())
lot_vs_notice['lots_per_notice'] = lot_vs_notice['award_rows'] / lot_vs_notice['unique_notices']
high_lots = lot_vs_notice[lot_vs_notice['lots_per_notice'] > 2].sort_values('lots_per_notice', ascending=False)
print('Countries with avg lots per notice > 2 (explains high CAN/CN ratio):')
print(high_lots.to_string(index=False))


# ----------------------------------------------------------------------
# ## Summary of Key Findings — Notebook 1: Volume & Market Structure
# **Data: January 2026 | TED EU Public Procurement**
#
# ----------------------------------------------------------------------
#
# ### 1. Dataset Scope
# - **71,432 total notices** in gold_notices, of which **68,972 fall within January 2026**
# - **2,460 notices** fall outside January 2026 (mostly late December 2025) — these are included in gold_notices but excluded from date-filtered analysis
# - **62 countries**, **21,955 unique buyers**, **45 CPV divisions**
# - gold_awards contains **97,913 rows** but only **32,053 unique notices** — it is a lot-level table, not a notice-level table
#
# ----------------------------------------------------------------------
#
# ### 2. Notice Types — Four Types, Not Two
# - **CAN (Contract Award Notice):** 39,773 — 55.7% — completed contracts
# - **CN (Contract Notice):** 29,537 — 41.3% — open tenders
# - **PIN (Prior Information Notice):** 2,110 — 3.0% — advance market signals, not yet open for bidding
# - **BusinessRegistrationInformationNotice:** 12 — negligible
# - All financial and competition analysis uses CAN notices only (gold_awards)
#
# ----------------------------------------------------------------------
#
# ### 3. Country Volume
# - **Germany leads open tenders (CN):** 6,195 — most active market for new procurement opportunities
# - **Poland leads awarded contracts (CAN):** 6,160 — most active by completed contracts
# - Top 5 by total notices: DEU (12,150), POL (11,062), FRA (7,739), CZE (5,398), ESP (5,110)
# - These 5 countries account for the majority of EU procurement activity in January 2026
#
# ----------------------------------------------------------------------
#
# ### 4. The Lot Structure Finding — Romania and the gold_awards Distortion
# - Romania has **3,526 notices** (6th in Europe) but **29,447 rows in gold_awards** — an average of **14.31 lots per notice**
# - This is 4x higher than the next country (Moldova: 7.4, Slovenia: 3.69, Poland: 3.55)
# - **Any metric computed from gold_awards row counts is heavily Romania-weighted**
# - For cross-country comparisons, notice-level metrics from gold_notices are more reliable than lot-level metrics from gold_awards
#
# ----------------------------------------------------------------------
#
# ### 5. The CN→CAN Procurement Type Flip
# - In **CN (open tenders):** Services 43.1%, Supplies 37.3%, Works 19.7%
# - In **CAN (awarded contracts):** Supplies 64.4%, Services 29.5%, Works 6.1%
# - This flip is **not** because Supplies is awarded more frequently
# - It is explained entirely by **lot structure**: Supplies notices average **4.69 lots per notice** vs Services at 1.92 and Works at 1.68
# - One Supplies CN generates ~5 CAN rows; one Services CN generates ~2 CAN rows
# - **Correct interpretation:** For true procurement type distribution, use CN notices (gold_opportunities), not CAN rows (gold_awards)
#
# ----------------------------------------------------------------------
#
# ### 6. The CN-to-CAN Ratio — Structural, Not a Data Quality Issue
# - Most countries have a CAN/CN ratio **above 1.0** — more awards than tenders
# - This is not missing data — it is caused by lot splitting: one CN generates multiple CANs
# - Countries with highest ratios (SRB 4.5, MNE 4.0, BGR 3.78, ROU 3.20) use the most lots per notice
# - **Only MKD** has a genuinely low ratio (0.076) suggesting limited award data coverage
# - The ratio chart should be interpreted as a **lot structure indicator**, not a data completeness indicator
#
# ----------------------------------------------------------------------
#
# ### 7. Publication Patterns — Within January 2026
# - **Weekdays only:** Monday–Friday average ~13,600–14,500 notices per day
# - **Weekends are essentially zero:** Saturday 397, Sunday 365 — procurement publishing is a working-day activity
# - **Week 1 (Jan 1–4) significantly lower:** Only 1,487 notices — New Year holiday effect
# - From **January 5 onwards**, activity is consistent at ~3,500–4,000 notices per working day
# - **29 unique publication dates** in January 2026, all within working days
#
# ----------------------------------------------------------------------
#
# ### Implications for Notebooks 2 and 3
# - All financial analysis (NB2) should be aware that Romania's totals in gold_awards reflect 14.31 lots per notice — notice-level normalisation is needed for fair country comparison
# - Procurement type analysis should use **gold_notices** as the source of truth, not gold_awards
# - Competition and savings metrics (NB3) computed from gold_awards rows are lot-weighted — high-lot countries like Romania are overrepresented in EU averages
