
# # Notebook 3 — Competition Analysis
# **TED Procurement Intelligence | IE University × Microsoft**
# **Analytics + Semantic Layer | Data: January 2026**
#
# ----------------------------------------------------------------------
# ### What this notebook covers
# 1. Competition intensity audit — null rates and coverage of `tenders_count`
# 2. Competition intensity by country — which markets attract the most bidders?
# 3. Competition intensity by CPV category — which categories are most contested?
# 4. Competition intensity by procurement type — Services vs Supplies vs Works
# 5. Competition vs savings — do more bidders produce lower final prices?
# 6. Competition by contract size segment — does value tier affect competitive intensity?
# 7. Market Competition Ranking — country-level competition profile
# 8. Microsoft spotlight — CPV 72 and CPV 48 competition profile
#
# > **Key rules carried forward from NB1 and NB2:**
# > - All country-level metrics use notice-level aggregation (`groupby notice_id` first). Romania has 14.31 lots per notice — raw row sums inflate its totals by ~14x.
# > - Hungary is excluded from all financial metrics. Competition metrics (tenders_count only, no EUR values) may include Hungary — flagged explicitly where relevant.
# > - Switzerland has 0% estimated value coverage — excluded from any metric that involves the `estimated` field.
# > - `tenders_count` is a lot-level field. It is averaged across lots when collapsed to notice-level.
# > - The four contract size tiers (Micro / Small / Mid / Large) match the segmentation defined in NB2 Section 11 for cross-notebook consistency.


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

print(f'gold_notices:       {len(gold_notices):>7,} rows')
print(f'gold_opportunities: {len(gold_opportunities):>7,} rows')
print(f'gold_awards:        {len(gold_awards):>7,} rows')
print(f'gold_country_kpis:  {len(gold_country_kpis):>7,} rows')
print(f'gold_cpv_analysis:  {len(gold_cpv_analysis):>7,} rows')
print()
print('gold_awards columns:')
print(list(gold_awards.columns))
print()
print('gold_opportunities columns:')
print(list(gold_opportunities.columns))

CAP = 5_000_000_000
gold_awards['cv_clean'] = gold_awards['contract_value'].clip(upper=CAP)

print(f'tenders_count column present: {"tenders_count" in gold_awards.columns}')
print()
print('All gold_awards columns:')
print(list(gold_awards.columns))

notice_awards = (
    gold_awards
    .groupby(['notice_id', 'buyer_country', 'proc_type', 'cpv_division', 'cpv_division_name'])
    .agg(
        awarded_eur_total = ('cv_clean',      'sum'),
        lot_count         = ('notice_id',     'count'),
        tenders_count_avg = ('tenders_count', 'mean'),
        tenders_count_max = ('tenders_count', 'max'),
    )
    .reset_index()
)

notice_awards_clean = notice_awards[notice_awards['awarded_eur_total'] > 0].copy()
notice_awards_exhun = notice_awards_clean[notice_awards_clean['buyer_country'] != 'HUN'].copy()

print()
print('=== DATASETS READY FOR NB3 ===')
print(f'notice_awards (all, notice-level):         {len(notice_awards):>7,}')
print(f'notice_awards_clean (awarded_eur > 0):     {len(notice_awards_clean):>7,}')
print(f'notice_awards_exhun (ex-Hungary):          {len(notice_awards_exhun):>7,}')
print()
print('tenders_count_avg null rate:', f"{notice_awards['tenders_count_avg'].isna().mean()*100:.1f}%")


# ----------------------------------------------------------------------
# ## 1. Competition Intensity Audit — Null Rates and Coverage
# > **Why this comes first:** Before drawing conclusions about competition levels, we must
# > understand how complete the `tenders_count` field is. If it is systematically missing
# > in certain countries or categories, those will appear artificially low in competition
# > metrics — the same problem the `estimated` field had in NB2.
# >
# > **What we are checking:**
# > - Overall null rate of `tenders_count` in gold_awards (lot-level)
# > - Null rate by country — are some countries never reporting bidder counts?
# > - Null rate by procurement type — do certain types report more reliably?
# > - Null rate by CPV division — are Microsoft-relevant categories well-covered?
# >
# > **Coverage note:** France (55.8%) and Italy report below the dataset average, meaning
# > competition scores for these markets reflect roughly half their actual tender population.
# > Their rankings should be interpreted with this caveat in mind.

n_lots_total = len(gold_awards)
n_tc_valid   = gold_awards['tenders_count'].notna().sum()
n_tc_null    = gold_awards['tenders_count'].isna().sum()
n_tc_zero    = (gold_awards['tenders_count'] == 0).sum()

print('=== TENDERS_COUNT NULL AUDIT (lot-level) ===')
print(f'Total lots in gold_awards:                 {n_lots_total:>7,}')
print(f'Lots with valid tenders_count:             {n_tc_valid:>7,}  ({n_tc_valid/n_lots_total*100:.1f}%)')
print(f'Lots with null tenders_count:              {n_tc_null:>7,}  ({n_tc_null/n_lots_total*100:.1f}%)')
print(f'Lots with tenders_count == 0:              {n_tc_zero:>7,}  ({n_tc_zero/n_lots_total*100:.1f}%)')
print()
print('Distribution of tenders_count (where not null and > 0):')
tc_valid = gold_awards[gold_awards['tenders_count'] > 0]['tenders_count']
print(f'  Min:    {tc_valid.min():.0f}')
print(f'  Median: {tc_valid.median():.1f}')
print(f'  Mean:   {tc_valid.mean():.2f}')
print(f'  Max:    {tc_valid.max():.0f}')
print(f'  P90:    {tc_valid.quantile(0.90):.1f}')
print(f'  P95:    {tc_valid.quantile(0.95):.1f}')

tc_country = (
    gold_awards
    .groupby('buyer_country')
    .agg(
        total_lots = ('notice_id',     'count'),
        valid_tc   = ('tenders_count', lambda x: x.notna().sum()),
        mean_tc    = ('tenders_count', 'mean'),
        median_tc  = ('tenders_count', 'median')
    )
    .reset_index()
)
tc_country['pct_coverage'] = tc_country['valid_tc'] / tc_country['total_lots'] * 100
tc_country = tc_country[tc_country['total_lots'] >= 50].sort_values('pct_coverage', ascending=False)

print('tenders_count coverage by country (countries with >= 50 lots):')
print(f'{"Country":<10} {"Total Lots":>12} {"Valid TC":>10} {"Coverage":>10} {"Median TC":>10} {"Mean TC":>10}')
print('-' * 65)
for _, row in tc_country.iterrows():
    med = f"{row['median_tc']:.1f}" if pd.notna(row['median_tc']) else 'N/A'
    mn  = f"{row['mean_tc']:.2f}"   if pd.notna(row['mean_tc'])   else 'N/A'
    print(f'{row["buyer_country"]:<10} {int(row["total_lots"]):>12,} {int(row["valid_tc"]):>10,}'
          f' {row["pct_coverage"]:>9.1f}% {med:>10} {mn:>10}')

tc_type = (
    gold_awards
    .groupby('proc_type')
    .agg(
        total_lots = ('notice_id',     'count'),
        valid_tc   = ('tenders_count', lambda x: x.notna().sum()),
        median_tc  = ('tenders_count', 'median'),
        mean_tc    = ('tenders_count', 'mean')
    )
    .reset_index()
)
tc_type['pct_coverage'] = tc_type['valid_tc'] / tc_type['total_lots'] * 100

print('tenders_count coverage by procurement type:')
print(tc_type.round(2).to_string(index=False))

tc_cpv = (
    gold_awards
    .groupby(['cpv_division', 'cpv_division_name'])
    .agg(
        total_lots = ('notice_id',     'count'),
        valid_tc   = ('tenders_count', lambda x: x.notna().sum()),
        median_tc  = ('tenders_count', 'median')
    )
    .reset_index()
)
tc_cpv['pct_coverage'] = tc_cpv['valid_tc'] / tc_cpv['total_lots'] * 100
tc_cpv = tc_cpv[tc_cpv['total_lots'] >= 100].sort_values('total_lots', ascending=False)

print()
print('tenders_count coverage by CPV division (>= 100 lots):')
print(f'{"CPV":>5} {"Division Name":35} {"Lots":>8} {"Coverage":>10} {"Median TC":>10}')
print('-' * 75)
for _, row in tc_cpv.head(20).iterrows():
    name = str(row['cpv_division_name'])[:33] if pd.notna(row['cpv_division_name']) else 'Unknown'
    med  = f"{row['median_tc']:.1f}" if pd.notna(row['median_tc']) else 'N/A'
    print(f'{str(row["cpv_division"]):>5} {name:<35} {int(row["total_lots"]):>8,}'
          f' {row["pct_coverage"]:>9.1f}% {med:>10}')


# ----------------------------------------------------------------------
# ## 2. Competition Intensity by Country
# > **KPI Definition — Competition Intensity Score (CIS):**
# > The median number of bids received per lot for a given country, computed at notice-level
# > (average tenders across lots per notice, then median across notices for that country).
# > A higher score means more suppliers are competing for each contract.
# > Median is used rather than mean to avoid distortion from a small number of
# > highly contested lots pulling the average up.
# >
# > **Why this matters:** A country with high notice volume (NB1) but low competition intensity
# > is a market dominated by incumbent suppliers — harder to break into.
# > A country with high competition intensity is an open, contested market.
# >
# > **Dataset context:** 21,795 notices have a valid tenders_count. The EU-wide median is
# > 2.0 bidders per lot and the mean is 3.36 — a right-skewed distribution where most
# > contracts attract few bidders but a long tail of highly contested tenders pulls the
# > mean up. 32.3% of notices have only one bidder; 19.4% attract five or more.

comp_country = (
    notice_awards[notice_awards['tenders_count_avg'].notna()]
    .groupby('buyer_country')
    .agg(
        median_bidders = ('tenders_count_avg', 'median'),
        mean_bidders   = ('tenders_count_avg', 'mean'),
        n_notices      = ('notice_id',         'count'),
        pct_contested  = ('tenders_count_avg', lambda x: (x > 1).mean() * 100)
    )
    .reset_index()
)
comp_country = comp_country[comp_country['n_notices'] >= 20].sort_values('median_bidders', ascending=False)

print('Competition Intensity by Country (>= 20 notices with tenders_count):')
print(f'{"Country":<10} {"Median Bidders":>15} {"Mean Bidders":>13} {"N Notices":>11} {"% > 1 Bidder":>14}')
print('-' * 68)
for _, row in comp_country.iterrows():
    flag = ' <- most competitive' if row['median_bidders'] >= 4 else (
           ' <- least competitive' if row['median_bidders'] <= 1 else '')
    print(f'{row["buyer_country"]:<10} {row["median_bidders"]:>15.1f} {row["mean_bidders"]:>13.2f}'
          f' {int(row["n_notices"]):>11,} {row["pct_contested"]:>13.1f}%{flag}')

top20_comp = comp_country.head(20)

fig, ax = plt.subplots(figsize=(12, 8))
colors = [C_GREEN if v >= 4 else (C_ORANGE if v >= 3 else C_RED)
          for v in top20_comp['median_bidders']]
bars = ax.barh(
    top20_comp['buyer_country'][::-1],
    top20_comp['median_bidders'][::-1],
    color=colors[::-1], alpha=0.85, edgecolor='white'
)
ax.axvline(3, color=C_ORANGE, linestyle='--', linewidth=1.2)
ax.axvline(4, color=C_GREEN,  linestyle='--', linewidth=1.2)
ax.set_xlabel('Median Bidders per Lot')
ax.set_title('Competition Intensity by Country — Median Bidders per Lot | January 2026', fontweight='bold')

green_patch  = mpatches.Patch(color=C_GREEN,  label='High (>=4 bidders)')
orange_patch = mpatches.Patch(color=C_ORANGE, label='Medium (3–4 bidders)')
red_patch    = mpatches.Patch(color=C_RED,    label='Low (<3 bidders)')
ax.legend(handles=[green_patch, orange_patch, red_patch], fontsize=9)

for bar, val in zip(bars, top20_comp['median_bidders'][::-1]):
    ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}', va='center', fontsize=8)

plt.tight_layout()
plt.show()

tc_notices = notice_awards[notice_awards['tenders_count_avg'].notna()]['tenders_count_avg']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Distribution of Competition Intensity — All Notices | January 2026',
             fontsize=13, fontweight='bold')

axes[0].hist(tc_notices.clip(upper=20), bins=40, color=C_BLUE, alpha=0.8, edgecolor='white')
axes[0].axvline(tc_notices.median(), color=C_ORANGE, linestyle='--', linewidth=1.5,
                label=f'Median ({tc_notices.median():.1f})')
axes[0].axvline(tc_notices.mean(), color=C_RED, linestyle='--', linewidth=1.5,
                label=f'Mean ({tc_notices.mean():.2f})')
axes[0].set_xlabel('Avg Bidders per Lot (capped at 20)')
axes[0].set_ylabel('Number of Notices')
axes[0].set_title('Full Distribution (capped at 20)', fontweight='bold')
axes[0].legend()

brackets = [
    ('1 bidder\n(no competition)', 1, 1.5),
    ('2-3 bidders\n(low)',         1.5, 3.5),
    ('4-5 bidders\n(medium)',      3.5, 5.5),
    ('6-10 bidders\n(high)',       5.5, 10.5),
    ('>10 bidders\n(very high)',   10.5, float('inf')),
]
labels  = [b[0] for b in brackets]
counts  = [((tc_notices >= lo) & (tc_notices < hi)).sum() for _, lo, hi in brackets]
bkt_clr = [C_RED, C_ORANGE, C_GOLD, C_TEAL, C_GREEN]

bars2 = axes[1].bar(range(len(labels)), counts, color=bkt_clr, alpha=0.85, edgecolor='white')
axes[1].set_xticks(range(len(labels)))
axes[1].set_xticklabels(labels, fontsize=9)
axes[1].set_ylabel('Number of Notices')
axes[1].set_title('Competition Bracket Distribution', fontweight='bold')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

for bar, cnt in zip(bars2, counts):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                 f'{cnt:,}\n({cnt/len(tc_notices)*100:.1f}%)',
                 ha='center', fontsize=8)

plt.tight_layout()
plt.show()

print(f'  Total notices with tenders_count:  {len(tc_notices):,}')
print(f'  Median bidders per lot:            {tc_notices.median():.2f}')
print(f'  Mean bidders per lot:              {tc_notices.mean():.2f}')
print(f'  % notices with only 1 bidder:      {(tc_notices == 1).mean()*100:.1f}%')
print(f'  % notices with 5+ bidders:         {(tc_notices >= 5).mean()*100:.1f}%')


# ----------------------------------------------------------------------
# ## 3. Competition Intensity by CPV Category
# > **Why this is the most important section for Microsoft:**
# > Different procurement categories attract very different numbers of bidders.
# > Construction works requires local presence and specialist equipment — naturally
# > limiting the supplier pool. IT services can be tendered remotely by any qualified
# > supplier across the EU — which might suggest higher competition. The data
# > tells a different story: CPV 72 has the lowest median of any major category.
# > Understanding where CPV 72 and CPV 48 sit in this landscape tells Microsoft
# > precisely how contested its core market is.

comp_cpv = (
    notice_awards[notice_awards['tenders_count_avg'].notna()]
    .groupby(['cpv_division', 'cpv_division_name'])
    .agg(
        median_bidders = ('tenders_count_avg', 'median'),
        mean_bidders   = ('tenders_count_avg', 'mean'),
        n_notices      = ('notice_id',         'count'),
        pct_solo       = ('tenders_count_avg', lambda x: (x <= 1).mean() * 100)
    )
    .reset_index()
)
comp_cpv = comp_cpv[comp_cpv['n_notices'] >= 20].sort_values('median_bidders', ascending=False)

print('Competition Intensity by CPV Division (>= 20 notices):')
print(f'{"CPV":>5} {"Division Name":35} {"Median Bidders":>15} {"Mean":>8} {"Notices":>9} {"% Solo":>8}')
print('-' * 85)
for _, row in comp_cpv.iterrows():
    name = str(row['cpv_division_name'])[:33] if pd.notna(row['cpv_division_name']) else 'Unknown'
    msft = ' * MSFT' if str(row['cpv_division']) in ['72', '48', '73', '79'] else ''
    print(f'{str(row["cpv_division"]):>5} {name:<35} {row["median_bidders"]:>15.1f}'
          f' {row["mean_bidders"]:>8.2f} {int(row["n_notices"]):>9,} {row["pct_solo"]:>7.1f}%{msft}')

top15_cpv_comp    = comp_cpv.head(15)
bottom15_cpv_comp = comp_cpv.tail(15)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Competition Intensity by CPV Division | January 2026', fontsize=13, fontweight='bold')

for ax, data, title, color in [
    (axes[0], top15_cpv_comp,    'Most Competitive Categories (Top 15)',     C_GREEN),
    (axes[1], bottom15_cpv_comp, 'Least Competitive Categories (Bottom 15)', C_RED),
]:
    labels_cpv = [
        f"CPV {row['cpv_division']} - {str(row['cpv_division_name'])[:28] if pd.notna(row['cpv_division_name']) else 'Unknown'}"
        for _, row in data.iterrows()
    ]
    ax.barh(labels_cpv[::-1], data['median_bidders'].tolist()[::-1],
            color=color, alpha=0.85, edgecolor='white')
    ax.set_xlabel('Median Bidders per Lot')
    ax.set_title(title, fontweight='bold')
    for i, val in enumerate(data['median_bidders'].tolist()[::-1]):
        ax.text(val + 0.05, i, f'{val:.1f}', va='center', fontsize=8)

plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------
# ## 4. Competition Intensity by Procurement Type
# > **The Works hypothesis from NB2:**
# > Works contracts had the lowest savings rate at 28.3% — buyers got the least discount
# > on construction contracts. A natural hypothesis is that Works also attracts fewer
# > bidders, explaining the poor savings outcome. The data refutes this directly:
# > Works has a median of **4.0 bidders** — the highest of any procurement type —
# > and a mean of 5.39, driven by multi-lot construction tenders with many sub-contractors.
# >
# > This means Works' low savings rate is not explained by lack of competition.
# > Cost complexity, scope uncertainty, and specialist sub-contracting structures
# > prevent buyers from achieving discounts even when multiple suppliers compete.
# >
# > The breakdown by competition bracket and type (Section 5) reinforces this further:
# > for Works, adding more bidders does not improve savings at all. For Supplies,
# > it does — dramatically. These are structurally different markets.
# >
# > Services and Supplies both have a median of 2.0 bidders, but Supplies' mean (2.61)
# > is lower than Services' mean (3.57), reflecting the broader tail of highly contested
# > service tenders.

comp_type = (
    notice_awards[notice_awards['tenders_count_avg'].notna()]
    .groupby('proc_type')
    .agg(
        median_bidders = ('tenders_count_avg', 'median'),
        mean_bidders   = ('tenders_count_avg', 'mean'),
        n_notices      = ('notice_id',         'count'),
        pct_solo       = ('tenders_count_avg', lambda x: (x <= 1).mean() * 100),
        pct_high       = ('tenders_count_avg', lambda x: (x >= 5).mean() * 100)
    )
    .reset_index()
    .sort_values('median_bidders', ascending=False)
)

print('Competition Intensity by Procurement Type:')
print(comp_type.round(2).to_string(index=False))
print()

savings_ref = {'Services': '~mid', 'Supplies': '56.9%', 'Works': '28.3%'}
print('Cross-reference with NB2 savings rates:')
print(f'{"Type":<12} {"Median Bidders":>15} {"NB2 Savings %":>15} {"Interpretation"}')
print('-' * 80)
for _, row in comp_type.iterrows():
    sav = savings_ref.get(row['proc_type'], 'N/A')
    if row['proc_type'] == 'Works':
        interp = 'Highest competition yet lowest savings -> cost complexity, not low competition'
    elif row['proc_type'] == 'Supplies':
        interp = 'Lowest mean bidders, highest savings -> efficient commodity markets'
    else:
        interp = 'Median same as Supplies; broader high-bidder tail'
    print(f'{row["proc_type"]:<12} {row["median_bidders"]:>15.1f} {sav:>15}  {interp}')

types       = comp_type['proc_type'].tolist()
clr_map     = {'Services': C_BLUE, 'Supplies': C_ORANGE, 'Works': C_GREEN}
type_colors = [clr_map.get(t, C_GREY) for t in types]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Competition Intensity by Procurement Type | January 2026', fontsize=13, fontweight='bold')

bars1 = axes[0].bar(types, comp_type['median_bidders'], color=type_colors, alpha=0.85, edgecolor='white')
axes[0].set_ylabel('Median Bidders per Lot')
axes[0].set_title('Median Bidders per Lot by Type', fontweight='bold')
for bar in bars1:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{bar.get_height():.1f}', ha='center', fontsize=11, fontweight='bold')

bars2 = axes[1].bar(types, comp_type['pct_solo'], color=type_colors, alpha=0.85, edgecolor='white')
axes[1].set_ylabel('% of Notices with Only 1 Bidder')
axes[1].set_title('% of Contracts with Zero Competition', fontweight='bold')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
for bar in bars2:
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{bar.get_height():.1f}%', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------
# ## 5. Competition vs Savings — Do More Bidders Produce Lower Prices?
# > **KPI Definition — Competition-Savings Correlation:**
# > The statistical relationship between the number of bidders per lot and the savings
# > percentage (how much below estimate the final contract came in). A positive correlation
# > means more competition drives more savings. A near-zero correlation means bidder count
# > alone is not a useful predictor of procurement value.
# >
# > **Key finding:** With 9,514 matched notices, the Pearson r is 0.027 and the Spearman r
# > is 0.045 — both negligible. Competition level alone does not explain savings rates
# > in EU public procurement. The bracket analysis confirms this: median savings are
# > stable across all five competition brackets (48–58%), with the 2–3 bidder bracket
# > producing the highest median savings (57.9%).
# >
# > **However, the procurement-type breakdown reveals a more nuanced picture:**
# > For Supplies, competition and savings move together strongly — contracts with
# > more than 10 bidders achieve a median savings of 94.4%, versus 50.2% for solo
# > Supplies contracts. For Works, the relationship is flat or negative — more bidders
# > do not produce better outcomes. The EU-wide correlation is near zero because these
# > two opposing dynamics cancel each other out in aggregate.
# >
# > **Subset used:** Only notices where tenders_count AND estimated AND contract_value
# > are all present. Switzerland excluded (no estimated values). Hungary excluded
# > (framework ceiling values inflate estimates).

CAP = 5_000_000_000

comp_sav_lots = gold_awards[
    gold_awards['tenders_count'].notna() &
    gold_awards['estimated'].notna() &
    gold_awards['cv_clean'].notna() &
    (gold_awards['cv_clean'] > 0) &
    (gold_awards['estimated'] > 0)
].copy()

comp_sav = (
    comp_sav_lots
    .groupby(['notice_id', 'buyer_country', 'proc_type', 'cpv_division', 'cpv_division_name'])
    .agg(
        estimated_total   = ('estimated',     'sum'),
        awarded_total     = ('cv_clean',      'sum'),
        tenders_count_avg = ('tenders_count', 'mean')
    )
    .reset_index()
)

comp_sav = comp_sav[
    (comp_sav['estimated_total'] > 0) &
    (comp_sav['awarded_total']   > 0) &
    (comp_sav['buyer_country']   != 'HUN') &
    (comp_sav['buyer_country']   != 'CHE')
].copy()

comp_sav['savings_pct'] = (
    (comp_sav['estimated_total'] - comp_sav['awarded_total'])
    / comp_sav['estimated_total'] * 100
)

comp_sav = comp_sav[
    (comp_sav['savings_pct'] >= -200) &
    (comp_sav['savings_pct'] <=  200)
].copy()

print('=== COMPETITION-SAVINGS DATASET ===')
print(f'Lot-level records (3 fields present):        {len(comp_sav_lots):>7,}')
print(f'Collapsed to notice-level (ex-HUN, ex-CHE): {len(comp_sav):>7,}')
print()
print(f'Median savings %:      {comp_sav["savings_pct"].median():.1f}%')
print(f'Median bidders/lot:    {comp_sav["tenders_count_avg"].median():.1f}')
print()

corr_pearson  = comp_sav[['tenders_count_avg', 'savings_pct']].corr().iloc[0, 1]
corr_spearman = comp_sav[['tenders_count_avg', 'savings_pct']].corr(method='spearman').iloc[0, 1]
print(f'Pearson r  (tenders vs savings):   {corr_pearson:.3f}')
print(f'Spearman r (tenders vs savings):   {corr_spearman:.3f}')
print()
print('Interpretation: r < 0.1 = negligible relationship.')
print('Competition intensity alone does not predict procurement savings in this dataset.')

comp_sav['comp_bracket'] = pd.cut(
    comp_sav['tenders_count_avg'],
    bins=[0, 1.5, 3.5, 5.5, 10.5, float('inf')],
    labels=['1 bidder', '2-3', '4-5', '6-10', '>10']
)

bracket_savings = (
    comp_sav.groupby('comp_bracket', observed=True)
    .agg(
        median_savings = ('savings_pct', 'median'),
        mean_savings   = ('savings_pct', 'mean'),
        n_contracts    = ('notice_id',   'count')
    )
    .reset_index()
)

print('Median savings % by competition bracket:')
print(bracket_savings.round(2).to_string(index=False))
print()

bracket_type = (
    comp_sav.groupby(['comp_bracket', 'proc_type'], observed=True)
    ['savings_pct'].median().unstack(fill_value=np.nan).round(1)
)
print('Median savings % by competition bracket AND procurement type:')
print(bracket_type.to_string())
print()
print('Key insight: for Supplies, savings scale strongly with competition (50% -> 94%).')
print('For Works, more bidders produce no improvement. These opposing effects cancel')
print('each other in aggregate, explaining the near-zero EU-wide correlation.')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Competition Intensity vs Savings % | January 2026 (Ex-Hungary, Ex-Switzerland)',
             fontsize=13, fontweight='bold')

bkt_labels = bracket_savings['comp_bracket'].astype(str).tolist()
bkt_colors = [C_RED, C_ORANGE, C_GOLD, C_TEAL, C_GREEN]

bars1 = axes[0].bar(bkt_labels, bracket_savings['median_savings'],
                    color=bkt_colors[:len(bkt_labels)], alpha=0.85, edgecolor='white')
axes[0].set_xlabel('Bidders per Lot')
axes[0].set_ylabel('Median Savings %')
axes[0].set_title('Median Savings by Competition Bracket', fontweight='bold')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
for bar, val in zip(bars1, bracket_savings['median_savings']):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')

axes[1].bar(bkt_labels, bracket_savings['n_contracts'],
            color=bkt_colors[:len(bkt_labels)], alpha=0.85, edgecolor='white')
axes[1].set_xlabel('Bidders per Lot')
axes[1].set_ylabel('Number of Contracts')
axes[1].set_title('Contract Count per Bracket', fontweight='bold')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

pt_colors = {'Services': C_BLUE, 'Supplies': C_ORANGE, 'Works': C_GREEN}
x_pos = range(len(bkt_labels))
for pt in ['Supplies', 'Works', 'Services']:
    if pt in bracket_type.columns:
        vals = bracket_type[pt].tolist()
        axes[2].plot(x_pos, vals, marker='o', linewidth=2, markersize=7,
                     color=pt_colors.get(pt, C_GREY), label=pt)
axes[2].set_xticks(list(x_pos))
axes[2].set_xticklabels(bkt_labels, fontsize=9)
axes[2].set_xlabel('Bidders per Lot')
axes[2].set_ylabel('Median Savings %')
axes[2].set_title('Savings by Competition Bracket & Type', fontweight='bold')
axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
axes[2].legend(fontsize=9)

plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------
# ## 6. Competition by Contract Size Segment
# > **Why this analysis matters:** NB2 Section 11 established four contract size tiers
# > based on awarded value: Micro (<€50K), Small (€50K–€500K), Mid (€500K–€5M),
# > and Large (>€5M). The assumption is that larger contracts attract more bidders —
# > higher value means more suppliers have the scale to compete.
# >
# > **Key finding (ex-Hungary dataset):** All four segments show a median of exactly
# > 2.0 bidders. The solo contract rate falls modestly from 38.3% (Micro) to
# > 31.5% (Large) — a 7-point gradient across the full value spectrum.
# > Competition intensity is structurally uniform across all contract sizes; value tier
# > alone does not predict how many suppliers will bid.
# >
# > This finding reinforces the conclusion from Section 5: contract value and competition
# > level are both weak predictors of savings rates. The structural floor of 2.0 median
# > bidders across all tiers reflects EU procurement rules requiring minimum advertising
# > thresholds, not competitive market forces.

seg_data = notice_awards_exhun[notice_awards_exhun['tenders_count_avg'].notna()].copy()

seg_data['size_segment'] = pd.cut(
    seg_data['awarded_eur_total'],
    bins=[0, 50_000, 500_000, 5_000_000, float('inf')],
    labels=['Micro (<50K)', 'Small (50K-500K)', 'Mid (500K-5M)', 'Large (>5M)']
)

comp_by_size = (
    seg_data.dropna(subset=['size_segment'])
    .groupby('size_segment', observed=True)
    .agg(
        n_notices      = ('notice_id',         'count'),
        median_bidders = ('tenders_count_avg', 'median'),
        mean_bidders   = ('tenders_count_avg', 'mean'),
        pct_solo       = ('tenders_count_avg', lambda x: (x <= 1).mean() * 100)
    )
    .reset_index()
)

print('Competition intensity by contract size segment (ex-Hungary):')
print(f'{"size_segment":<22} {"n_notices":>10} {"median_bidders":>15} {"mean_bidders":>13} {"pct_solo":>10}')
print('-' * 75)
for _, row in comp_by_size.iterrows():
    print(f'{str(row["size_segment"]):<22} {int(row["n_notices"]):>10,} {row["median_bidders"]:>15.1f}'
          f' {row["mean_bidders"]:>13.2f} {row["pct_solo"]:>9.2f}%')
print()
print('Median bidders is 2.0 across all four segments.')
print('Solo contract rate falls from 38.3% (Micro) to 31.5% (Large) — a modest gradient.')
print('Competition intensity does not scale with contract value.')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Competition Intensity by Contract Size Segment | January 2026 (Ex-Hungary)',
             fontsize=13, fontweight='bold')

seg_labels = comp_by_size['size_segment'].astype(str).tolist()
seg_colors = [C_TEAL, C_BLUE, C_PURPLE, C_ORANGE]

bars1 = axes[0].bar(range(len(seg_labels)), comp_by_size['median_bidders'],
                    color=seg_colors, alpha=0.85, edgecolor='white')
axes[0].set_xticks(range(len(seg_labels)))
axes[0].set_xticklabels(seg_labels, rotation=15, ha='right', fontsize=9)
axes[0].set_ylabel('Median Bidders per Lot')
axes[0].set_title('Median Bidders by Segment', fontweight='bold')
axes[0].set_ylim(0, comp_by_size['median_bidders'].max() * 1.3)
for bar, val in zip(bars1, comp_by_size['median_bidders']):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                 f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')

bars2 = axes[1].bar(range(len(seg_labels)), comp_by_size['pct_solo'],
                    color=seg_colors, alpha=0.85, edgecolor='white')
axes[1].set_xticks(range(len(seg_labels)))
axes[1].set_xticklabels(seg_labels, rotation=15, ha='right', fontsize=9)
axes[1].set_ylabel('% Notices with Only 1 Bidder')
axes[1].set_title('Solo Contract Rate by Segment', fontweight='bold')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
for bar, val in zip(bars2, comp_by_size['pct_solo']):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

bars3 = axes[2].bar(range(len(seg_labels)), comp_by_size['n_notices'],
                    color=seg_colors, alpha=0.85, edgecolor='white')
axes[2].set_xticks(range(len(seg_labels)))
axes[2].set_xticklabels(seg_labels, rotation=15, ha='right', fontsize=9)
axes[2].set_ylabel('Number of Notices')
axes[2].set_title('Notice Count by Segment', fontweight='bold')
axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
for bar, val in zip(bars3, comp_by_size['n_notices']):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(comp_by_size['n_notices'])*0.01,
                 f'{int(val):,}', ha='center', fontsize=9)

plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------
# ## 7. Market Competition Ranking by Country
# > **KPI Definition — Market Competition Ranking:**
# > A country-level classification based on competition intensity (median bidders per lot),
# > using the thresholds applied in the pd.cut bins: Low (≤2 bidders), Medium (2–4 bidders),
# > High (>4 bidders). This ranking feeds into the dashboard's country comparison view
# > and the semantic layer definition of market openness.
# >
# > **Key finding:** No EU country reaches the High competition tier in the January 2026
# > dataset. The most competitive national markets — Switzerland (4.0) and Malta (3.5)
# > — fall in the Medium bracket. The majority of countries, including major markets
# > such as Poland, Greece, and Bulgaria, sit at a median of 1.0 bidder, placing them
# > firmly in the Low tier. EU procurement competition is universally low; the variation
# > between countries is a question of degree, not of kind.

mai = comp_country[['buyer_country', 'median_bidders', 'n_notices']].copy()
mai = mai[mai['n_notices'] >= 20].copy()

mai['market_type'] = pd.cut(
    mai['median_bidders'],
    bins=[0, 2, 4, float('inf')],
    labels=['Low competition', 'Medium competition', 'High competition']
)

print('Market Competition Ranking by Country:')
print(f'{"Country":<10} {"Median Bidders":>15} {"N Notices":>11} {"Market Type"}')
print('-' * 55)
for _, row in mai.sort_values('median_bidders', ascending=False).iterrows():
    print(f'{row["buyer_country"]:<10} {row["median_bidders"]:>15.1f} {int(row["n_notices"]):>11,}  {row["market_type"]}')
print()
print('Distribution by tier:')
print(mai['market_type'].value_counts().to_string())
print()
print('No country reaches the High competition tier (>4 median bidders).')
print('The most competitive market in the dataset (CHE) reaches exactly 4.0.')

fig, ax = plt.subplots(figsize=(10, 8))
mai_sorted = mai.sort_values('median_bidders', ascending=True)
type_color_map = {'Low competition': C_RED, 'Medium competition': C_ORANGE, 'High competition': C_GREEN}
colors_bar = [type_color_map.get(str(t), C_GREY) for t in mai_sorted['market_type']]

ax.barh(mai_sorted['buyer_country'], mai_sorted['median_bidders'],
        color=colors_bar, alpha=0.85, edgecolor='white')
ax.axvline(2, color=C_RED,    linestyle='--', linewidth=1, alpha=0.7)
ax.axvline(4, color=C_ORANGE, linestyle='--', linewidth=1, alpha=0.7)
ax.set_xlabel('Median Bidders per Lot')
ax.set_title('Market Competition Ranking by Country | January 2026', fontweight='bold')

low_p  = mpatches.Patch(color=C_RED,    label='Low (<=2 bidders)')
mid_p  = mpatches.Patch(color=C_ORANGE, label='Medium (2-4 bidders)')
high_p = mpatches.Patch(color=C_GREEN,  label='High (>4 bidders)')
ax.legend(handles=[low_p, mid_p, high_p], fontsize=9)

for bar, val in zip(ax.patches, mai_sorted['median_bidders']):
    ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}', va='center', fontsize=8)

plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------
# ## 8. Microsoft Spotlight — CPV 72 and CPV 48 Competition Profile
# > **Why this section exists:** Everything above is EU-wide analysis.
# > This section focuses on the two CPV divisions most directly relevant to Microsoft's
# > public sector business:
# > - **CPV 72 — IT services** (cloud, infrastructure, consulting, data services)
# > - **CPV 48 — Software packages and information systems**
# >
# > NB2 established that CPV 72 + 48 represent the single largest procurement sector
# > by awarded value. This section adds the competition dimension.
# >
# > **Key findings:**
# > - CPV 72 has a median of **1.0 bidder** per lot and **50.5% solo contracts** — the
# >   most closed of any major IT procurement category. Half of all EU IT services
# >   contracts are awarded with no competitive alternative.
# > - CPV 48 (Software) has a median of 2.0 and 46.8% solo — marginally more contested
# >   but structurally concentrated compared to the EU average of 32.3%.
# > - Both categories sit well below the EU-wide mean of 3.36 bidders.
# >
# > **Country dimension:** CPV 72 competition varies substantially by market. Switzerland,
# > Italy, and Ireland have the most competitive IT procurement (median 3.0–4.0). Eastern
# > European markets — Croatia (89.3% solo), Bulgaria (81.6%), Poland (74.2%), and
# > Romania (64.2%) — are the most concentrated. For Microsoft, these are the markets
# > where incumbent lock-in is strongest.
# >
# > **Size dimension within CPV 72:** Micro, Small, and Mid contracts all have a median of
# > 1.0 bidder. Only Large contracts (>€5M) show a median of 2.0, suggesting that
# > competition in IT procurement exists almost exclusively at the highest value tier.
# > The concentration is structural — not an artefact of contract size.

msft_divs = ['72', '48']
msft_data = notice_awards[notice_awards['cpv_division'].astype(str).isin(msft_divs)].copy()

tc_notices = notice_awards[notice_awards['tenders_count_avg'].notna()]['tenders_count_avg']

print('=== MICROSOFT SPOTLIGHT: CPV 72 + CPV 48 ===')
print()

for div in msft_divs:
    sub  = msft_data[msft_data['cpv_division'].astype(str) == div]
    name = sub['cpv_division_name'].dropna().iloc[0] if len(sub) > 0 else 'Unknown'
    tc   = sub['tenders_count_avg'].dropna()

    print(f'CPV {div} - {name}')
    print(f'  Total notices (all):           {len(sub):,}')
    print(f'  Notices with tenders_count:    {len(tc):,}')
    print(f'  Median bidders per lot:        {tc.median():.1f}')
    print(f'  Mean bidders per lot:          {tc.mean():.2f}')
    print(f'  % notices with only 1 bidder:  {(tc <= 1).mean()*100:.1f}%')
    print(f'  % notices with 5+ bidders:     {(tc >= 5).mean()*100:.1f}%')
    print()

print('EU overall baseline:')
print(f'  Median bidders per lot:        {tc_notices.median():.1f}')
print(f'  Mean bidders per lot:          {tc_notices.mean():.2f}')
print(f'  % with only 1 bidder:          {(tc_notices == 1).mean()*100:.1f}%')

cpv72 = notice_awards[notice_awards['cpv_division'].astype(str) == '72'].copy()

cpv72_comp_country = (
    cpv72[cpv72['tenders_count_avg'].notna()]
    .groupby('buyer_country')
    .agg(
        n_notices      = ('notice_id',         'count'),
        median_bidders = ('tenders_count_avg', 'median'),
        mean_bidders   = ('tenders_count_avg', 'mean'),
        pct_solo       = ('tenders_count_avg', lambda x: (x <= 1).mean() * 100)
    )
    .reset_index()
)
cpv72_comp_country = cpv72_comp_country[cpv72_comp_country['n_notices'] >= 10].sort_values('median_bidders', ascending=False)

print('CPV 72 (IT Services) — Competition by Country (>= 10 notices):')
print(f'{"Country":<10} {"N Notices":>11} {"Median Bidders":>15} {"Mean Bidders":>13} {"% Solo":>8}')
print('-' * 62)
for _, row in cpv72_comp_country.iterrows():
    print(f'{row["buyer_country"]:<10} {int(row["n_notices"]):>11,} {row["median_bidders"]:>15.1f}'
          f' {row["mean_bidders"]:>13.2f} {row["pct_solo"]:>7.1f}%')

cpv72_seg = notice_awards_exhun[
    notice_awards_exhun['cpv_division'].astype(str) == '72'
].copy()
cpv72_seg = cpv72_seg[cpv72_seg['tenders_count_avg'].notna()].copy()

cpv72_seg['size_segment'] = pd.cut(
    cpv72_seg['awarded_eur_total'],
    bins=[0, 50_000, 500_000, 5_000_000, float('inf')],
    labels=['Micro (<50K)', 'Small (50K-500K)', 'Mid (500K-5M)', 'Large (>5M)']
)

cpv72_by_size = (
    cpv72_seg.dropna(subset=['size_segment'])
    .groupby('size_segment', observed=True)
    .agg(
        n_notices      = ('notice_id',         'count'),
        median_bidders = ('tenders_count_avg', 'median'),
        pct_solo       = ('tenders_count_avg', lambda x: (x <= 1).mean() * 100)
    )
    .reset_index()
)

print('CPV 72 — Competition by Contract Size Segment (ex-Hungary):')
print(cpv72_by_size.round(2).to_string(index=False))
print()
print('Micro, Small, and Mid segments all have median 1.0 bidder.')
print('Only Large contracts (>5M) reach a median of 2.0 — competition in IT')
print('procurement exists almost exclusively at the highest value tier.')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Microsoft Spotlight — CPV 72 & CPV 48 | Competition Profile | January 2026',
             fontsize=13, fontweight='bold')

tc_notices_local = notice_awards[notice_awards['tenders_count_avg'].notna()]['tenders_count_avg']

comp_comp = []
for div in ['72', '48']:
    tc = notice_awards[notice_awards['cpv_division'].astype(str) == div]['tenders_count_avg'].dropna()
    comp_comp.append({'CPV': f'CPV {div}', 'median': tc.median(), 'mean': tc.mean()})
comp_comp.append({'CPV': 'EU Average', 'median': tc_notices_local.median(), 'mean': tc_notices_local.mean()})
cc_df = pd.DataFrame(comp_comp)

x_cc = range(len(cc_df))
w = 0.38
axes[0, 0].bar([i - w/2 for i in x_cc], cc_df['median'], w, color=C_BLUE,  alpha=0.85, label='Median')
axes[0, 0].bar([i + w/2 for i in x_cc], cc_df['mean'],   w, color=C_TEAL,  alpha=0.85, label='Mean')
axes[0, 0].set_xticks(list(x_cc))
axes[0, 0].set_xticklabels(cc_df['CPV'])
axes[0, 0].set_ylabel('Bidders per Lot')
axes[0, 0].set_title('Competition Intensity — IT vs EU Baseline', fontweight='bold')
axes[0, 0].legend()

solo_data = []
for div in ['72', '48']:
    tc = notice_awards[notice_awards['cpv_division'].astype(str) == div]['tenders_count_avg'].dropna()
    solo_data.append({'CPV': f'CPV {div}', 'pct_solo': (tc <= 1).mean() * 100})
solo_data.append({'CPV': 'EU Average', 'pct_solo': (tc_notices_local == 1).mean() * 100})
sd_df = pd.DataFrame(solo_data)

bars_solo = axes[0, 1].bar(sd_df['CPV'], sd_df['pct_solo'],
                            color=[C_BLUE, C_PURPLE, C_GREY], alpha=0.85, edgecolor='white')
axes[0, 1].set_ylabel('% Contracts with Only 1 Bidder')
axes[0, 1].set_title('Solo Contract Rate — IT vs EU Baseline', fontweight='bold')
axes[0, 1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
for bar in bars_solo:
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{bar.get_height():.1f}%', ha='center', fontsize=10, fontweight='bold')

if len(cpv72_comp_country) > 0:
    top12 = cpv72_comp_country.head(12)
    axes[1, 0].barh(top12['buyer_country'][::-1], top12['median_bidders'][::-1],
                    color=C_BLUE, alpha=0.85, edgecolor='white')
    axes[1, 0].set_xlabel('Median Bidders per Lot')
    axes[1, 0].set_title('CPV 72 Competition by Country (Top 12)', fontweight='bold')
    for bar, val in zip(axes[1, 0].patches, top12['median_bidders'][::-1]):
        axes[1, 0].text(val + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}', va='center', fontsize=8)

if len(cpv72_by_size) > 0:
    seg_lbl = cpv72_by_size['size_segment'].astype(str).tolist()
    seg_clr = [C_TEAL, C_BLUE, C_PURPLE, C_ORANGE]
    bars4 = axes[1, 1].bar(range(len(seg_lbl)), cpv72_by_size['median_bidders'],
                            color=seg_clr[:len(seg_lbl)], alpha=0.85, edgecolor='white')
    axes[1, 1].set_xticks(range(len(seg_lbl)))
    axes[1, 1].set_xticklabels(seg_lbl, rotation=15, ha='right', fontsize=9)
    axes[1, 1].set_ylabel('Median Bidders per Lot')
    axes[1, 1].set_title('CPV 72 Competition by Contract Size', fontweight='bold')
    for bar, val in zip(bars4, cpv72_by_size['median_bidders']):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()


# ----------------------------------------------------------------------
# ## Summary of Key Findings — Notebook 3: Competition Analysis
# **Data: January 2026 | TED EU Public Procurement | Notice-Level (Romania lot-normalised, Hungary excluded where financial data used)**
#
# ----------------------------------------------------------------------
#
# ### 1. Overall Competition Intensity
# The January 2026 dataset contains **21,795 notices** with a valid `tenders_count` field.
# The EU-wide median is **2.0 bidders per lot** and the mean is **3.36**, confirming a strongly
# right-skewed distribution: most contracts attract very few bidders, but a long tail of
# highly contested tenders pulls the mean up. **32.3% of contracts have only one bidder** —
# meaning no competitive pressure — while only **19.4% attract five or more bidders**.
# This structural feature of EU public procurement sets the baseline for all subsequent analysis.
#
# ----------------------------------------------------------------------
#
# ### 2. Competition by Procurement Type
# The type breakdown reveals a counterintuitive pattern:
#
# | Type     | Median | Mean | N Notices |
# |----------|--------|------|-----------|
# | Works    | 4.0    | 5.39 | 2,550     |
# | Services | 2.0    | 3.57 | 9,782     |
# | Supplies | 2.0    | 2.61 | 9,463     |
#
# **Works has the highest competition** despite having the lowest savings rate in NB2 (28.3%).
# This disproves the hypothesis that low savings equals low competition. Construction cost
# complexity, scope uncertainty, and specialist sub-contracting structures explain Works'
# poor savings performance independently of how many firms compete. Services and Supplies
# share the same median (2.0), but Services has a higher mean driven by a broader tail of
# multi-bidder tenders.
#
# ----------------------------------------------------------------------
#
# ### 3. Competition vs Savings — Near-Zero Aggregate Correlation, Type-Dependent Dynamics
# Across **9,514 matched notices** (tenders_count + estimated + awarded all present,
# ex-Hungary, ex-Switzerland):
# - **Pearson r = 0.027** — negligible linear relationship
# - **Spearman r = 0.045** — negligible rank-order relationship
#
# Median savings by competition bracket are structurally flat in aggregate:
#
# | Bracket   | Median Savings | N Contracts |
# |-----------|----------------|-------------|
# | 1 bidder  | 48.9%          | 3,761       |
# | 2–3       | 57.9%          | 3,306       |
# | 4–5       | 50.9%          | 1,247       |
# | 6–10      | 48.7%          | 758         |
# | >10       | 49.4%          | 330         |
#
# However, the procurement-type breakdown reveals two opposing dynamics that cancel out
# in the aggregate correlation. For **Supplies**, savings scale strongly with competition:
# solo contracts achieve a median of 50.2%, while contracts with more than 10 bidders
# reach 94.4%. For **Works**, once any competition is present savings collapse immediately — from 49.2% in solo contracts to 31.0% at the 2–3 bidder bracket — and then remain flat between 27% and 36% regardless of how many additional firms compete. This distinction is critical
# for understanding where competitive pressure actually translates into value for money.
#
# ----------------------------------------------------------------------
#
# ### 4. Competition by Contract Size Segment
# Using the same four tiers defined in NB2 Section 11 (ex-Hungary):
#
# | Segment              | N Notices | Median Bidders | Mean Bidders | % Solo |
# |----------------------|-----------|----------------|--------------|--------|
# | Micro (<€50K)        | 2,815     | 2.0            | 3.02         | 38.3%  |
# | Small (€50K–€500K)   | 5,882     | 2.0            | 3.35         | 34.1%  |
# | Mid (€500K–€5M)      | 6,232     | 2.0            | 3.10         | 36.0%  |
# | Large (>€5M)         | 3,808     | 2.0            | 3.13         | 31.5%  |
#
# All four segments share an identical median of 2.0 bidders. The solo contract rate falls
# from 38.3% (Micro) to 31.5% (Large) — a modest 7-point gradient across the full value
# spectrum. Competition intensity is **structurally uniform across contract sizes**: larger
# contracts attract marginally fewer sole-bidder situations, but the effect is far smaller
# than expected.
#
# ----------------------------------------------------------------------
#
# ### 5. Market Competition Ranking
# No EU country reaches the High competition tier (>4 median bidders) in January 2026.
# Switzerland (4.0) and Malta (3.5) are the most competitive markets in the dataset,
# classified as Medium. The majority of countries — including Poland, Greece, Bulgaria,
# Luxembourg, and Croatia — have a median of 1.0 bidder, placing them in the Low tier.
# 13 countries are Medium, 16 are Low, and zero are High. EU procurement competition
# is universally low; the differences between countries reflect degree, not kind.
#
# ----------------------------------------------------------------------
#
# ### 6. Microsoft Spotlight — CPV 72 and CPV 48
# IT procurement is **structurally less competitive than the EU average**:
#
# | Category                   | Notices | Median Bidders | % Solo |
# |----------------------------|---------|----------------|--------|
# | CPV 72 — IT services       | 1,628   | **1.0**        | 50.5%  |
# | CPV 48 — Software packages | 729     | 2.0            | 46.8%  |
# | EU Overall                 | 21,795  | 2.0            | 32.3%  |
#
# **CPV 72 is a concentrated market.** With a median of 1.0 bidder, the typical IT services
# contract receives exactly one tender. 50.5% of all IT services contracts are awarded
# with no competitive alternative — 17 percentage points above the EU-wide rate.
#
# The country breakdown reveals where concentration is most extreme: Croatia (89.3% solo),
# Bulgaria (81.6%), Poland (74.2%), and Romania (64.2%) are the markets most dominated
# by incumbents. Switzerland (20.0% solo) and Norway (6.2%) are the most open.
#
# The contract size breakdown within CPV 72 shows that Micro, Small, and Mid segments
# all have a median of 1.0 bidder. Only Large contracts (>€5M) rise to a median of 2.0,
# indicating that meaningful competition in IT procurement occurs almost exclusively at
# the highest value tier. The concentration is structural, not driven by contract size.
#
# ----------------------------------------------------------------------
#
# ### Connections Between NB3 and NB2
#
# **Works savings vs Works competition:** NB2 found Works had the lowest savings rate
# (28.3%). NB3 shows Works has the *highest* competition (median 4.0 bidders). Together
# these findings refute the narrative that low savings equals low competition. Works'
# savings shortfall is a cost-complexity and scope-uncertainty problem, not a market
# concentration problem.
#
# **Supplies: the one category where competition works as expected:** The bracket analysis
# in Section 5 shows Supplies savings rising from 50.2% (1 bidder) to 94.4% (>10 bidders).
# This is the cleanest evidence of competitive dynamics translating into procurement value.
# It also explains why Supplies has the highest savings rate in NB2 despite the lowest mean
# competition: commodity markets with even modest bidder counts clear efficiently.
#
# **Size segments:** The four tiers (Micro / Small / Mid / Large) defined in NB2 Section 11
# are reused in NB3 Sections 6 and 8. The flat competition profile across all tiers
# (median 2.0 EU-wide, median 1.0 within CPV 72) reinforces that value concentration
# in Large contracts is a financial phenomenon, not a consequence of competitive dynamics.
#
# ----------------------------------------------------------------------
#
# ### KPIs Defined in This Notebook
#
# | KPI | Definition | Source |
# |-----|-----------|--------|
# | Competition Intensity Score (CIS) | Median bidders per lot, notice-level | gold_awards → notice_awards |
# | % Solo contracts | % notices with only 1 bidder (tenders_count ≤ 1) | gold_awards |
# | Mean bidders per lot | Average tenders across lots per notice, then averaged across notices | gold_awards |
# | Competition-savings Pearson r | Pearson correlation between tenders_count and savings_pct | Matched subset (ex-HUN, ex-CHE) |
# | Competition-savings Spearman r | Rank-order correlation between tenders_count and savings_pct | Matched subset |
# | Market Competition Ranking | Country tier: High (>4) / Medium (2–4) / Low (≤2) based on median bidders | comp_country |
# | CPV competition profile | Median bidders and % solo within CPV 72 and CPV 48 | notice_awards |
