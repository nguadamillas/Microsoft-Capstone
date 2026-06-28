
# # Notebook 4 — Micro-Analysis: Contract Profiles, Buyer Archetypes & Category Intelligence
# **TED Procurement Intelligence | IE University × Microsoft**
# **Analytics + Semantic Layer | Data: January 2026**
#
# ----------------------------------------------------------------------
# ### What this notebook covers
# 1. CPV × Country cross-analysis — which country+category combinations are most and least competitive
# 2. Buyer profile segmentation — four archetypes from the top 100 buyers
# 3. Contract size × Country × CPV deep-dive — where Large contracts (>€5M) are won competitively
# 4. Savings opportunity map — a 2×2 matrix positioning every country by savings rate vs. competition
# 5. CPV 72 IT Services complete profile — the most detailed look at the core technology category
# 6. Summary cross-reference table — master reference across 15 countries on all key metrics
#
# > **Hungary inclusion policy (matching NB2 and NB3):**
# > - **Competition metrics only (tenders_count):** Hungary IS included — its bidder counts are normal and informative.
# > - **Financial metrics (spend, savings, buyer profiles):** Hungary is EXCLUDED — its 29 framework contracts recorded at exactly €5B each are ceiling values, not real transaction prices.
# > - Switzerland is excluded from savings analysis (0% estimated value coverage).
# > - `contract_value` capped at €5B per lot. All country financial aggregates computed at notice level first.
#


# ----------------------------------------------------------------------
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
import matplotlib.patches as mpatches
import seaborn as sns
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


# ── Build notice_awards exactly as NB2 and NB3 do ─────────────────────
# Group gold_awards (lot-level) to notice-level, keeping CPV and country columns.
# This is the same pattern as NB2 and NB3 — cpv_division types stay consistent.

CAP = 5_000_000_000
gold_awards['cv_clean'] = gold_awards['contract_value'].clip(upper=CAP)

notice_awards = (
    gold_awards
    .groupby(['notice_id', 'buyer_country', 'proc_type', 'cpv_division', 'cpv_division_name'])
    .agg(
        awarded_eur_total = ('cv_clean',      'sum'),
        lot_count         = ('notice_id',     'count'),
        tenders_count     = ('tenders_count', 'mean'),
    )
    .reset_index()
)

# Add estimated value and buyer_name from gold_notices
notice_awards = notice_awards.merge(
    gold_notices[['notice_id', 'estimated', 'buyer_name']],
    on='notice_id', how='left'
)

# — Financial subsets (Hungary excluded: framework ceiling values, not real spend) —
notice_awards_clean  = notice_awards[notice_awards['awarded_eur_total'] > 0].copy()
notice_awards_exhun  = notice_awards_clean[notice_awards_clean['buyer_country'] != 'HUN'].copy()

# — Competition subset (Hungary INCLUDED: tenders_count is valid) —
notice_awards_comp   = notice_awards_clean.copy()

# Contract size segments (matching NB2 Section 11)
def size_segment(v):
    if v < 50_000:    return 'Micro (<50K)'
    if v < 500_000:   return 'Small (50K-500K)'
    if v < 5_000_000: return 'Mid (500K-5M)'
    return 'Large (>5M)'

notice_awards_exhun['size_seg'] = notice_awards_exhun['awarded_eur_total'].apply(size_segment)
notice_awards_comp['size_seg']  = notice_awards_comp['awarded_eur_total'].apply(size_segment)

# Savings subset: estimated present, ex-HUN, ex-CHE
notice_savings = notice_awards_exhun[
    (notice_awards_exhun['estimated'].notna()) &
    (notice_awards_exhun['estimated'] > 0) &
    (notice_awards_exhun['buyer_country'] != 'CHE')
].copy()
notice_savings['savings_pct'] = (
    (notice_savings['estimated'] - notice_savings['awarded_eur_total'])
    / notice_savings['estimated'] * 100
)

SEG_ORDER = ['Micro (<50K)', 'Small (50K-500K)', 'Mid (500K-5M)', 'Large (>5M)']

print('=== DATASETS READY FOR NB4 ===')
print(f'notice_awards (all notices):       {len(notice_awards):>7,}')
print(f'notice_awards_clean (>0 spend):    {len(notice_awards_clean):>7,}')
print(f'notice_awards_exhun (ex-HUN):      {len(notice_awards_exhun):>7,}  ← financial analysis')
print(f'notice_awards_comp  (incl. HUN):   {len(notice_awards_comp):>7,}  ← competition analysis')
print(f'notice_savings:                    {len(notice_savings):>7,}  ← savings analysis')
print()
print('CPV division dtype:', notice_awards['cpv_division'].dtype)
print('Unique CPV divisions:', notice_awards['cpv_division'].nunique())
print()
print('Size segment distribution (ex-HUN):')
print(notice_awards_exhun['size_seg'].value_counts().reindex(SEG_ORDER).to_string())



# ----------------------------------------------------------------------
# ## 1. CPV × Country Cross-Analysis
#
# > **Key finding:** Competition intensity varies dramatically by category-country combination — not just by country alone. CPV 72 (IT Services) shows median 1 bidder in CZE, POL, ROU, BGR, and ESP, while ITA (4.0), CHE (4.0), FRA (3.0), and NLD (3.0) show genuinely open IT markets. Construction (CPV 45) is the most competitive category everywhere, with DEU reaching median 6.0 and ITA 8.5.
#
# > **Hungary:** included in competition metrics (tenders_count) here.
#

# Top 8 CPV divisions by notice count (competition dataset, includes HUN)
top_cpv_divs = (
    notice_awards_comp
    .groupby(['cpv_division', 'cpv_division_name'])
    .size()
    .reset_index(name='notice_count')
    .sort_values('notice_count', ascending=False)
    .head(8)
)
print('Top 8 CPV divisions by notice count:')
print(top_cpv_divs.to_string(index=False))


TOP_CPVS      = top_cpv_divs['cpv_division'].tolist()
TOP_COUNTRIES = ['DEU', 'POL', 'FRA', 'CZE', 'ESP', 'ROU', 'ITA', 'BGR', 'NLD', 'SWE']

print('CPV division dtype:', notice_awards_comp['cpv_division'].dtype)
print('TOP_CPVS:', TOP_CPVS)
print('Sample CPV values in data:', notice_awards_comp['cpv_division'].dropna().unique()[:10])

cpv_country = (
    notice_awards_comp[
        notice_awards_comp['cpv_division'].isin(TOP_CPVS) &
        notice_awards_comp['buyer_country'].isin(TOP_COUNTRIES)
    ]
    .groupby(['cpv_division', 'cpv_division_name', 'buyer_country'])
    .agg(
        notice_count   = ('notice_id',     'nunique'),
        median_bidders = ('tenders_count', 'median'),
    )
    .reset_index()
)

print(f'\nRows before notice filter: {len(cpv_country)}')

cpv_country_filt = cpv_country[cpv_country['notice_count'] >= 10].copy()
print(f'Rows after >= 10 notices filter: {len(cpv_country_filt)}')
print()
print(cpv_country_filt[['cpv_division_name','buyer_country','notice_count','median_bidders']].to_string(index=False))

cpv_label_map = {
    45: 'CPV 45\nConstruction',
    72: 'CPV 72\nIT Services',
    33: 'CPV 33\nMedical Equip.',
    79: 'CPV 79\nBusiness Svc.',
    34: 'CPV 34\nTransport',
    50: 'CPV 50\nRepair & Maint.',
    71: 'CPV 71\nArchitectural',
    48: 'CPV 48\nSoftware',
}

# Cast cpv_division to int so map lookup works regardless of stored type
cpv_country_filt['cpv_label'] = cpv_country_filt['cpv_division'].astype(int).map(cpv_label_map)

heatmap_pivot = cpv_country_filt.pivot_table(
    index='cpv_label',
    columns='buyer_country',
    values='median_bidders',
    aggfunc='median'
)

print('Pivot shape:', heatmap_pivot.shape)
print(heatmap_pivot.round(1).to_string())

if heatmap_pivot.empty:
    print('No data to plot — check CPV division type mismatch above.')
else:
    # Convert to float so seaborn handles missing values correctly
    heatmap_pivot = heatmap_pivot.astype(float)
    mask = heatmap_pivot.isna()

    fig, ax = plt.subplots(figsize=(14, 7))

    sns.heatmap(
        heatmap_pivot,
        ax=ax,
        cmap='YlOrRd',
        annot=True,
        fmt='.1f',
        linewidths=0.8,
        linecolor='#cccccc',
        cbar_kws={'label': 'Median Bidders per Contract', 'shrink': 0.8},
        vmin=1, vmax=6,
        mask=mask,
        annot_kws={'size': 11, 'weight': 'bold'},
    )
    sns.heatmap(
        heatmap_pivot.where(mask),
        ax=ax,
        cmap=['#f0f0f0'],
        cbar=False,
        linewidths=0.8,
        linecolor='#cccccc',
    )

    ax.set_title(
        'Competition Intensity Heatmap: Median Bidders by CPV Division and Country\n'
        'Top 8 CPV divisions × Top 10 countries  |  cells with ≥10 notices  |  grey = no data',
        fontsize=13, fontweight='bold', pad=16
    )
    ax.set_xlabel('Buyer Country', fontsize=12, labelpad=10)
    ax.set_ylabel('CPV Division', fontsize=12, labelpad=10)
    ax.tick_params(axis='y', rotation=0, labelsize=10)
    ax.tick_params(axis='x', labelsize=11)
    plt.tight_layout()
    plt.show()

# CPV 72 country competition preview — full profile in Section 5
cpv72_prev = notice_awards_comp[notice_awards_comp['cpv_division'].astype(int) == 72].copy()

cpv72_prev_ctry = (
    cpv72_prev
    .groupby('buyer_country')
    .agg(
        notices        = ('notice_id',     'nunique'),
        median_bidders = ('tenders_count', 'median'),
        mean_bidders   = ('tenders_count', 'mean'),
    )
    .reset_index()
)
cpv72_solo_prev = (
    cpv72_prev[cpv72_prev['tenders_count'].notna()]
    .groupby('buyer_country')
    .agg(solo_pct=('tenders_count', lambda x: (x == 1).mean() * 100))
    .reset_index()
)
cpv72_prev_ctry = (
    cpv72_prev_ctry
    .merge(cpv72_solo_prev, on='buyer_country', how='left')
    .query('notices >= 5')
    .sort_values('median_bidders', ascending=False)
)
print('CPV 72 competition by country (includes HUN, >= 5 notices):')
print(cpv72_prev_ctry[['buyer_country','notices','median_bidders','mean_bidders','solo_pct']].to_string(index=False))


# ### Section 1 — Findings
#
# - **Construction (CPV 45)** is the most competitive category across all countries in the dataset. DEU reaches median 6.0 bidders, ESP 6.0, ITA 8.5 — the highest values in the heatmap. BGR (1.0) is the only exception, reflecting a structurally low-competition market even for construction.
# - **Architectural & Engineering (CPV 71)** is also highly competitive, particularly in ITA (10.0 median bidders) and FRA (4.0). Italian law requires open architectural competitions, which explains the outlier figure.
# - **IT Services (CPV 72)** is the most consistently low-competition category. CZE, POL, ROU, BGR, and POL all sit at median 1.0 bidder. DEU reaches only 1.5 and ESP 2.0. The exceptions are ITA (4.0), FRA (3.0), and NLD (3.0) — the only genuinely open IT markets among the top 10 countries.
# - **Medical Equipment (CPV 33)** is largely single-bidder dominated across Eastern Europe (CZE, ESP, POL all at 1.0) with slightly more competition in ITA (2.3) and ROU (2.2).
# - **Key insight:** country-level competition rankings from NB3 conceal sharp within-country variation. CPV 72 in ITA (median 4.0) is a fundamentally different market from CPV 72 in CZE (median 1.0), despite Italy ranking below CZE in overall notice volume.
#


# ----------------------------------------------------------------------
# ## 2. Buyer Profile Segmentation
#
# > **Key finding:** The top 100 buyers (ex-HUN) account for €300.8B in awarded spend, split across four distinct archetypes with fundamentally different procurement behaviours. Mega Buyers (28 institutions) alone represent €155.9B — 51.8% of the top-100 total. Hungary is excluded from financial metrics — its framework ceiling values would distort the segmentation.
#

buyer_profile = (
    notice_awards_exhun
    .groupby(['buyer_name', 'buyer_country'])
    .agg(
        total_spend_eur = ('awarded_eur_total', 'sum'),
        notice_count    = ('notice_id',          'nunique'),
        median_cv       = ('awarded_eur_total',  'median'),
    )
    .reset_index()
    .sort_values('total_spend_eur', ascending=False)
    .head(100)
    .reset_index(drop=True)
)

buyer_cpv = (
    notice_awards_exhun
    .groupby(['buyer_name', 'cpv_division_name'])
    .size()
    .reset_index(name='n')
    .sort_values('n', ascending=False)
    .drop_duplicates(subset='buyer_name')
    [['buyer_name', 'cpv_division_name']]
    .rename(columns={'cpv_division_name': 'dominant_cpv'})
)
buyer_profile = buyer_profile.merge(buyer_cpv, on='buyer_name', how='left')

print(f'Top 100 buyers (ex-HUN). Total spend: {fmt_eur(buyer_profile["total_spend_eur"].sum())}')
print()
print(
    buyer_profile
    .assign(spend_m=lambda d: (d['total_spend_eur']/1e6).round(1))
    [['buyer_name', 'buyer_country', 'spend_m', 'notice_count', 'dominant_cpv']]
    .head(15)
    .to_string(index=False)
)


spend_med  = buyer_profile['total_spend_eur'].median()
volume_med = buyer_profile['notice_count'].median()

def assign_archetype(row):
    hi_s = row['total_spend_eur'] >= spend_med
    hi_v = row['notice_count']    >= volume_med
    if hi_s and hi_v:      return 'Mega Buyer'
    if hi_s and not hi_v:  return 'High-Value Low-Volume'
    if not hi_s and hi_v:  return 'High-Volume Low-Value'
    return 'Balanced'

buyer_profile['archetype'] = buyer_profile.apply(assign_archetype, axis=1)

print(f'Spend median:  {fmt_eur(spend_med)}  |  Volume median: {volume_med:.0f} notices')
print()
print('Archetype distribution:')
print(buyer_profile['archetype'].value_counts().to_string())


arch_summary = (
    buyer_profile
    .groupby('archetype')
    .agg(
        n_buyers       = ('buyer_name',       'count'),
        total_spend_bn = ('total_spend_eur',  lambda x: x.sum()/1e9),
        avg_spend_m    = ('total_spend_eur',  lambda x: x.mean()/1e6),
        avg_notices    = ('notice_count',     'mean'),
    )
    .reset_index()
    .sort_values('total_spend_bn', ascending=False)
)
print('Buyer Archetype Summary:')
print(arch_summary.to_string(index=False))


archetype_colours = {
    'Mega Buyer':             C_RED,
    'High-Value Low-Volume':  C_BLUE,
    'High-Volume Low-Value':  C_GREEN,
    'Balanced':               C_ORANGE,
}

fig, ax = plt.subplots(figsize=(13, 8))

for arch, grp in buyer_profile.groupby('archetype'):
    ax.scatter(
        grp['notice_count'],
        grp['total_spend_eur'] / 1e9,
        color=archetype_colours[arch],
        label=arch, s=100, alpha=0.8,
        edgecolors='white', linewidth=0.8, zorder=3
    )

ax.axvline(volume_med, color=C_GREY, linestyle='--', linewidth=1.2, alpha=0.7, zorder=2)
ax.axhline(spend_med / 1e9, color=C_GREY, linestyle='--', linewidth=1.2, alpha=0.7, zorder=2)

# Quadrant labels
xlim = ax.get_xlim(); ylim = ax.get_ylim()
kw = dict(fontsize=8.5, alpha=0.7, fontstyle='italic')
ax.text(xlim[0]+1, ylim[1]*0.97, 'HIGH VALUE\nLOW VOLUME', color=C_BLUE, va='top', **kw)
ax.text(xlim[1]*0.55, ylim[1]*0.97, 'MEGA BUYERS', color=C_RED, va='top', **kw)
ax.text(xlim[0]+1, ylim[0]+0.3, 'BALANCED', color=C_ORANGE, va='bottom', **kw)
ax.text(xlim[1]*0.55, ylim[0]+0.3, 'HIGH VOLUME\nLOW VALUE', color=C_GREEN, va='bottom', **kw)

# Label top 12 by spend
for _, row in buyer_profile.head(12).iterrows():
    label = (row['buyer_name'][:24] + '...') if len(row['buyer_name']) > 24 else row['buyer_name']
    ax.annotate(
        f"{label}\n({row['buyer_country']})",
        xy=(row['notice_count'], row['total_spend_eur']/1e9),
        xytext=(7, 5), textcoords='offset points',
        fontsize=6.5, color='#222222',
        arrowprops=dict(arrowstyle='-', color='#aaaaaa', lw=0.5)
    )

ax.set_xlabel('Number of Notices (Volume)', fontsize=12)
ax.set_ylabel('Total Awarded Spend (€B)', fontsize=12)
ax.set_title(
    'Buyer Archetype Segmentation — Top 100 Buyers (ex-HUN)\n'
    'Dashed lines = median spend and median notice count',
    fontsize=13, fontweight='bold'
)
ax.legend(fontsize=10, framealpha=0.95, loc='upper right')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'€{x:.0f}B'))
plt.tight_layout()
plt.show()


for arch in ['Mega Buyer', 'High-Value Low-Volume', 'High-Volume Low-Value', 'Balanced']:
    sub = buyer_profile[buyer_profile['archetype'] == arch].head(5)
    print(f'\n── {arch} ──')
    print(
        sub
        .assign(spend_m=lambda d: (d['total_spend_eur']/1e6).round(1))
        [['buyer_name', 'buyer_country', 'spend_m', 'notice_count', 'dominant_cpv']]
        .to_string(index=False)
    )



# ### Section 2 — Findings
#
# - **Mega Buyers** (28 institutions, €155.9B total, avg. €5.6B per buyer, avg. 42 contracts): DPP Prague (€30.5B, 170 contracts), Region Stockholm (€13.3B, 8 contracts), Ministerstvo vnitra CZE (€12.5B, 99 contracts), České dráhy (€10.4B, 81 contracts). Prolific and high-spend — central purchasing bodies and state enterprises.
# - **High-Value Low-Volume** (22 institutions, €91.9B total, avg. €4.2B per buyer, avg. 3 contracts): Sinfra Sweden (€11.6B, 4 contracts), Isavia Iceland (€9.4B, 4 contracts), CFR Romania (€6.2B, 5 contracts). Few but extremely large contracts — relationship-driven procurement where new vendors have minimal access without a prior track record.
# - **High-Volume Low-Value** (29 institutions, €30.1B total, avg. €1.0B per buyer, avg. 16 contracts): The most accessible entry point. Frequent procurement activity with smaller individual contract values means lower incumbent dependency and more opportunities to establish a reference.
# - **Balanced** (21 institutions, €22.9B total, avg. €1.1B per buyer, avg. 2 contracts): Mid-market buyers with consistent but infrequent procurement. GIP RESAH France (€6.5B, 6 contracts) is the largest in this group.
#


# ----------------------------------------------------------------------
# ## 3. Contract Size × Country × CPV Deep-Dive
#
# > **Spend** uses `notice_awards_exhun` (ex-HUN). **Competition** uses `notice_awards_comp` (includes HUN).
# > Large contracts (>€5M) represent 19.0% of notices but 95.5% of total awarded spend. Overall Large contract median: 3.0 bidders.
#

# Size overview — spend from exhun, competition from comp
size_spend = (
    notice_awards_exhun
    .groupby('size_seg')
    .agg(
        notices  = ('notice_id',         'nunique'),
        spend_bn = ('awarded_eur_total', lambda x: x.sum()/1e9),
        raw_spend= ('awarded_eur_total', 'sum'),
    )
    .reset_index()
)
total_spend = notice_awards_exhun['awarded_eur_total'].sum()
size_spend['pct_spend'] = (size_spend['raw_spend'] / total_spend * 100).round(1)

size_comp = (
    notice_awards_comp[notice_awards_comp['tenders_count'].notna()]
    .groupby('size_seg')
    .agg(
        median_bid = ('tenders_count', 'median'),
        solo_pct   = ('tenders_count', lambda x: (x == 1).mean() * 100),
    )
    .reset_index()
)

size_overview = (
    size_spend
    .merge(size_comp, on='size_seg', how='left')
    .drop(columns='raw_spend')
)
size_overview['_ord'] = size_overview['size_seg'].map({s:i for i,s in enumerate(SEG_ORDER)})
size_overview = size_overview.sort_values('_ord').drop(columns='_ord')

print('Size segment overview (spend ex-HUN | competition incl. HUN):')
print(size_overview.to_string(index=False))


# CPV dominance within each size segment (spend, ex-HUN)
for seg in SEG_ORDER:
    sub = notice_awards_exhun[notice_awards_exhun['size_seg'] == seg]
    sub_comp = notice_awards_comp[
        (notice_awards_comp['size_seg'] == seg) &
        notice_awards_comp['tenders_count'].notna()
    ]
    n_total = sub['notice_id'].nunique()
    med_bid = sub_comp['tenders_count'].median()
    solo    = (sub_comp['tenders_count'] == 1).mean() * 100
    top_cpv = (
        sub.groupby('cpv_division_name')
        .agg(n=('notice_id','nunique'), spend_m=('awarded_eur_total', lambda x: x.sum()/1e6))
        .sort_values('n', ascending=False)
        .head(5)
    )
    print(f'\n── {seg} | {n_total:,} notices | Med bidders: {med_bid:.1f} | Solo: {solo:.1f}%')
    print(top_cpv.to_string())


# Large contracts: competition by country (incl. HUN) vs spend (ex-HUN)
large_comp = (
    notice_awards_comp[
        (notice_awards_comp['size_seg'] == 'Large (>5M)') &
        notice_awards_comp['tenders_count'].notna()
    ]
    .groupby('buyer_country')
    .agg(
        notices_comp   = ('notice_id',     'nunique'),
        median_bidders = ('tenders_count', 'median'),
        solo_pct       = ('tenders_count', lambda x: (x==1).mean()*100),
    )
    .reset_index()
)
large_spend = (
    notice_awards_exhun[notice_awards_exhun['size_seg'] == 'Large (>5M)']
    .groupby('buyer_country')
    .agg(spend_bn=('awarded_eur_total', lambda x: x.sum()/1e9))
    .reset_index()
)
large_combined = (
    large_comp
    .merge(large_spend, on='buyer_country', how='left')
    .query('notices_comp >= 10')
    .sort_values('median_bidders', ascending=False)
)
overall_large_med = large_comp['median_bidders'].median()
print(f'Overall Large contract median bidders (incl. HUN): {overall_large_med:.1f}')
print()
print(large_combined.to_string(index=False))


plot_df     = large_combined.dropna(subset=['median_bidders']).sort_values('median_bidders', ascending=True)
bar_colours = [C_GREEN if m > overall_large_med else C_BLUE for m in plot_df['median_bidders']]

fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh(
    plot_df['buyer_country'], plot_df['median_bidders'],
    color=bar_colours, edgecolor='white', linewidth=0.6, height=0.65
)
ax.axvline(overall_large_med, color=C_RED, linestyle='--', linewidth=1.8,
           label=f'Dataset median ({overall_large_med:.1f} bidders)', zorder=3)

for bar, val, solo in zip(bars, plot_df['median_bidders'], plot_df['solo_pct']):
    ax.text(
        val + 0.06, bar.get_y() + bar.get_height()/2,
        f'{val:.1f} bidders | {solo:.0f}% solo',
        va='center', fontsize=8.5, color='#333333'
    )

ax.set_xlabel('Median Bidders per Large Contract (>€5M)', fontsize=12)
ax.set_title(
    'Competition in Large Contracts (>€5M) by Country\n'
    'Green = above dataset median  |  incl. HUN for competition  |  ≥10 notices',
    fontsize=12, fontweight='bold'
)
ax.legend(fontsize=10, framealpha=0.9)
ax.set_xlim(0, plot_df['median_bidders'].max() + 2.5)
plt.tight_layout()
plt.show()



# ### Section 3 — Findings
#
# - **Micro (<€50K)** — 3,685 notices (13.7% of total), essentially 0% of spend. Solo rate 37.7%. These are largely framework call-offs or direct awards below the meaningful tendering threshold.
# - **Large (>€5M)** — 5,113 notices (19.0% of total), 95.5% of spend (€425.2B). Solo rate 31.2% — only 6.5 points below Micro despite the vastly higher contract value. The incremental competitive barrier above €5M is smaller than the value concentration implies.
# - **Median bidders is 2.0 across all four size segments.** Size alone does not drive competition. What drives competition is category and country, as shown in Section 1.
# - **Within Large contracts**, IRL (5.0) and FIN (5.0) are the most competitive markets. POL stands out negatively: median 1.0 bidder and 60.5% solo rate on large contracts — the highest lock-in rate in the dataset at that contract tier. CZE (median 2.0, 44.8% solo) and HUN (median 2.0, 37.7% solo) also show significant incumbent dominance at scale.
# - **By CPV within Large contracts**, Construction (CPV 45) dominates by notice count (912 notices, €93.6B). IT services ranks 5th by count (275 notices) but represents €24.6B — a high average contract value of €89M per notice.
#


# ----------------------------------------------------------------------
# ## 4. Savings Opportunity Map (2×2 Matrix)
#
# > Savings: ex-HUN, ex-CHE (0% estimated coverage). Competition: includes HUN.
# > The savings threshold (median of country medians) is **2.9%**. The competition threshold is **2.0 median bidders**.
# > Countries with both high savings and low competition represent the easiest procurement entry points.
#

country_savings = (
    notice_savings
    .groupby('buyer_country')
    .agg(
        n_savings      = ('notice_id',   'count'),
        median_savings = ('savings_pct', 'median'),
    )
    .reset_index()
)
# Competition: include HUN
country_comp = (
    notice_awards_comp[notice_awards_comp['tenders_count'].notna()]
    .groupby('buyer_country')
    .agg(
        n_comp         = ('notice_id',     'count'),
        median_bidders = ('tenders_count', 'median'),
        solo_rate_pct  = ('tenders_count', lambda x: (x==1).mean()*100),
    )
    .reset_index()
)
opp_map = (
    country_savings
    .merge(country_comp, on='buyer_country', how='inner')
    .query('n_savings >= 30 and n_comp >= 30')
)
print(f'Countries in opportunity map: {len(opp_map)}')
print(opp_map.sort_values('median_savings', ascending=False).to_string(index=False))


savings_thresh = opp_map['median_savings'].median()
bidders_thresh = opp_map['median_bidders'].median()

def assign_quadrant(row):
    hi_s = row['median_savings'] >= savings_thresh
    hi_c = row['median_bidders'] >= bidders_thresh
    if hi_s and not hi_c: return 'High Savings / Low Competition'
    if hi_s and hi_c:     return 'High Savings / High Competition'
    if not hi_s and hi_c: return 'Low Savings / High Competition'
    return 'Low Savings / Low Competition'

opp_map['quadrant'] = opp_map.apply(assign_quadrant, axis=1)
print(f'Savings threshold: {savings_thresh:.1f}%  |  Bidders threshold: {bidders_thresh:.1f}')
print()
print(opp_map[['buyer_country','median_savings','median_bidders','quadrant']].sort_values('quadrant').to_string(index=False))


quad_colours = {
    'High Savings / Low Competition':  C_GREEN,
    'High Savings / High Competition': C_TEAL,
    'Low Savings / Low Competition':   C_ORANGE,
    'Low Savings / High Competition':  C_RED,
}

fig, ax = plt.subplots(figsize=(13, 9))

for quad, grp in opp_map.groupby('quadrant'):
    ax.scatter(
        grp['median_bidders'], grp['median_savings'],
        color=quad_colours[quad], label=quad,
        s=grp['n_savings'] * 0.6 + 80,
        alpha=0.85, edgecolors='white', linewidth=0.8, zorder=3
    )

for _, row in opp_map.iterrows():
    ax.annotate(
        row['buyer_country'],
        xy=(row['median_bidders'], row['median_savings']),
        xytext=(5, 5), textcoords='offset points',
        fontsize=10, fontweight='bold', color='#222222'
    )

ax.axvline(bidders_thresh, color=C_GREY, linestyle='--', linewidth=1.5, alpha=0.6, zorder=2)
ax.axhline(savings_thresh, color=C_GREY, linestyle='--', linewidth=1.5, alpha=0.6, zorder=2)

# Light background shading per quadrant
xlim = ax.get_xlim(); ylim = ax.get_ylim()
ax.fill_betweenx([savings_thresh, ylim[1]], xlim[0], bidders_thresh, color=C_GREEN, alpha=0.04)
ax.fill_betweenx([savings_thresh, ylim[1]], bidders_thresh, xlim[1],  color=C_TEAL,  alpha=0.04)
ax.fill_betweenx([ylim[0], savings_thresh], xlim[0], bidders_thresh, color=C_ORANGE,alpha=0.04)
ax.fill_betweenx([ylim[0], savings_thresh], bidders_thresh, xlim[1],  color=C_RED,   alpha=0.04)

kw = dict(fontsize=9, fontweight='bold', alpha=0.65)
ax.text(xlim[0]+0.05, ylim[1]-1, 'HIGH SAVINGS\nLOW COMPETITION\n→ Easiest entry', color=C_GREEN, va='top', **kw)
ax.text(xlim[1]-0.05, ylim[1]-1, 'HIGH SAVINGS\nHIGH COMPETITION', color=C_TEAL, va='top', ha='right', **kw)
ax.text(xlim[0]+0.05, ylim[0]+0.5,'LOW SAVINGS\nLOW COMPETITION\n→ Incumbent-friendly', color=C_ORANGE, va='bottom', **kw)
ax.text(xlim[1]-0.05, ylim[0]+0.5,'LOW SAVINGS\nHIGH COMPETITION\n→ Thin margins', color=C_RED, va='bottom', ha='right', **kw)

ax.set_xlabel('Median Bidders per Contract (Competition Intensity)', fontsize=12)
ax.set_ylabel('Median Savings % (Awarded vs. Estimated)', fontsize=12)
ax.set_title(
    'Procurement Opportunity Map — Country Positioning\n'
    'Bubble size = contracts with savings data  |  savings: ex-HUN, ex-CHE  |  competition: incl. HUN',
    fontsize=12, fontweight='bold'
)
handles = [mpatches.Patch(color=c, label=q) for q, c in quad_colours.items()]
ax.legend(handles=handles, fontsize=9, loc='center right', framealpha=0.9)
plt.tight_layout()
plt.show()



# ### Section 4 — Findings
#
# - **High Savings / Low Competition** — HRV (75.0% median savings, 1.85 median bidders) and BGR (59.6%, 1.17): buyers consistently award contracts well below estimates with thin bidder fields. Pricing headroom is high and competitive pressure is low.
# - **High Savings / High Competition** — SVK (95.1%, 2.0), CZE (75.0%, 2.0), ESP (30.1%, 2.0), ITA (33.4%, 2.75), BEL (14.6%, 2.0), MLT (15.2%, 3.0), LTU (8.7%, 2.1): contracts award below estimate but at least two vendors typically compete. Good value markets where genuine tendering occurs.
# - **Low Savings / High Competition** — DEU (2.5%, 4.0), FRA (0.0%, 2.7), NOR (0.0%, 3.0), DNK (0.0%, 2.5), FIN (0.8%, 3.0), IRL (0.0%, 3.0), NLD (0.0%, 2.0), SWE (0.0%, 2.0), AUT (0.0%, 3.0): procurement is efficiently priced — awarded values match estimates closely and competition is real. Harder to win on price alone; differentiation on quality and capability matters more.
# - **Low Savings / Low Competition** — POL (2.4%, 1.0): low savings combined with the lowest competition of any country in the map. Poland's market is neither price-competitive nor open — a strong incumbent environment.
# - **Data quality note:** several Western European countries (FRA, SWE, NOR, NLD, IRL, DNK, AUT) show 0.0% median savings. This reflects low estimated value coverage in those countries rather than perfect budget accuracy.
#


# ----------------------------------------------------------------------
# ## 5. CPV 72 IT Services — Complete Market Profile
#
# > **€25.6B awarded spend (ex-HUN) | 1,389 notices | median contract €735K | mean contract €18.4M**
# > Median bidders: 1.0 | Solo rate: 51.7% | Mean bidders: 2.56
# > Competition metrics include HUN. Spend metrics exclude HUN.
#

cpv72_fin  = notice_awards_exhun[notice_awards_exhun['cpv_division'].astype(int) == 72].copy()
# Competition profile — incl. HUN
cpv72_comp = notice_awards_comp[notice_awards_comp['cpv_division'].astype(int) == 72].copy()

print(f'cpv72_fin  rows: {len(cpv72_fin)}')
print(f'cpv72_comp rows: {len(cpv72_comp)}')

tc = cpv72_comp['tenders_count'].dropna()
print('CPV 72 IT Services — baseline statistics')
print(f'  Notices (ex-HUN, financial): {cpv72_fin["notice_id"].nunique():,}')
print(f'  Total spend (ex-HUN):        {fmt_eur(cpv72_fin["awarded_eur_total"].sum())}')
print(f'  Median spend/notice:         {fmt_eur(cpv72_fin["awarded_eur_total"].median())}')
print(f'  Mean spend/notice:           {fmt_eur(cpv72_fin["awarded_eur_total"].mean())}')
print()
print(f'  Notices with tenders_count (incl. HUN): {len(tc):,}')
print(f'  Median bidders:              {tc.median():.1f}')
print(f'  Mean bidders:                {tc.mean():.2f}')
print(f'  Solo rate (1 bidder):        {(tc==1).mean()*100:.1f}%')
print(f'  5+ bidders rate:             {(tc>=5).mean()*100:.1f}%')


# Spend by country (ex-HUN)
cpv72_spend = (
    cpv72_fin
    .groupby('buyer_country')
    .agg(
        notices_fin    = ('notice_id',         'nunique'),
        spend_bn       = ('awarded_eur_total', lambda x: x.sum()/1e9),
        median_cv_m    = ('awarded_eur_total', lambda x: x.median()/1e6),
    )
    .reset_index()
)
# Competition by country (incl. HUN)
cpv72_comp_ctry = (
    cpv72_comp
    .groupby('buyer_country')
    .agg(
        notices_comp   = ('notice_id',     'nunique'),
        median_bidders = ('tenders_count', 'median'),
    )
    .reset_index()
)
cpv72_solo_ctry = (
    cpv72_comp[cpv72_comp['tenders_count'].notna()]
    .groupby('buyer_country')
    .agg(solo_pct=('tenders_count', lambda x: (x==1).mean()*100))
    .reset_index()
)

cpv72_ctry = (
    cpv72_spend
    .merge(cpv72_comp_ctry, on='buyer_country', how='outer')
    .merge(cpv72_solo_ctry, on='buyer_country', how='left')
    .query('notices_fin >= 5 or notices_comp >= 5')
    .sort_values('spend_bn', ascending=False, na_position='last')
)
print('CPV 72 — Spend (ex-HUN) and Competition (incl. HUN) by Country:')
print(
    cpv72_ctry
    [['buyer_country','notices_fin','spend_bn','median_cv_m','median_bidders','solo_pct']]
    .rename(columns={'notices_fin':'Notices','spend_bn':'Spend(€B)',
                     'median_cv_m':'Med.CV(€M)','median_bidders':'Med.Bid','solo_pct':'Solo%'})
    .to_string(index=False)
)


fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('CPV 72 IT Services — Spend and Competition Profile\n'
             'Spend: ex-HUN  |  Competition: incl. HUN',
             fontsize=13, fontweight='bold')

# Left: spend by country (top 12, ex-HUN)
top12 = cpv72_ctry.dropna(subset=['spend_bn']).head(12)
bar_cols = [C_GOLD if c == 'CZE' else C_BLUE for c in top12['buyer_country']]
ax = axes[0]
bars = ax.barh(
    top12['buyer_country'][::-1], top12['spend_bn'][::-1],
    color=bar_cols[::-1], edgecolor='white', linewidth=0.5, height=0.65
)
for bar, val in zip(bars, top12['spend_bn'][::-1]):
    ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
            f'€{val:.1f}B', va='center', fontsize=9, fontweight='bold')
ax.set_xlabel('Total Awarded Spend (€B)', fontsize=11)
ax.set_title('Total IT Spend by Country (ex-HUN)', fontsize=11, fontweight='bold')
ax.set_xlim(0, top12['spend_bn'].max() * 1.25)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'€{x:.0f}B'))

# Right: solo rate vs median bidders (incl. HUN)
comp_plot = cpv72_ctry.dropna(subset=['solo_pct', 'median_bidders', 'notices_comp'])
dot_cols  = [C_GREEN if m >= 2 else C_RED for m in comp_plot['median_bidders']]
ax = axes[1]
sc = ax.scatter(
    comp_plot['solo_pct'], comp_plot['median_bidders'],
    c=dot_cols, s=comp_plot['notices_comp'] * 4 + 50,
    alpha=0.8, edgecolors='white', linewidth=0.8, zorder=3
)
for _, row in comp_plot.iterrows():
    ax.annotate(
        row['buyer_country'],
        xy=(row['solo_pct'], row['median_bidders']),
        xytext=(5, 4), textcoords='offset points',
        fontsize=9, fontweight='bold'
    )
ax.axhline(2.0, color=C_GREY, linestyle='--', linewidth=1.2, alpha=0.7,
           label='Median bidders = 2 threshold')
ax.set_xlabel('Solo Rate % (single-bidder contracts)', fontsize=11)
ax.set_ylabel('Median Bidders per Contract', fontsize=11)
ax.set_title('IT Competition by Country (incl. HUN)\nbubble size = notice count', fontsize=11, fontweight='bold')

green_patch = mpatches.Patch(color=C_GREEN, label='Median ≥ 2 bidders (open)')
red_patch   = mpatches.Patch(color=C_RED,   label='Median < 2 bidders (closed)')
ax.legend(handles=[green_patch, red_patch, plt.Line2D([0],[0],color=C_GREY,linestyle='--',label='= 2 threshold')],
          fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.show()


# CPV 72 by contract size (spend ex-HUN, competition incl. HUN)
cpv72_fin['size_seg']  = cpv72_fin['awarded_eur_total'].apply(size_segment)
cpv72_comp['size_seg'] = cpv72_comp['awarded_eur_total'].apply(size_segment)

cpv72_size_spend = (
    cpv72_fin
    .groupby('size_seg')
    .agg(notices=('notice_id','nunique'), spend_bn=('awarded_eur_total', lambda x: x.sum()/1e9))
    .reset_index()
)
cpv72_size_comp = (
    cpv72_comp[cpv72_comp['tenders_count'].notna()]
    .groupby('size_seg')
    .agg(
        median_bidders = ('tenders_count', 'median'),
        solo_pct       = ('tenders_count', lambda x: (x==1).mean()*100),
    )
    .reset_index()
)
cpv72_size = cpv72_size_spend.merge(cpv72_size_comp, on='size_seg', how='left')
cpv72_size['_ord'] = cpv72_size['size_seg'].map({s:i for i,s in enumerate(SEG_ORDER)})
cpv72_size = cpv72_size.sort_values('_ord').drop(columns='_ord')
print('CPV 72 — by contract size (spend ex-HUN | competition incl. HUN):')
print(cpv72_size.to_string(index=False))


# CPV 72 savings by country (where estimated value is available)
cpv72_sav = notice_savings[notice_savings['cpv_division'].astype(int) == 72].copy()

if len(cpv72_sav) > 0:
    cpv72_sav_ctry = (
        cpv72_sav
        .groupby('buyer_country')
        .agg(
            n            = ('notice_id',   'count'),
            med_savings  = ('savings_pct', 'median'),
            mean_savings = ('savings_pct', 'mean'),
        )
        .reset_index()
        .query('n >= 5')
        .sort_values('med_savings', ascending=False)
    )
    print('CPV 72 savings by country (ex-HUN, ex-CHE, >= 5 notices with estimated value):')
    print(cpv72_sav_ctry.to_string(index=False))
else:
    print('No savings data available for CPV 72 in this dataset.')

# Top 20 CPV 72 contracts by value (ex-HUN)
cpv72_top20 = (
    cpv72_fin
    .sort_values('awarded_eur_total', ascending=False)
    .head(20)
    [['notice_id','buyer_name','buyer_country','awarded_eur_total','tenders_count','size_seg']]
    .copy()
)
cpv72_top20['spend_m'] = (cpv72_top20['awarded_eur_total']/1e6).round(1)
print('Top 20 CPV 72 IT Services contracts by awarded value (ex-HUN):')
print(
    cpv72_top20.drop(columns='awarded_eur_total')
    .rename(columns={'notice_id':'Notice','buyer_name':'Buyer','buyer_country':'Country',
                     'tenders_count':'Bidders','size_seg':'Segment','spend_m':'Spend(€M)'})
    .to_string(index=False)
)


# Open vs closed IT market classification
it_class     = cpv72_ctry[cpv72_ctry['notices_comp'] >= 10].copy()
spend_med_it = it_class['spend_bn'].median()

def classify_it(row):
    hi_spend = pd.notna(row['spend_bn']) and row['spend_bn'] >= spend_med_it
    hi_comp  = pd.notna(row['median_bidders']) and row['median_bidders'] >= 2.0
    if hi_spend and hi_comp:     return 'Open IT Market'
    if hi_spend and not hi_comp: return 'Closed IT Market (high spend, solo-dominated)'
    if not hi_spend and hi_comp: return 'Competitive Small IT Market'
    return 'Low-Spend Low-Competition'

it_class['market_type'] = it_class.apply(classify_it, axis=1)
print('CPV 72 Market Classification (>= 10 notices):')
print(
    it_class[['buyer_country','notices_fin','spend_bn','median_bidders','solo_pct','market_type']]
    .sort_values('spend_bn', ascending=False, na_position='last')
    .to_string(index=False)
)



# ### Section 5 — Findings
#
# - **Scale:** €25.6B in awarded IT spend across 1,389 notices (ex-HUN). The distribution is extremely right-skewed — median contract is €735K but mean is €18.4M, driven by a small number of very large contracts. The top contract alone (ČEZ Distribuce CZE) is €4.5B.
# - **Competition:** median 1.0 bidder, 51.7% solo rate. More than half of all IT contracts attract exactly one bidder. IT procurement is structurally incumbent-dominated across Europe, reflecting integration continuity requirements and high switching costs.
# - **Size dynamics:** competition does not improve meaningfully until the Large tier (>€5M), which reaches median 2.0 bidders and 48.9% solo rate. Below €5M, IT contracts are 50–54% solo regardless of size segment.
# - **Country spend concentration:** CZE (€11.6B, 62.8% solo), ROU (€6.1B, 64.2% solo), and SWE (€4.2B, 41.0% solo) account for 85.5% of all IT spend (ex-HUN). CZE is the largest IT market but the most closed. ROU is similarly closed. SWE is the exception — large spend with a relatively open profile.
# - **Open IT markets** (median ≥2.0 bidders): ITA (4.0, 36.4% solo), CHE (4.0, 20.0%), FRA (3.0, 25.6%), NLD (3.0, 21.9%), NOR (3.0, 8.3%), GRC (3.0, 0.0%), IRL (3.0, 25.0%). These markets have genuine competitive tendering for IT contracts.
# - **Closed IT markets** (median <2.0 bidders, meaningful spend): POL (1.0, 74.2% solo), HRV (1.0, 85.7%), BGR (1.0, 81.6%), AUT (1.0, 53.8%), BEL (1.0, 57.9%). These are effectively locked by incumbents.
# - **Savings in IT:** SVK (98.9%), HRV (73.5%), and ESP (50.0%) show the highest IT-specific savings rates. CZE mean savings is anomalous (€-2.76B) due to a single outlier contract with an inflated estimated value — the median (43.0%) is the correct figure to use.
# - **Strategic implication:** the optimal entry profile for a new IT vendor is a Large contract (>€5M) in an Open IT Market country. ITA, CHE, NOR, NLD, and FRA offer real competitive opportunities. A single contract win in these markets provides the EU public sector reference needed to pursue larger volumes elsewhere.
#


# ----------------------------------------------------------------------
# ## 6. Summary Cross-Reference Table
#
# > Master reference for the entire analytical project. Top 15 countries by awarded spend (ex-HUN).
# > **Spend / Savings / Dominant CPV / Top Buyer:** ex-HUN. **Median Bidders / Solo Rate:** includes HUN.
# > NaN in Median Savings indicates insufficient estimated value coverage for that country (CHE, ISL).
#
# | KPI | Definition | Hungary |
# |---|---|---|
# | **Total Awarded Spend (€B)** | Sum of awarded values, €5B cap, aggregated to notice level | Excluded |
# | **Notice Count** | Unique notices with valid awarded value | Excluded |
# | **Median Bidders** | 50th percentile of tenders_count across notices with valid data | Included |
# | **Solo Rate (%)** | % of notices where tenders_count = 1 | Included |
# | **Median Savings (%)** | (Estimated − Awarded) ÷ Estimated × 100, notice-level median | Excluded |
# | **Dominant CPV** | CPV division with the most notices for that country | Excluded |
# | **Top Buyer** | Buyer with the largest total awarded spend in that country | Excluded |
#

col_spend = (
    notice_awards_exhun
    .groupby('buyer_country')
    .agg(spend_bn=('awarded_eur_total', lambda x: x.sum()/1e9), notice_count=('notice_id','nunique'))
    .reset_index()
)
col_comp = (
    notice_awards_comp[notice_awards_comp['tenders_count'].notna()]
    .groupby('buyer_country')
    .agg(
        median_bidders = ('tenders_count', 'median'),
        solo_rate_pct  = ('tenders_count', lambda x: (x==1).mean()*100),
    )
    .reset_index()
)
col_savings = (
    notice_savings
    .groupby('buyer_country')
    .agg(median_savings_pct=('savings_pct','median'))
    .reset_index()
)
col_cpv = (
    notice_awards_exhun
    .groupby(['buyer_country','cpv_division_name']).size().reset_index(name='n')
    .sort_values('n', ascending=False).drop_duplicates(subset='buyer_country')
    [['buyer_country','cpv_division_name']].rename(columns={'cpv_division_name':'dominant_cpv'})
)
col_buyer = (
    notice_awards_exhun
    .groupby(['buyer_country','buyer_name'])['awarded_eur_total'].sum().reset_index()
    .sort_values('awarded_eur_total', ascending=False)
    .drop_duplicates(subset='buyer_country')
    [['buyer_country','buyer_name']].rename(columns={'buyer_name':'top_buyer'})
)

summary = (
    col_spend
    .merge(col_comp,    on='buyer_country', how='left')
    .merge(col_savings, on='buyer_country', how='left')
    .merge(col_cpv,     on='buyer_country', how='left')
    .merge(col_buyer,   on='buyer_country', how='left')
    .sort_values('spend_bn', ascending=False)
    .head(15)
    .reset_index(drop=True)
)
summary.index = summary.index + 1

print('Master Cross-Reference Table — Top 15 Countries:')
print(summary[['buyer_country','spend_bn','notice_count','median_bidders',
               'solo_rate_pct','median_savings_pct','dominant_cpv','top_buyer']].to_string())


disp = summary[['buyer_country','spend_bn','notice_count','median_bidders',
                'solo_rate_pct','median_savings_pct','dominant_cpv','top_buyer']].copy()
disp.columns = ['Country','Spend (€B)','Notices','Med. Bidders','Solo %','Med. Savings %','Dominant CPV','Top Buyer']
disp['Spend (€B)']     = disp['Spend (€B)'].round(1)
disp['Med. Bidders']   = disp['Med. Bidders'].round(1)
disp['Solo %']         = disp['Solo %'].round(1)
disp['Med. Savings %'] = disp['Med. Savings %'].round(1)
disp['Top Buyer']      = disp['Top Buyer'].apply(lambda x: (x[:28]+'...') if isinstance(x,str) and len(x)>28 else x)

display(
    disp.style
    .background_gradient(subset=['Spend (€B)'],     cmap='Blues',    vmin=0)
    .background_gradient(subset=['Med. Bidders'],   cmap='RdYlGn',  vmin=1, vmax=5)
    .background_gradient(subset=['Solo %'],         cmap='RdYlGn_r',vmin=10, vmax=65)
    .background_gradient(subset=['Med. Savings %'], cmap='Greens',  vmin=0, vmax=100)
    .set_caption('Master Reference Table | Top 15 Countries by Spend (ex-HUN) | Competition incl. HUN | Jan 2026')
    .set_table_styles([{'selector':'caption','props':[('font-size','13px'),('font-weight','bold'),('caption-side','top')]}])
)


summary.to_csv('/content/nb4_summary_cross_reference.csv')
print('Exported: /content/nb4_summary_cross_reference.csv')



# ----------------------------------------------------------------------
# ## Analytical Conclusions
#
# **1. Category × Country combinations are the correct unit of analysis.**
# CPV 72 in Italy (median 4.0 bidders) and CPV 72 in CZE (median 1.0) are opposite markets despite being the same category. Country-level rankings from NB3 conceal this variation. Market strategy must be built at the category × country level.
#
# **2. CPV 72 IT Services: €25.6B in spend, 51.7% awarded with a single bidder.**
# The largest contracts are concentrated in CZE (€11.6B) and ROU (€6.1B) — both closed markets. The accessible IT markets (ITA, CHE, NOR, NLD, FRA) have smaller spend but genuine competition. Entry path: Large contracts (>€5M) in open markets, building toward the closed high-spend markets with a reference in hand.
#
# **3. Buyer archetype determines the required sales model.**
# High-Value Low-Volume buyers (Sinfra, Isavia, CFR) issue 4–8 mega-contracts per year — these require dedicated senior account management years in advance. High-Volume Low-Value buyers issue 10–25 smaller contracts — the right entry point for establishing a track record.
#
# **4. The savings map quantifies pricing headroom by country.**
# HRV (75.0% median savings), BGR (59.6%), SVK (95.1%), and CZE (75.0%) consistently award contracts well below their estimated values. Buyers in these markets expect significant price reduction from estimates. Western European markets (FRA, DEU, SWE, NOR) show near-zero median savings — efficient pricing where bids must win on capability, not price.
#
# **5. Large contracts (>€5M) hold 95.5% of spend but are not proportionally harder to win.**
# Solo rate at the Large tier (31.2%) is only 6.5 points below Micro contracts (37.7%). For a vendor with the scale to bid, the competitive barrier above €5M is smaller than the spend concentration implies. POL is the exception: 60.5% solo rate on large contracts — the most incumbent-locked large-contract market in the dataset.
#
# ----------------------------------------------------------------------
# *Data notes: financial metrics ex-HUN (framework ceiling distortions) and ex-CHE for savings (0% estimated coverage). Contract value capped at €5B/lot. All financial aggregates computed at notice level first to avoid lot-count inflation. tenders\_count null rate 29.5% at notice level — competition metrics are directional. CZE mean savings anomaly (-€2.76B) driven by a single outlier estimated value — always use median.*
#
