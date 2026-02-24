import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

df = pd.read_csv('project_data.csv')

BLUE = "#2E75B6"
ORANGE = "#E8703A"
RED = "#C0392B"
GREEN = "#27AE60"
LIGHT = "#EBF3FB"


# CHART 1: Class Distribution Bar Chart
# shows the imbalance clearly

labeled = df['Class'].dropna()
counts = labeled.value_counts()

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(['No Disability', 'Has Disability'], counts.values,
              color=[BLUE, ORANGE], width=0.5, edgecolor='white')

for bar, count in zip(bars, counts.values):
    pct = round(count / len(labeled) * 100, 1)
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
            f'{count:,}\n({pct}%)', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_title('Class Distribution: Disability Status', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Number of Households', fontsize=12)
ax.set_ylim(0, max(counts.values) * 1.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig('chart1_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 1 saved: class distribution")

# CHART 2: Missing Values by Topic
# shows which topic areas have the most missing data

topic_vars = {
    'Structural\n(exterior)': ['LOTSIZE', 'FNDCRUMB', 'ROOFHOLE', 'ROOFSHIN', 'ROOFSAG',
                                'WALLSIDE', 'WALLSLOPE', 'WINBROKE', 'WINBARS', 'WINBOARD', 'FOUNDTYPE'],
    'Neighborhood': ['NHQSCHOOL', 'NHQPUBTRN', 'NHQPCRIME', 'NHQSCRIME', 'NHQRISK',
                     'RATINGNH', 'RATINGHS', 'NEARABAND', 'NEARTRASH', 'NEARBARCL'],
    'Demographics\n(SOGI)': ['HHPRNTHOME', 'HHSOGISO', 'HHSOGIG', 'HHSOGILGBT', 'SOGIRESP'],
    'Disability\nIndicators': ['HHMEMRY', 'HHERRND', 'HHCARE', 'HHSEE'],
    'Healthy Homes': ['HHLDASTHMAER', 'NUMASTHMAST', 'HHLDASTHMA'],
    'Housing\nProblems': ['MOLDLROOM', 'NOWAT', 'MOLDKITCH', 'MOLDBATH',
                          'MOLDBEDRM', 'MOLDBASEM', 'MOLDOTHER', 'HOT', 'COLD',
                          'SMOKALRM', 'SEWBREAK', 'FUSEBLOW', 'NOTOIL'],
    'Demographics\n(household)': ['HHSPAN', 'HHMAR', 'HHSEX', 'HHRACE', 'HHGRAD',
                                   'NUMELDERS', 'NUMADULTS', 'NUMPEOPLE', 'NUMVETS'],
}

topic_missing = {}
for topic, cols in topic_vars.items():
    existing = [c for c in cols if c in df.columns]
    avg_missing = df[existing].isnull().mean().mean() * 100
    topic_missing[topic] = round(avg_missing, 1)

topics = list(topic_missing.keys())
pcts = list(topic_missing.values())
colors = [ORANGE if p > 20 else BLUE for p in pcts]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(topics, pcts, color=colors, edgecolor='white', height=0.6)

for bar, pct in zip(bars, pcts):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{pct}%', va='center', fontsize=10, fontweight='bold')

ax.set_title('Average Missing Values (%) by Variable Group', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Average Missing %', fontsize=11)
ax.set_xlim(0, max(pcts) * 1.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=10)

high = mpatches.Patch(color=ORANGE, label='High missing (>20%) - structural/skip logic')
low  = mpatches.Patch(color=BLUE,   label='Lower missing - survey answered')
ax.legend(handles=[high, low], fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig('chart2_missing_by_topic.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 2 saved: missing values by topic")

# CHART 3: NUMELDERS distribution by Class
# shows elderly members strongly linked to disability

df_labeled = df.dropna(subset=['Class'])
df_labeled = df_labeled.dropna(subset=['NUMELDERS'])

no_dis  = df_labeled[df_labeled['Class'] == 'No']['NUMELDERS'].value_counts().sort_index()
yes_dis = df_labeled[df_labeled['Class'] == 'Yes']['NUMELDERS'].value_counts().sort_index()

# convert to percentages within each class
no_pct  = (no_dis  / no_dis.sum()  * 100).round(1)
yes_pct = (yes_dis / yes_dis.sum() * 100).round(1)

x = np.array([0, 1, 2, 3, 4])
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, [no_pct.get(i, 0)  for i in x], width, label='No Disability',  color=BLUE,   edgecolor='white')
ax.bar(x + width/2, [yes_pct.get(i, 0) for i in x], width, label='Has Disability', color=ORANGE, edgecolor='white')

ax.set_title('Number of Elderly Members (65+) by Disability Status', fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Number of Elderly Members in Household', fontsize=11)
ax.set_ylabel('% of Households (within each class)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(['0 elderly', '1 elderly', '2 elderly', '3 elderly', '4 elderly'])
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig('chart3_numelders_by_class.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 3 saved: NUMELDERS by class")

# CHART 4: Income boxplot by Class
# shows disability households have lower income

df_inc = df_labeled.dropna(subset=['HINCP'])
df_inc = df_inc[df_inc['HINCP'] < df_inc['HINCP'].quantile(0.95)]  # remove top 5% for readability

fig, ax = plt.subplots(figsize=(7, 5))
data_no  = df_inc[df_inc['Class'] == 'No']['HINCP']
data_yes = df_inc[df_inc['Class'] == 'Yes']['HINCP']

bp = ax.boxplot([data_no, data_yes], patch_artist=True,
                medianprops=dict(color='white', linewidth=2.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                flierprops=dict(marker='o', markersize=3, alpha=0.3))

bp['boxes'][0].set_facecolor(BLUE)
bp['boxes'][1].set_facecolor(ORANGE)

ax.set_xticklabels(['No Disability', 'Has Disability'], fontsize=12)
ax.set_title('Household Income Distribution by Disability Status', fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel('Annual Household Income (USD)', fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=10)

med_no  = int(data_no.median())
med_yes = int(data_yes.median())
ax.text(1, med_no  + 2000, f'Median: ${med_no:,}',  ha='center', fontsize=9, color='white', fontweight='bold')
ax.text(2, med_yes + 2000, f'Median: ${med_yes:,}', ha='center', fontsize=9, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('chart4_income_by_class.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 4 saved: income by class")

print()
print("All 4 charts saved. Add them to your report folder.")


# CHART 5: Missing Value Heatmap - Vacant vs Normal rows
# shows the 1038 pattern visually

# get the 40 columns that have exactly 1038 missing
missing_counts = df.isnull().sum()
cols_1038 = missing_counts[missing_counts == 1038].index.tolist()
first_col = cols_1038[0]

# take 80 vacant rows and 80 normal rows for visualization
vacant_rows = df[df[first_col].isnull()].head(80)
normal_rows = df[df[first_col].notna()].head(80)

# pick columns to show: the 1038 pattern cols + 10 other cols
other_cols = [c for c in df.columns if c not in cols_1038][:10]
show_cols = cols_1038[:20] + other_cols

combined = pd.concat([vacant_rows[show_cols], normal_rows[show_cols]])
missing_matrix = combined.isnull().astype(int)

fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(missing_matrix, ax=ax, cmap=['#2E75B6', '#E8703A'],
            cbar=False, xticklabels=True, yticklabels=False,
            linewidths=0, linecolor='none')

ax.axhline(y=80, color='white', linewidth=3)
ax.set_title('Missing Value Pattern: Vacant Units (top) vs Normal Households (bottom)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel('Variables (orange = missing, blue = present)', fontsize=11)
ax.tick_params(axis='x', labelsize=7, rotation=45)

ax.text(-0.5, 40, 'Vacant\nUnits\n(1,038)', va='center', ha='right',
        fontsize=10, fontweight='bold', color=ORANGE,
        transform=ax.get_yaxis_transform())
ax.text(-0.5, 120, 'Normal\nHouseholds', va='center', ha='right',
        fontsize=10, fontweight='bold', color=BLUE,
        transform=ax.get_yaxis_transform())

orange_patch = mpatches.Patch(color=ORANGE, label='Missing value')
blue_patch   = mpatches.Patch(color=BLUE,   label='Value present')
ax.legend(handles=[orange_patch, blue_patch], loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('chart5_missing_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 5 saved: missing value heatmap")

# CHART 6: Disability RATE by number of elderly members
#

df_labeled = df.dropna(subset=['Class', 'NUMELDERS'])

disability_rate = []
counts_total = []
for n in range(5):
    subset = df_labeled[df_labeled['NUMELDERS'] == n]
    if len(subset) > 0:
        rate = (subset['Class'] == 'Yes').mean() * 100
        disability_rate.append(round(rate, 1))
        counts_total.append(len(subset))
    else:
        disability_rate.append(0)
        counts_total.append(0)

x = range(5)
colors = [BLUE if r < 30 else ORANGE if r < 50 else RED for r in disability_rate]

fig, ax1 = plt.subplots(figsize=(9, 5))

bars = ax1.bar(x, disability_rate, color=colors, width=0.5, edgecolor='white', zorder=3)

for i, (bar, rate, count) in enumerate(zip(bars, disability_rate, counts_total)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
             f'{rate}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
             f'n={count:,}', ha='center', va='center', fontsize=9,
             color='white', fontweight='bold')

ax1.plot(list(x), disability_rate, color='black', marker='o',
         linewidth=2, markersize=6, zorder=4, linestyle='--', alpha=0.6)

ax1.set_title('Disability Rate Increases Sharply with Number of Elderly Members',
              fontsize=13, fontweight='bold', pad=15)
ax1.set_xlabel('Number of Persons Aged 65+ in Household', fontsize=11)
ax1.set_ylabel('% of Households with Disability (Yes)', fontsize=11)
ax1.set_xticks(list(x))
ax1.set_xticklabels([f'{i} elderly\nmember{"s" if i!=1 else ""}' for i in x], fontsize=10)
ax1.set_ylim(0, max(disability_rate) * 1.2)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='y', alpha=0.3, zorder=0)

plt.tight_layout()
plt.savefig('chart6_disability_rate_by_elderly.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 6 saved: disability rate by elderly members")


print()
print("All 4 advanced charts saved successfully.")
print("Files: chart5_missing_heatmap.png, chart6_disability_rate_by_elderly.png,")
print("       chart7_variable_types.png, chart8_imbalance_danger.png")


df_labeled = df.dropna(subset=['Class'])



df_labeled = df.dropna(subset=['Class'])

# CHART 7: Descriptive stats table for key numeric vars
# + skewness shown visually for HINCP


# key variables to show - most relevant to disability
key_vars = {
    'HINCP':      'Household Income ($)',
    'NUMELDERS':  'Elderly Members (65+)',
    'NUMPEOPLE':  'Total Household Size',
    'NUMADULTS':  'Number of Adults',
    'NUMVETS':    'Number of Veterans',
    'PERPOVLVL':  'Poverty Level (%)',
    'TOTHCAMT':   'Total Housing Cost ($/yr)',
}

rows = []
for col, label in key_vars.items():
    if col in df_labeled.columns:
        data = df_labeled[col].dropna()
        mean   = data.mean()
        median = data.median()
        std    = data.std()
        mn     = data.min()
        mx     = data.max()
        skew   = data.skew()
        skew_label = 'right skewed' if skew > 1 else 'left skewed' if skew < -1 else 'approx normal'
        rows.append([label, f'{mean:,.1f}', f'{median:,.1f}', f'{std:,.1f}',
                     f'{mn:,.0f}', f'{mx:,.0f}', f'{skew:.2f}', skew_label])

# print table
print("Descriptive Statistics Table:")
print(f"{'Variable':<30} {'Mean':>12} {'Median':>12} {'Std Dev':>12} {'Min':>10} {'Max':>12} {'Skew':>8} {'Shape'}")
print("-" * 110)
for r in rows:
    print(f"{r[0]:<30} {r[1]:>12} {r[2]:>12} {r[3]:>12} {r[4]:>10} {r[5]:>12} {r[6]:>8}  {r[7]}")

# main figure: 2 rows
# top: descriptive stats as visual table
# bottom: HINCP histogram showing skew + mean vs median

fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('white')

# --- top half: stats table as image ---
ax_table = fig.add_axes([0.02, 0.52, 0.96, 0.44])
ax_table.axis('off')

col_labels = ['Variable', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness', 'Distribution Shape']
col_widths = [0.22, 0.09, 0.09, 0.09, 0.09, 0.11, 0.09, 0.16]

# header
x = 0.01
for label, w in zip(col_labels, col_widths):
    ax_table.add_patch(plt.Rectangle((x, 0.85), w - 0.005, 0.12,
                                      facecolor=BLUE, transform=ax_table.transAxes))
    ax_table.text(x + (w - 0.005)/2, 0.91, label, transform=ax_table.transAxes,
                  ha='center', va='center', fontsize=9.5, fontweight='bold', color='white')
    x += w

# rows
for i, row in enumerate(rows):
    bg = '#EBF3FB' if i % 2 == 0 else 'white'
    y_pos = 0.85 - (i + 1) * 0.115
    x = 0.01
    for val, w in zip(row, col_widths):
        ax_table.add_patch(plt.Rectangle((x, y_pos), w - 0.005, 0.105,
                                          facecolor=bg, transform=ax_table.transAxes))
        # color skew label
        color = RED if 'right' in str(val) else '#27AE60' if 'left' in str(val) else '#555555'
        fw = 'bold' if 'skewed' in str(val) else 'normal'
        ax_table.text(x + (w - 0.005)/2, y_pos + 0.052, str(val),
                      transform=ax_table.transAxes,
                      ha='center', va='center', fontsize=9, color=color, fontweight=fw)
        x += w

ax_table.set_title('Descriptive Statistics: Key Numeric Variables',
                   fontsize=13, fontweight='bold', pad=10, loc='left')

#  bottom half: HINCP histogram
ax_hist = fig.add_axes([0.06, 0.06, 0.55, 0.38])

hincp = df_labeled['HINCP'].dropna()
hincp_trim = hincp[hincp < hincp.quantile(0.97)]  # trim top 3% for readability

mean_val   = hincp.mean()
median_val = hincp.median()

n, bins, patches = ax_hist.hist(hincp_trim, bins=50, color=BLUE,
                                 edgecolor='white', linewidth=0.5, alpha=0.85)

# color bars differently past the mean to show right tail
for patch, left in zip(patches, bins[:-1]):
    if left > mean_val:
        patch.set_facecolor(ORANGE)
        patch.set_alpha(0.7)

ax_hist.axvline(mean_val,   color=RED,   linewidth=2.5, linestyle='--', label=f'Mean: ${mean_val:,.0f}')
ax_hist.axvline(median_val, color=GREEN, linewidth=2.5, linestyle='-',  label=f'Median: ${median_val:,.0f}')

ax_hist.set_title('HINCP (Household Income): Right Skewed Distribution\nThis is why we used Median imputation, not Mean',
                  fontsize=11, fontweight='bold', pad=10)
ax_hist.set_xlabel('Annual Household Income (USD)', fontsize=10)
ax_hist.set_ylabel('Number of Households', fontsize=10)
ax_hist.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}k'))
ax_hist.spines['top'].set_visible(False)
ax_hist.spines['right'].set_visible(False)
ax_hist.legend(fontsize=10)

# annotation arrow showing the skew gap
ax_hist.annotate('Mean is pulled right\nby high income outliers\n- not a fair replacement\nfor missing values',
                 xy=(mean_val, max(n) * 0.6),
                 xytext=(mean_val + 60000, max(n) * 0.75),
                 fontsize=8.5, color=RED,
                 arrowprops=dict(arrowstyle='->', color=RED, lw=1.5),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDECEA', edgecolor=RED, alpha=0.8))

ax_hist.annotate('Median stays here\n- better estimate of\ntypical household income',
                 xy=(median_val, max(n) * 0.5),
                 xytext=(median_val - 120000, max(n) * 0.65),
                 fontsize=8.5, color=GREEN,
                 arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#EAFAF1', edgecolor=GREEN, alpha=0.8))

# --- bottom right: skewness summary box ---
ax_box = fig.add_axes([0.65, 0.06, 0.32, 0.38])
ax_box.axis('off')

skew_summary = [
    ('HINCP', hincp.skew(), 'Income'),
    ('TOTHCAMT', df_labeled['TOTHCAMT'].dropna().skew(), 'Housing Cost'),
    ('PERPOVLVL', df_labeled['PERPOVLVL'].dropna().skew(), 'Poverty Level'),
    ('NUMELDERS', df_labeled['NUMELDERS'].dropna().skew(), 'Elderly Count'),
    ('NUMVETS', df_labeled['NUMVETS'].dropna().skew(), 'Veteran Count'),
    ('NUMPEOPLE', df_labeled['NUMPEOPLE'].dropna().skew(), 'Household Size'),
]

ax_box.text(0.5, 0.97, 'Skewness Summary', transform=ax_box.transAxes,
            ha='center', va='top', fontsize=11, fontweight='bold', color='#1F3864')
ax_box.text(0.5, 0.89, 'Why median was chosen for all numeric imputation',
            transform=ax_box.transAxes, ha='center', va='top', fontsize=8.5,
            color='gray', style='italic')

y = 0.78
for col, skew_val, label in skew_summary:
    color = RED if skew_val > 1 else GREEN if skew_val < -1 else '#555555'
    shape = 'Right skewed' if skew_val > 1 else 'Left skewed' if skew_val < -1 else 'Near normal'
    bar_width = min(abs(skew_val) / 8, 0.45)

    ax_box.add_patch(plt.Rectangle((0.5, y - 0.025), bar_width * (1 if skew_val > 0 else -1),
                                    0.055, facecolor=color, alpha=0.3,
                                    transform=ax_box.transAxes))
    ax_box.text(0.03, y, label, transform=ax_box.transAxes,
                va='center', fontsize=9, color='#333333')
    ax_box.text(0.97, y, f'{skew_val:.2f} - {shape}', transform=ax_box.transAxes,
                va='center', ha='right', fontsize=8.5, color=color, fontweight='bold')
    ax_box.axhline(y=y, xmin=0.48, xmax=0.52, color='#CCCCCC', linewidth=0.5)
    y -= 0.13

plt.savefig('chart7_stats_skewness.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print("Chart 7 saved: descriptive stats + skewness")