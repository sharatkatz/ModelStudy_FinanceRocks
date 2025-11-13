import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- Data Extracted from Uploaded Tables (image_66aed6.png and image_66aeba.png) ---

# --- 1. Market Share by Headcount Class ---
# Data from image_66aed6.png (Row 'All')
hc_data = {
    'Class': ['1', '2-4', '5-9', '10-19', '20-49', '50-99', '100-249', '250-499', '500-999', '+1000', 'Unknown'],
    'Count': [32523, 126119, 13570, 7957, 5322, 1822, 880, 338, 157, 88, 13001]
}
hc_df = pd.DataFrame(hc_data).set_index('Class')
# Reorder to show SME dominance clearly
hc_df = hc_df.reindex(['2-4', '1', '5-9', '10-19', '20-49', '50-99', '100-249', '250-499', '500-999', '+1000', 'Unknown'])


# --- 2. Market Share by Revenue Class ---
# Data from image_66aeba.png (Row 'All')
rev_data = {
    'Class': ['0-0.2M', '0.2-0.4M', '0.4-1M', '1-2M', '2-10M', '10-20M', '20-50M', '50-100M', '+100M', 'Unknown'],
    'Count': [118480, 21477, 21812, 10922, 12136, 2016, 1430, 525, 490, 12489]
}
rev_df = pd.DataFrame(rev_data).set_index('Class')
# Reorder based on size
rev_df = rev_df.reindex(['0-0.2M', '0.2-0.4M', '0.4-1M', '1-2M', '2-10M', '10-20M', '20-50M', '50-100M', '+100M', 'Unknown'])


# --- 3. Top 10 Industries by Total Company Count ---
# Data from image_66aeba.png (Column 'ALL')
industry_data = {
    'Industry': ['Pro-Sci-Tech', 'Construction', 'Wholesale-Retail-Motors', 'Real estate activities', 
                 'Manufacturing', 'Information-Comms', 'Admin-Support', 'Transport-Storage', 
                 'Finance-Insurance', 'Accom-Catering', 'Health-Social'],
    'Count': [34067, 27995, 27409, 17352, 13813, 12919, 11116, 9717, 9345, 8881, 8158] # Taking 11 to ensure top 10 is clear
}
industry_df = pd.DataFrame(industry_data).set_index('Industry').sort_values(by='Count', ascending=True).tail(10)


# --- 4. Enterprise Companies (>€20M Revenue) by Industry ---
# Summing 20-50M, 50-100M, +100M columns from image_66aeba.png (The '>=20M' column is also provided, but summing ensures accuracy)
enterprise_rows = ['Wholesale-Retail-Motors', 'Manufacturing', 'Information-Comms', 'Pro-Sci-Tech', 'Transport-Storage', 
                   'Electricity-Gas', 'Construction', 'Finance-Insurance', 'Real estate activities', 'Admin-Support']
# Summing 20-50M, 50-100M, and +100M columns for each industry from the table:
enterprise_data_calculated = {
    'Wholesale-Retail-Motors': 421 + 135 + 125, # 681
    'Manufacturing': 307 + 131 + 153, # 591
    'Information-Comms': 89 + 37 + 30, # 156
    'Pro-Sci-Tech': 78 + 29 + 18, # 125
    'Transport-Storage': 91 + 27 + 21, # 139
    'Electricity-Gas': 50 + 28 + 38, # 116
    'Construction': 144 + 41 + 35, # 220 (Error in my manual calc vs. plot, sticking to sum: 144+41+35 = 220)
    'Finance-Insurance': 51 + 14 + 10, # 75
    'Real estate activities': 43 + 17 + 9, # 69
    'Admin-Support': 47 + 25 + 23 # 95
}
enterprise_df = pd.DataFrame(list(enterprise_data_calculated.items()), columns=['Industry', 'Count']).set_index('Industry').sort_values(by='Count', ascending=True)


# --- Plotting ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
plt.style.use('seaborn-v0_8-darkgrid')

# 1. Headcount Market Share (Top Left)
axes[0, 0].bar(hc_df.index, hc_df['Count'], color='skyblue')
axes[0, 0].set_title('1. Finnish Market: Share by Headcount Class', fontsize=14)
axes[0, 0].set_ylabel('Number of Companies', fontsize=12)
axes[0, 0].tick_params(axis='x', rotation=45)
# Highlight SME dominance
axes[0, 0].text(0, 130000, f"Dominant SME Class (2-4 employees)\n({hc_df.loc['2-4', 'Count']:,} Companies)", 
                ha='left', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))


# 2. Revenue Market Share (Top Right)
axes[0, 1].bar(rev_df.index, rev_df['Count'], color='coral')
axes[0, 1].set_title('2. Finnish Market: Share by Revenue Class', fontsize=14)
axes[0, 1].set_ylabel('Number of Companies', fontsize=12)
axes[0, 1].tick_params(axis='x', rotation=45)
# Highlight micro-revenue dominance
axes[0, 1].text(0, 120000, f"Micro-SME Class (0-0.2M)\n({rev_df.loc['0-0.2M', 'Count']:,} Companies)", 
                ha='left', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))


# 3. Top 10 Industries (Total Count) (Bottom Left)
axes[1, 0].barh(industry_df.index, industry_df['Count'], color='green')
axes[1, 0].set_title('3. Top 10 Industries by Total Company Count', fontsize=14)
axes[1, 0].set_xlabel('Number of Companies', fontsize=12)
axes[1, 0].tick_params(axis='y', labelsize=10)


# 4. Enterprise Companies (>€20M Revenue) (Bottom Right)
axes[1, 1].barh(enterprise_df.index, enterprise_df['Count'], color='darkviolet')
axes[1, 1].set_title('4. Top Industries by Enterprise Company Count (Revenue > €20M)', fontsize=14)
axes[1, 1].set_xlabel('Number of Enterprise Companies', fontsize=12)
axes[1, 1].tick_params(axis='y', labelsize=10)
# Highlight key targets (Wholesale-Retail-Motors and Manufacturing)
# axes[1, 1].text(600, 9.5, "High-Value Targets", 
#                 ha='right', va='center', fontsize=10, color='red', weight='bold')


plt.tight_layout()
plt.show()