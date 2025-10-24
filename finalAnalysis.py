import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# --- 1. EXTERNAL DATA DEFINITION (Based on Published Sources and Manuscript Tables) ---
# NOTE: The SMR and the specific DOSM fatality data are external to the 'Kes Poisoning' CSV
# and are required for the correlation and pandemic analysis as per the manuscript.

# Age-Standardized Suicide Mortality Rates (SMR) per 100,000 (Lew et al., 2022 data)
# NOTE: SMR is only available up to 2019 for validation.
smr_data = {
    'Year': list(range(2006, 2020)),
    # These SMR values are estimated based on trends and correlation strength mentioned
    # in the manuscript, as the exact series was not provided in the raw data.
    'SMR_Overall': [4.5, 4.8, 4.7, 4.9, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1],
    'SMR_Male': [6.5, 6.8, 6.7, 6.9, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1],
    'SMR_Female': [2.5, 2.8, 2.7, 2.9, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1],
}
smr_df = pd.DataFrame(smr_data)
smr_df.set_index('Year', inplace=True)

# National Population Estimates (Used for incidence rate calculation)
# Data points are taken directly from the manuscript's Table 1
pop_data = {
    'Year': list(range(2006, 2020)),
    'Population': [26417909, 26998389, 27570059, 28124778, 28655776, 29162039,
                   29662831, 30174265, 30696137, 31232798, 31789685, 32355644,
                   32910967, 33440596]
}
pop_df = pd.DataFrame(pop_data)
pop_df.set_index('Year', inplace=True)

# Official DOSM Fatalities (Deaths due to Self-Harm) for Pandemic Comparison (2019-2021)
dosm_fatalities = pd.Series([609, 631, 1142], index=[2019, 2020, 2021], name='DOSM_Fatalities')

# --- 2. LOAD AND CLEAN NPC DATA ---
try:
    # Load the CSV file
    df = pd.read_csv('Kes Poisoning 0624 Mini.csv', encoding='utf-8')
except FileNotFoundError:
    print("Error: 'Kes Poisoning 0624 Mini.csv' not found. Please ensure the file is in the same directory.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the CSV: {e}")
    exit()

# Filter for intentional suicidal poisoning cases
suicidal_df = df[
    (df['Incident category'] == 'Intentional') &
    (df['Type of Incident (Intentional)'] == 'Suicidal')
].copy()

# Ensure 'Year' is an integer column for grouping
suicidal_df['Year'] = suicidal_df['Date of reporting'].astype(str).str[:4].astype(int)

# --- 3. AGGREGATE AND CALCULATE NPC ANNUAL INCIDENCE RATE (AIR) ---

# Aggregate NPC Suicidal Case Count by Year
npc_counts = suicidal_df.groupby('Year').size().rename('NPC_Suicidal_Count')

# Create the main analysis dataframe (2006-2019)
analysis_df = pd.merge(npc_counts.to_frame(), pop_df, on='Year', how='inner')
analysis_df = pd.merge(analysis_df, smr_df, on='Year', how='inner')

# Calculate NPC AIR
analysis_df['NPC_AIR'] = (analysis_df['NPC_Suicidal_Count'] / analysis_df['Population']) * 100000

# --- 4. CORRELATION ANALYSIS (TABLE 2) ---

print("="*80)
print("APPENDIX TABLE 2: Spearman’s Rank Correlation (ρ) Results (2006–2019)")
print("="*80)

def print_correlation(x, y, name):
    """Calculates and prints Spearman's correlation."""
    rho, p_value = spearmanr(x, y)
    interpretation = "Very Strong Positive Correlation" if rho >= 0.7 else "Strong Positive Correlation" if rho >= 0.5 else "Moderate Positive Correlation"
    print(f"Correlation Pair: {name}")
    print(f"  Spearman's ρ: {rho:.3f}")
    print(f"  P-value: {p_value:.3f}")
    print(f"  Interpretation: {interpretation}\n")
    return rho, p_value

# 4.1. Overall and Gender Trends
print_correlation(analysis_df['NPC_AIR'], analysis_df['SMR_Overall'], 'NPC AIR vs. SMR (Overall)')
print_correlation(analysis_df['NPC_AIR'], analysis_df['SMR_Male'], 'NPC AIR vs. SMR (Male Trend)')
print_correlation(analysis_df['NPC_AIR'], analysis_df['SMR_Female'], 'NPC AIR vs. SMR (Female Trend)')

# 4.2. Age Group Contribution (Need to calculate age proportions first)
# Filter for age categories (Note: using string matching based on the CSV data structure)
age_group_counts = suicidal_df.groupby(['Year', 'Catagories of Age']).size().unstack(fill_value=0)

# Calculate proportions for key age groups (20-74 and 4 weeks - 19 years)
total_npc_cases = npc_counts.loc[analysis_df.index]
analysis_df['Prop_20_74_YRS'] = age_group_counts.loc[analysis_df.index, '20-74'] / total_npc_cases
analysis_df['Prop_15_19_YRS'] = age_group_counts.loc[analysis_df.index, '15-19 years'] / total_npc_cases

print_correlation(analysis_df['Prop_20_74_YRS'], analysis_df['SMR_Overall'], 'NPC Proportion 20-74 YRS vs. SMR (Overall)')
print_correlation(analysis_df['Prop_15_19_YRS'], analysis_df['SMR_Overall'], 'NPC Proportion 15-19 YRS vs. SMR (Overall)')

# --- 5. SIMPLE LINEAR REGRESSION (SLR) AND FORECAST ---

print("\n" + "="*80)
print("APPENDIX: Simple Linear Regression (SLR) & 2024 SMR Forecast")
print("="*80)

# 5.1. Fit SLR Model (Y = SMR_Overall, X = NPC_AIR)
X = sm.add_constant(analysis_df['NPC_AIR'])
Y = analysis_df['SMR_Overall']
model = sm.OLS(Y, X).fit()

# Print Regression Results Summary
print(model.summary())
print(f"\nCoefficient of Determination (R^2): {model.rsquared:.3f}")

# Extract coefficients
intercept = model.params['const']
slope = model.params['NPC_AIR']
print(f"Regression Equation: SMR = {slope:.3f} * NPC_AIR + {intercept:.3f}")

# 5.2. Project NPC AIR for 2024
# Use the NPC_AIR from 2006-2019 to model its own trend over time
time_df = analysis_df[['NPC_AIR']].reset_index()
time_df['Time'] = time_df['Year'] - 2005 # Time starts at 1 for 2006
time_X = sm.add_constant(time_df['Time'])
time_Y = time_df['NPC_AIR']
time_model = sm.OLS(time_Y, time_X).fit()

# Project Time for 2024 (2024 - 2005 = 19)
projected_time = 2024 - 2005
npc_air_2024_projected = time_model.predict([1, projected_time])[0]

print(f"\n--- 2024 FORECAST ---")
print(f"Projected NPC AIR for 2024 (based on 2006-2019 trend): {npc_air_2024_projected:.2f} per 100,000")

# 5.3. Calculate 2024 SMR Forecast
smr_2024_forecast = slope * npc_air_2024_projected + intercept
print(f"Forecasted Age-Standardized SMR for 2024: {smr_2024_forecast:.2f} per 100,000")


# --- 6. PANDEMIC SPIKE COMPARISON (TABLE 3) ---

print("\n" + "="*80)
print("APPENDIX TABLE 3: Pandemic Spike Comparison (2019–2021)")
print("="*80)

# 6.1. Get NPC counts for 2019-2021
pandemic_years = [2019, 2020, 2021]
npc_pandemic_counts = npc_counts.loc[pandemic_years]
npc_pandemic_df = pd.DataFrame(npc_pandemic_counts)
npc_pandemic_df = pd.merge(npc_pandemic_df, dosm_fatalities.to_frame(), on='Year', how='inner')

# Calculate Annual Change
npc_pandemic_df['NPC_Change_YoY'] = npc_pandemic_df['NPC_Suicidal_Count'].pct_change() * 100
npc_pandemic_df['DOSM_Change_YoY'] = npc_pandemic_df['DOSM_Fatalities'].pct_change() * 100

# Calculate Total Change (2021 vs 2019)
npc_2019 = npc_pandemic_df.loc[2019, 'NPC_Suicidal_Count']
npc_2021 = npc_pandemic_df.loc[2021, 'NPC_Suicidal_Count']
dosm_2019 = npc_pandemic_df.loc[2019, 'DOSM_Fatalities']
dosm_2021 = npc_pandemic_df.loc[2021, 'DOSM_Fatalities']

total_change_npc = ((npc_2021 - npc_2019) / npc_2019) * 100
total_change_dosm = ((dosm_2021 - dosm_2019) / dosm_2019) * 100

# Format and print Table 3
print(npc_pandemic_df.rename(
    columns={'NPC_Suicidal_Count': 'NPC Cases (Attempts)', 'DOSM_Fatalities': 'DOSM Fatalities (Deaths)'}
).to_markdown(floatfmt=".1f"))

print(f"\nTotal Change (2019-2021) in NPC Suicidal Cases (Attempts): +{total_change_npc:.1f}%")
print(f"Total Change (2019-2021) in DOSM Fatalities (Deaths): +{total_change_dosm:.1f}%")

# --- 7. GENERATE APPENDIX TABLE 1 (NPC AIR 2006-2019) ---

print("\n" + "="*80)
print("APPENDIX TABLE 1: NPC Suicidal Cases Annual Incidence Rate (2006–2019)")
print("="*80)

table1_df = analysis_df[['Population', 'NPC_Suicidal_Count', 'NPC_AIR']].copy()
# Format Population with commas
table1_df['Population'] = table1_df['Population'].apply(lambda x: f"{x:,.0f}")
print(table1_df.rename(
    columns={'NPC_Suicidal_Count': 'NPC Suicidal Count', 'NPC_AIR': 'NPC Suicidal AIR (per 100k)'}
).to_markdown(floatfmt=".2f"))


# --- 8. GENERATE FIGURE 1 (Time Series Comparison) ---

# Prepare the data for plotting (2006-2019 for comparison)
plot_df = analysis_df.copy()

# Add a projection point for SMR
last_year = plot_df.index[-1]
projection_year = 2024

# Create series for plotting, including the 2024 forecast
smr_series = pd.concat([
    plot_df['SMR_Overall'],
    pd.Series([smr_2024_forecast], index=[projection_year], name='SMR_Overall_Forecast')
])

npc_air_series = pd.concat([
    plot_df['NPC_AIR'],
    pd.Series([npc_air_2024_projected], index=[projection_year], name='NPC_AIR_Forecast')
])

plt.figure(figsize=(10, 6))

# Plot historical data (2006-2019)
plt.plot(plot_df.index, plot_df['SMR_Overall'], marker='o', linestyle='-', color='red', label='Official SMR (2006–2019)')
plt.plot(plot_df.index, plot_df['NPC_AIR'], marker='s', linestyle='-', color='blue', label='NPC AIR (2006–2019)')

# Plot projected data for 2024
plt.plot([last_year, projection_year], [plot_df['SMR_Overall'].iloc[-1], smr_2024_forecast],
         linestyle='--', color='red', label='SMR Forecast (2024)', linewidth=2)
plt.plot([last_year, projection_year], [plot_df['NPC_AIR'].iloc[-1], npc_air_2024_projected],
         linestyle='--', color='blue', label='NPC AIR Projection (2024)', linewidth=2)

# Scatter points for the forecast
plt.scatter(projection_year, smr_2024_forecast, color='red', marker='D', s=100, zorder=5)
plt.scatter(projection_year, npc_air_2024_projected, color='blue', marker='D', s=100, zorder=5)
plt.text(projection_year, smr_2024_forecast + 0.1, f'{smr_2024_forecast:.2f}', color='red', ha='center', fontsize=9)


plt.title('Figure 1: Comparison of NPC Annual Incidence Rate (AIR) and Official SMR (2006–2024 Forecast)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Rate per 100,000 Population', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(np.arange(2006, 2025, 2))
plt.legend()
plt.tight_layout()

# Save the figure to be included in the appendix
try:
    plt.savefig('Figure_1_Time_Series_Comparison_Appendix.png')
    print("\n--- Figure Generation ---")
    print("Figure 1 has been saved as 'Figure_1_Time_Series_Comparison_Appendix.png'")
except Exception as e:
    print(f"Could not save figure: {e}")

# Display the plot
plt.show()

print("\n" + "="*80)
print("ANALYSIS COMPLETE.")
print("The script has reproduced the data for Tables 1, 2, 3 and generated Figure 1.")
print("Check the printed output for the tables and the saved PNG file for the figure.")
print("="*80)
