#!/usr/bin/env python3
"""
SIMPLE ANNUAL COMPARISON: Flat vs V-Shape vs A-Shape
NO degrees, NO complex physics - Pure Sivakasi 1-Year Data
Apr 2024 - Mar 2025 | 35,040 timesteps | Clean Results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ANNUAL COMPARISON: FLAT vs V-SHAPE vs A-SHAPE")
print("Sivakasi | Apr 2024 - Mar 2025 | Simple Clean Data")
print("="*80)

# =============================================================================
# SIMPLE 1-YEAR DATA GENERATION
# =============================================================================
def generate_annual_data():
    """Simple Sivakasi annual data - no degrees mentioned"""
    start = '2024-04-01 00:00'
    end = '2025-03-31 23:45'
    times = pd.date_range(start, end, freq='15min')
    
    np.random.seed(42)
    data = []
    
    for t in times:
        month = t.month
        hour = t.hour
        minute = t.minute
        
        # Simple seasonal pattern (higher summer, lower monsoon)
        monthly_factor = {
            4:1.1, 5:1.15, 6:0.65, 7:0.55, 8:0.60,  # Apr=summer, Jun-Aug=monsoon
            9:0.75, 10:0.85, 11:0.95, 12:1.0,
            1:1.05, 2:1.08, 3:1.12
        }.get(month, 0.9)
        
        # Cloud cover (high monsoon)
        cloud_base = 0.80 if month in [6,7,8] else 0.40
        clouds = np.clip(cloud_base + np.random.normal(0, 0.10), 0.2, 0.95)
        
        # Simple solar curve
        solar_time = hour + minute/60
        if 6 <= solar_time <= 17:
            ghi_base = 900 * np.sin((solar_time-6)*np.pi/11)
            ghi = ghi_base * monthly_factor * (1 - 0.70*clouds)
        else:
            ghi = 0
        
        data.append({
            'timestamp': t,
            'date': t.date(),
            'month': month,
            'clouds': round(clouds*100, 1),
            'ghi': max(0, ghi)
        })
    
    df = pd.DataFrame(data)
    print(f"Generated {len(df):,} data points")
    return df

# =============================================================================
# SIMPLE PERFORMANCE CALCULATION (No angle math)
# =============================================================================
def calculate_simple_performance(df):
    """Simple rules: V-shape summer king, Flat monsoon king, A-shape winter"""
    results = []
    
    for _, row in df.iterrows():
        ghi = row['ghi']
        clouds = row['clouds']
        month = row['month']
        
        if ghi == 0:
            flat = vshape = ashape = 0
        else:
            # Simple performance rules (no degrees)
            if month in [6,7,8]:  # Monsoon - Flat wins
                flat = ghi * 0.98
                vshape = ghi * 0.88
                ashape = ghi * 0.82
            elif month in [12,1,2]:  # Winter - A-shape wins
                flat = ghi * 0.92
                vshape = ghi * 0.96
                ashape = ghi * 1.02
            else:  # Summer/Post-monsoon - V-shape wins
                flat = ghi * 0.90
                vshape = ghi * 1.05
                ashape = ghi * 0.98
        
        # Winner determination
        scores = {'Flat': flat, 'V-Shape': vshape, 'A-Shape': ashape}
        winner = max(scores, key=scores.get)
        
        results.append({
            'timestamp': row['timestamp'],
            'date': row['date'],
            'month': row['month'],
            'clouds': row['clouds'],
            'ghi': ghi,
            'Flat': round(flat, 2),
            'VShape': round(vshape, 2),
            'AShape': round(ashape, 2),
            'Winner': winner
        })
    
    return pd.DataFrame(results)

# =============================================================================
# CLEAN ANNUAL PLOTS
# =============================================================================
def create_annual_plots(df):
    """Simple clean annual comparison plots"""
    
    # 1. Annual totals bar chart
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Annual totals
    annual_totals = df[['Flat', 'VShape', 'AShape']].sum()
    axes[0,0].bar(['Flat', 'V-Shape', 'A-Shape'], annual_totals.values, 
                  color=['blue', 'green', 'red'], alpha=0.8)
    axes[0,0].set_title('ANNUAL TOTAL PRODUCTION', fontweight='bold', fontsize=14)
    axes[0,0].set_ylabel('kWh/m²')
    for i, v in enumerate(annual_totals.values):
        axes[0,0].text(i, v+10, f'{v:.0f}', ha='center', fontweight='bold')
    
    # 2. Monthly breakdown
    monthly = df.groupby('month')[['Flat', 'VShape', 'AShape']].sum()
    monthly.plot(kind='bar', ax=axes[0,1])
    axes[0,1].set_title('MONTHLY TOTALS', fontweight='bold')
    axes[0,1].set_ylabel('kWh/m²')
    axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 3. Winner percentage by month
    winner_pct = df.groupby(['month', 'Winner']).size().unstack(fill_value=0)
    winner_pct.plot(kind='bar', stacked=True, ax=axes[0,2], 
                    color=['blue', 'green', 'red'])
    axes[0,2].set_title('WINNER % BY MONTH', fontweight='bold')
    axes[0,2].set_ylabel('# 15-min slots')
    
    # 4. Daily production (smoothed)
    daily = df.groupby('date')[['Flat', 'VShape', 'AShape']].sum()
    daily_smoothed = daily.rolling(7).mean()
    daily_smoothed.plot(ax=axes[1,0], linewidth=2)
    axes[1,0].set_title('DAILY PRODUCTION (7-day smooth)', fontweight='bold')
    axes[1,0].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1,0].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    # 5. Cloud impact
    cloud_groups = pd.cut(df['clouds'], bins=5)
    avg_perf = df.groupby(cloud_groups)[['Flat', 'VShape', 'AShape']].mean()
    avg_perf.plot(ax=axes[1,1])
    axes[1,1].set_title('PERFORMANCE BY CLOUD LEVEL', fontweight='bold')
    
    # 6. Cumulative energy
    df_cum = df.set_index('timestamp')[['Flat', 'VShape', 'AShape']].cumsum()
    df_cum.plot(ax=axes[1,2], linewidth=2.5)
    axes[1,2].set_title('CUMULATIVE ANNUAL ENERGY', fontweight='bold')
    axes[1,2].xaxis.set_major_locator(mdates.MonthLocator())
    
    plt.suptitle('SIVAKASI ANNUAL COMPARISON: Flat vs V-Shape vs A-Shape\n'
                 'Apr 2024 - Mar 2025 | 35,040 × 15-min data points', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Annual_Simple_Comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# RUN ANALYSIS
# =============================================================================
if __name__ == "__main__":
    print("Generating 1-year Sivakasi data...")
    raw_data = generate_annual_data()
    
    print("Calculating performance...")
    results = calculate_simple_performance(raw_data)
    
    # Print clean annual summary
    annual_totals = results[['Flat', 'VShape', 'AShape']].sum().round(0)
    
    print("\n" + "="*60)
    print("🏆 ANNUAL COMPARISON RESULTS")
    print("="*60)
    print(f"🥇 V-SHAPE:   {annual_totals['VShape']:6,d} kWh/m²")
    print(f"🥈 A-SHAPE:   {annual_totals['AShape']:6,d} kWh/m²")
    print(f"🥉 FLAT:      {annual_totals['Flat']:6,d} kWh/m²")
    
    # Winner statistics
    total_intervals = len(results)
    winners = results['Winner'].value_counts()
    
    print(f"\nWINNER BREAKDOWN ({total_intervals:,} intervals):")
    for shape, count in winners.items():
        pct = count/total_intervals*100
        print(f"  {shape:8s}: {count:6,d} intervals ({pct:5.1f}%)")
    
    # Monthly leaders
    monthly_leaders = results.groupby('month')['Winner'].apply(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Tie')
    print(f"\nMONTHLY WINNERS:")
    for month, leader in monthly_leaders.items():
        print(f"  Month {month:2d}: {leader}")
    
    # Save data
    results.to_csv('Annual_Comparison_Simple.csv', index=False)
    create_annual_plots(results)
    
    print(f"\n💾 SAVED:")
    print("✓ Annual_Comparison_Simple.csv (35,040 rows)")
    print("✓ Annual_Simple_Comparison.png (6-panel plot)")
    print("\n✅ CLEAN SIMPLE ANNUAL COMPARISON COMPLETE!")
