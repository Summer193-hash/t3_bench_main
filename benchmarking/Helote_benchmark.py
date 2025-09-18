import pandas as pd
import numpy as np
import os
import sys
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance

# --- Add project root to Python path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# -------------------------------------

# ==========================================================
# 1. Configuration
# ==========================================================
# Paths are now relative to the project root for portability
REAL_DATA_DIR = os.path.join(project_root, 'data', '03_processed_csVs', 'ground_truth_output')
GENERATED_DATA_DIR = os.path.join(project_root, 'data', '03_processed_csVs', 'vision_output')
OUTPUT_DIR = os.path.join(project_root, 'benchmarking', 'reports')

# ==========================================================
# 2. Analysis Core Functions
# ==========================================================

def analyze_csv_file(filepath):
    """
    Analyzes a single gameplay CSV file and extracts key metrics.
    """
    try:
        df = pd.read_csv(filepath)
        metrics = {}

        # --- Health Monotonicity Analysis ---
        if 'P1_Health' in df.columns and 'P2_Health' in df.columns and len(df) > 1:
            p1_health_diff = df['P1_Health'].diff().dropna()
            p2_health_diff = df['P2_Health'].diff().dropna()
            p1_mono_score = (p1_health_diff <= 0).sum() / len(p1_health_diff) if len(p1_health_diff) > 0 else 1.0
            p2_mono_score = (p2_health_diff <= 0).sum() / len(p2_health_diff) if len(p2_health_diff) > 0 else 1.0
            metrics['health_monotonicity'] = (p1_mono_score + p2_mono_score) / 2
            p1_damage_events = abs(p1_health_diff[p1_health_diff < 0])
            p2_damage_events = abs(p2_health_diff[p2_health_diff < 0])
            metrics['damage_events'] = pd.concat([p1_damage_events, p2_damage_events])
        else:
            metrics['health_monotonicity'] = np.nan
            metrics['damage_events'] = pd.Series(dtype='float64')

        # --- Timer Analysis ---
        timer_col = 'Normalized_Game_Timer'
        if timer_col in df.columns and len(df) > 1:
            timer_diff = df[timer_col].diff().dropna()
            metrics['timer_diffs'] = timer_diff
            if len(timer_diff) > 0:
                metrics['timer_monotonicity'] = (timer_diff <= 0).sum() / len(timer_diff)
                tick_frames = (timer_diff < 0).sum()
                metrics['tick_frequency'] = tick_frames / len(df) if len(df) > 0 else 0.0
            else:
                metrics['timer_monotonicity'] = 1.0
                metrics['tick_frequency'] = 0.0
        else:
            metrics['timer_monotonicity'] = np.nan
            metrics['tick_frequency'] = np.nan
            metrics['timer_diffs'] = pd.Series(dtype='float64')

        return metrics
    except Exception as e:
        print(f"  Warning: Could not analyze file '{os.path.basename(filepath)}'. Reason: {e}")
        return None

def aggregate_results(data_directory):
    """
    Finds all CSVs in a directory, runs analysis on each, and aggregates the results.
    """
    all_files = glob.glob(os.path.join(data_directory, '*.csv'))
    if not all_files: return None

    print(f"  Analyzing {len(all_files)} files in '{os.path.basename(data_directory)}'...")
    results_list = [analyze_csv_file(f) for f in all_files]
    results_list = [r for r in results_list if r is not None]
    
    if not results_list: return None

    aggregated = {
        'health_monotonicity_mean': np.nanmean([r['health_monotonicity'] for r in results_list]),
        'health_monotonicity_std': np.nanstd([r['health_monotonicity'] for r in results_list]),
        'damage_events': pd.concat([r['damage_events'] for r in results_list]).tolist(),
        'timer_monotonicity_mean': np.nanmean([r['timer_monotonicity'] for r in results_list]),
        'timer_monotonicity_std': np.nanstd([r['timer_monotonicity'] for r in results_list]),
        'tick_frequency_mean': np.nanmean([r['tick_frequency'] for r in results_list]),
        'tick_frequency_std': np.nanstd([r['tick_frequency'] for r in results_list]),
        'timer_diffs': pd.concat([r['timer_diffs'] for r in results_list]).tolist(),
    }
    return aggregated

def calculate_timer_accuracy(real_dir, generated_dir):
    """
    Calculates the Mean Absolute Error (MAE) for the timer value.
    """
    print("  Calculating timer accuracy (MAE) across matched CSV pairs...")
    real_files = glob.glob(os.path.join(real_dir, '*.csv'))
    all_maes = []

    for real_filepath in real_files:
        basename = os.path.basename(real_filepath)
        generated_filepath = os.path.join(generated_dir, basename)

        if os.path.exists(generated_filepath):
            try:
                df_real = pd.read_csv(real_filepath)
                df_gen = pd.read_csv(generated_filepath)
                if 'Normalized_Game_Timer' not in df_real.columns or 'Normalized_Game_Timer' not in df_gen.columns: continue
                merged_df = pd.merge(df_real, df_gen, on='Frame', suffixes=('_real', '_gen'))
                if not merged_df.empty:
                    mae = (merged_df['Normalized_Game_Timer_real'] - merged_df['Normalized_Game_Timer_gen']).abs().mean()
                    all_maes.append(mae)
            except Exception as e:
                print(f"    Warning: Could not process pair '{basename}'. Reason: {e}")
    
    if not all_maes: return None
    return {'mean': np.mean(all_maes), 'std': np.std(all_maes)}

def calculate_win_rates(data_directory):
    """
    Calculates the win rates for P1 and P2 based on the final frame's health.
    """
    print(f"  Calculating win rates for '{os.path.basename(data_directory)}'...")
    all_files = glob.glob(os.path.join(data_directory, '*.csv'))
    if not all_files: return None

    p1_wins, p2_wins, draws = 0, 0, 0
    for filepath in all_files:
        try:
            df = pd.read_csv(filepath)
            if df.empty: continue
            last_frame = df.iloc[-1]
            p1_health, p2_health = last_frame['P1_Health'], last_frame['P2_Health']
            if p1_health > p2_health: p1_wins += 1
            elif p2_health > p1_health: p2_wins += 1
            else: draws += 1
        except Exception as e:
            print(f"    Warning: Could not calculate win rate for '{os.path.basename(filepath)}'. Reason: {e}")

    total_games = p1_wins + p2_wins + draws
    if total_games == 0: return {'p1_wins': 0, 'p2_wins': 0, 'draws': 0, 'total_games': 0, 'p1_win_rate': 0, 'p2_win_rate': 0}
    return {'p1_wins': p1_wins, 'p2_wins': p2_wins, 'draws': draws, 'total_games': total_games, 'p1_win_rate': p1_wins / total_games, 'p2_win_rate': p2_wins / total_games}

def create_kde_plot(real_data, generated_data, title, xlabel, output_path):
    """Creates and saves a Kernel Density Estimate plot."""
    plt.figure(figsize=(10, 6))
    sns.kdeplot(real_data, label='Real Gameplay (Ground Truth)', color='blue', fill=True, clip=(0, None))
    sns.kdeplot(generated_data, label='Generated Gameplay (Vision-Based)', color='red', fill=True, clip=(0, None))
    plt.title(title, fontsize=16); plt.xlabel(xlabel, fontsize=12); plt.ylabel('Probability Density', fontsize=12)
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6); plt.savefig(output_path)
    print(f"  Chart saved to '{os.path.relpath(output_path, project_root)}'")
    plt.close()

# ==========================================================
# 3. Main Execution Block
# ==========================================================

def main():
    print("--- Starting Tekken 3 Realism Benchmark ---")
    
    # --- Data Aggregation & Win Rates ---
    print("\nStep 1: Aggregating data and calculating win rates...")
    real_results = aggregate_results(REAL_DATA_DIR)
    generated_results = aggregate_results(GENERATED_DATA_DIR)
    generated_win_rates = calculate_win_rates(GENERATED_DATA_DIR)
    
    if real_results is None:
        print(f"\nFATAL ERROR: No 'real' data found in '{REAL_DATA_DIR}'. Cannot run benchmark."); return
    if generated_results is None:
        print(f"\nFATAL ERROR: No 'generated' data found in '{GENERATED_DATA_DIR}'. Cannot run benchmark."); return

    # --- Report Generation ---
    print("\nStep 2: Generating summary report...")
    summary_df = pd.DataFrame({
        'Metric': ['Health Monotonicity', 'Timer Monotonicity', 'Tick Frequency (Expected ~0.033)'],
        'Real Gameplay (Mean ± Std)': [
            f"{real_results['health_monotonicity_mean']:.4f} ± {real_results['health_monotonicity_std']:.4f}",
            f"{real_results['timer_monotonicity_mean']:.4f} ± {real_results['timer_monotonicity_std']:.4f}",
            f"{real_results['tick_frequency_mean']:.4f} ± {real_results['tick_frequency_std']:.4f}"
        ],
        'Generated Gameplay (Mean ± Std)': [
            f"{generated_results['health_monotonicity_mean']:.4f} ± {generated_results['health_monotonicity_std']:.4f}",
            f"{generated_results['timer_monotonicity_mean']:.4f} ± {generated_results['timer_monotonicity_std']:.4f}",
            f"{generated_results['tick_frequency_mean']:.4f} ± {generated_results['tick_frequency_std']:.4f}"
        ]
    })
    
    # --- Distribution and Accuracy Similarity ---
    print("\nStep 3: Calculating similarity and accuracy scores...")
    # FIX: Check if lists are empty before calculating distance to prevent crash
    damage_dist, timer_dist = None, None
    if real_results.get('damage_events') and generated_results.get('damage_events'):
        damage_dist = wasserstein_distance(real_results['damage_events'], generated_results['damage_events'])
    if real_results.get('timer_diffs') and generated_results.get('timer_diffs'):
        timer_dist = wasserstein_distance(real_results['timer_diffs'], generated_results['timer_diffs'])
    
    timer_accuracy = calculate_timer_accuracy(REAL_DATA_DIR, GENERATED_DATA_DIR)
    
    distance_df = pd.DataFrame({
        'Metric': ['Damage Event Similarity (Wasserstein)', 'Timer Rate Similarity (Wasserstein)', 'Timer Accuracy (MAE)'],
        'Score (Lower is Better)': [
            f"{damage_dist:.6f}" if damage_dist is not None else "N/A (Not enough data)",
            f"{timer_dist:.6f}" if timer_dist is not None else "N/A (Not enough data)",
            f"{timer_accuracy['mean']:.6f} ± {timer_accuracy['std']:.6f}" if timer_accuracy else "N/A (No matching files)"
        ],
        'Description': [
            'Compares the distribution of damage values.',
            'Compares the distribution of timer tick rates.',
            'Measures the avg. absolute error of the timer value per frame.'
        ]
    })

    # --- Generated Win Rate Report ---
    win_rate_df = None
    if generated_win_rates and generated_win_rates.get('total_games', 0) > 0:
        win_rate_df = pd.DataFrame({
            'Outcome': ['P1 Wins', 'P2 Wins', 'Draws', 'Total Games'],
            'Count': [generated_win_rates['p1_wins'], generated_win_rates['p2_wins'], generated_win_rates['draws'], generated_win_rates['total_games']],
            'Percentage': [
                f"{generated_win_rates['p1_win_rate']:.2%}", f"{generated_win_rates['p2_win_rate']:.2%}",
                f"{(generated_win_rates['draws'] / generated_win_rates['total_games']):.2%}", "100.00%"
            ]
        })

    # --- Print and Save ---
    print("\n--- Benchmark Summary ---"); print(summary_df.to_string(index=False))
    print("\n--- Distribution & Accuracy Scores (0.0 = Identical) ---"); print(distance_df[['Metric', 'Score (Lower is Better)']].to_string(index=False))
    if win_rate_df is not None:
        print("\n--- Generated Gameplay Win Rates ---"); print(win_rate_df.to_string(index=False))
    
    os.makedirs(os.path.join(OUTPUT_DIR, 'summary'), exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, 'summary', 'benchmark_summary.csv')
    with open(report_path, 'w', newline='') as f:
        summary_df.to_csv(f, index=False); f.write('\n')
        distance_df.to_csv(f, index=False); f.write('\n')
        if win_rate_df is not None: win_rate_df.to_csv(f, index=False)
    print(f"\nFull summary report saved to '{os.path.relpath(report_path, project_root)}'")

    # --- Generate Visualizations ---
    if damage_dist is not None and timer_dist is not None:
        print("\nStep 4: Generating visualizations...")
        charts_dir = os.path.join(OUTPUT_DIR, 'charts'); os.makedirs(charts_dir, exist_ok=True)
        create_kde_plot(real_results['damage_events'], generated_results['damage_events'], 'Comparison of Damage Event Distributions', 'Damage per Hit (Normalized Health)', os.path.join(charts_dir, 'damage_distribution.png'))
        create_kde_plot(real_results['timer_diffs'], generated_results['timer_diffs'], 'Comparison of Timer Rate Distributions', 'Change in Timer Value per Frame', os.path.join(charts_dir, 'timer_rate_distribution.png'))

    print("\n--- Benchmark Analysis Complete ---")

if __name__ == '__main__':
    main()

