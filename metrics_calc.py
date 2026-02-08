import pandas as pd
import numpy as np
import os
from pathlib import Path

def calculate_metrics_for_csv(csv_file):
    """Calculate metrics for last 3 columns of a CSV file."""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Get the last 3 columns (excluding the last 'used_core_inverse' column)
        target_columns = df.columns[-4:-1]  # Gets columns at positions n-3, n-2, n-1
        
        print(f"\n{'='*60}")
        print(f"File: {csv_file}")
        print(f"{'='*60}")
        
        # Calculate metrics for each target column
        results = {}
        for col in target_columns:
            data = df[col].astype(float)
            
            mean_val = np.mean(data)
            median_val = np.median(data)
            std_val = np.std(data)
            min_val = np.min(data)
            max_val = np.max(data)
            
            results[col] = {
                'mean': mean_val,
                'median': median_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'count': len(data)
            }
            
            # Print results for this column
            print(f"\n{col}:")
            print(f"  Count:     {len(data):>6}")
            print(f"  Mean:      {mean_val:>10.4f}")
            print(f"  Median:    {median_val:>10.4f}")
            print(f"  Std Dev:   {std_val:>10.4f}")
            print(f"  Min:       {min_val:>10.4f}")
            print(f"  Max:       {max_val:>10.4f}")
        
        return results
        
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return None

def process_all_csv_files(folder_path):
    """Process all CSV files in a folder."""
    folder = Path(folder_path)
    
    # Find all CSV files in the folder
    csv_files = list(folder.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    all_results = {}
    
    # Process each CSV file
    for csv_file in csv_files:
        results = calculate_metrics_for_csv(csv_file)
        if results:
            all_results[csv_file.name] = results
    
    return all_results

def export_summary_to_csv(all_results, output_file="summary_metrics.csv"):
    """Export summary of all metrics to a CSV file."""
    if not all_results:
        print("No results to export")
        return
    
    summary_data = []
    
    for filename, metrics in all_results.items():
        for col_name, col_metrics in metrics.items():
            summary_data.append({
                'filename': filename,
                'column': col_name,
                'count': col_metrics['count'],
                'mean': col_metrics['mean'],
                'median': col_metrics['median'],
                'std': col_metrics['std'],
                'min': col_metrics['min'],
                'max': col_metrics['max']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_file, index=False)
    # print(f"\nSummary exported to: {output_file}")
    
    return summary_df

print("My Adaptive - No Downsampling Full Quality Image")
single_result_DEEP_NO = calculate_metrics_for_csv(r"C:\Users\Michel Massad\Desktop\Robotics\Code\results_final.csv")

# print("Old Adaptive (My adaptive old with no gating)")
single_result_OLD_ADAPTIVE = calculate_metrics_for_csv(r"C:\Users\Michel Massad\Desktop\Robotics\Code\results_1.csv")


# # print("CHAT Adaptive - No Downsampling Full Quality Image")
# # single_result_2_CHAT_NO = calculate_metrics_for_csv(r"C:\Users\Michel Massad\Desktop\Robotics\Code\results_final_adaptive_new_down_scale.csv")

# print("My adaptive - 900 px")
# single_result_3_DEEP_900 = calculate_metrics_for_csv(r"C:\Users\Michel Massad\Desktop\Robotics\Code\results_SIFT_ADAPTIVE.csv")

# print("MAGSAC")
# single_result_MAGSAC = calculate_metrics_for_csv(r"C:\Users\Michel Massad\Desktop\Robotics\Code\results_final_adaptive_USAC_MAGSAC.csv")
# # print("CHAT adaptive - 900 px")
# single_result_4_CHAT_900 = calculate_metrics_for_csv(r"C:\Users\Michel Massad\Desktop\Robotics\Code\results_SIFT_ADAPTIVE_NEW_Fix_2.csv")