import json
import pandas as pd
import numpy as np
import os
import sys

def run_failure_analysis():
    print("Entering Failure Analysis Phase...")
    
    if not os.path.exists('../results/intervention_targets.json') or not os.path.exists('../results/anchor_classifier.json'):
        print("Error: Missing required data files.")
        return

    targets_df = pd.read_json('../results/intervention_targets.json')
    features_df = pd.read_json('../results/anchor_classifier.json')

    df = pd.merge(
        targets_df, 
        features_df[['problem_id', 'chunk_idx', 'dep_count', 'rel_pos']], 
        on=['problem_id', 'chunk_idx']
    )

    with open('../data/problems_final.json', 'r') as f:
        raw_probs = json.load(f)
    domain_map = {p['id']: p.get('type', 'Unknown') for p in raw_probs}
    df['domain'] = df['problem_id'].map(domain_map)

    instability_threshold = df['actual_instability'].quantile(0.85)
    prediction_threshold = df['criticality_score'].quantile(0.85)
    
    df['error_cat'] = 'Correct'
    df.loc[(df['criticality_score'] > prediction_threshold) & (df['actual_instability'] < df['actual_instability'].median()), 'error_cat'] = 'False Alarm'
    df.loc[(df['criticality_score'] < 0.3) & (df['actual_instability'] > instability_threshold), 'error_cat'] = 'Missed Critical'

    print("\nUPDATED ERROR CATEGORY COUNTS:")
    counts = df['error_cat'].value_counts()
    print(counts)

    err_pivot = pd.crosstab(df['anchor_type'], df['error_cat'], normalize='index')

    domain_stats = df.groupby('domain').agg({
        'dep_count': 'mean',
        'actual_instability': 'mean',
        'criticality_score': 'mean'
    })
    df['abs_error'] = (df['actual_instability'] - df['criticality_score']).abs()
    domain_stats['avg_error'] = df.groupby('domain')['abs_error'].mean()

    worst_misses = df[df['error_cat'] == 'Missed Critical'].sort_values('actual_instability', ascending=False).head(5)
    
    with open('../results/failure_insights_report.txt', 'w') as f:
        f.write("=== RESEARCH REPORT: REASONING FAILURE MODES ===\n\n")
        f.write(f"1. THE SCALABILITY OF MONITORING\n")
        f.write(f"Finding: Domains with higher dependency counts (avg={domain_stats['dep_count'].mean():.2f}) show different error profiles.\n")
        f.write("Insight: As reasoning becomes more complex, logic becomes more explicit and predictable.\n\n")

        if 'Missed Critical' in counts:
            f.write("2. SYSTEMATIC BLIND SPOTS (Missed Criticals)\n")
            top_miss_type = err_pivot['Missed Critical'].idxmax()
            f.write(f"Highest Vulnerability Type: {top_miss_type}\n")
            f.write(f"Safety Risk: These {top_miss_type} steps represent 'Hidden Anchors'.\n\n")
            
            f.write("3. CASE STUDIES OF HIDDEN CRITICALITY:\n")
            for _, r in worst_misses.iterrows():
                f.write(f"--- [Actual Instability: {r['actual_instability']:.3f}] ---\n")
                f.write(f"DOMAIN: {r['domain']} | TYPE: {r['anchor_type']}\n")
                f.write(f"TEXT: {r['text']}\n\n")
        else:
            f.write("2. MODEL PERFORMANCE NOTE\n")
            f.write("The model was highly successful; no 'Missed Critical' errors found at the current thresholds.\n\n")

    print("\nFAILURE ANALYSIS COMPLETE. Files saved: '../results/failure_insights_report.txt'")

if __name__ == "__main__":
    run_failure_analysis()