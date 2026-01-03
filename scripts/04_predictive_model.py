import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import re
import os

def classify_chunk(chunk):
    tags = set(chunk.get('function_tags', []))
    text = chunk.get('chunk', '').lower()
    if tags.intersection({"plan_generation", "problem_setup"}): return "Planning"
    if tags.intersection({"uncertainty_management", "self_checking"}): return "Error-Correction"
    if tags.intersection({"result_consolidation", "final_answer_emission"}): return "Decision"
    
    depends_on = chunk.get('depends_on', [])
    is_jump = False
    try:
        if depends_on:
            max_dep = max([int(d) for d in depends_on])
            is_jump = (chunk['chunk_idx'] - max_dep) > 7
    except: pass
    if any(text.startswith(k) for k in ["now", "next", "moving on"]) or is_jump:
        return "State-Transition"
    if tags.intersection({"active_computation", "computation"}) or (len(re.findall(r'[\$\\]', text)) > 5):
        return "Computational"
    return "Neutral/Elaboration"

def run_predictive_model():
    print("Loading data and building causal features...")
    if not os.path.exists('../data/problems_final.json'):
        print("Error: '../data/problems_final.json' not found.")
        return

    with open('../data/problems_final.json', 'r') as f:
        problems = json.load(f)

    all_data = []
    for prob in problems:
        chunks = prob['anchor_data']['chunks_with_importance']
        num_chunks = len(chunks)
        
        for c in chunks:
            p = c.get('accuracy', 0)
            c['instability_val'] = p * (1 - p)

        for i, target in enumerate(chunks):
            t_idx = str(target['chunk_idx'])
            downs = [f['instability_val'] for f in chunks[i+1:] if t_idx in f.get('depends_on', [])]
            cascade = np.mean(downs) if downs else 0
            
            all_data.append({
                "problem_id": prob['id'],
                "chunk_idx": target['chunk_idx'],
                "text": target['chunk'],
                "anchor_type": classify_chunk(target),
                "actual_instability": target['instability_val'],
                "path_shift_kl": target.get('resampling_importance_kl', 0),
                "rel_pos": target['chunk_idx'] / num_chunks,
                "dep_count": len(target.get('depends_on', [])),
                "cascade_influence": cascade
            })

    df = pd.DataFrame(all_data)

    unique_pids = df['problem_id'].unique()
    train_ids, test_ids = train_test_split(unique_pids, test_size=0.30, random_state=42)
    
    train_df = df[df['problem_id'].isin(train_ids)].copy()
    test_df = df[df['problem_id'].isin(test_ids)].copy()

    print(f"Training on {len(train_ids)} problems, Testing on {len(test_ids)} problems.")

    formula = "actual_instability ~ C(anchor_type) * rel_pos + dep_count + path_shift_kl + cascade_influence"
    model = smf.ols(formula=formula, data=train_df).fit()
    
    test_df['predicted_criticality'] = model.predict(test_df)
    
    c_min, c_max = test_df['predicted_criticality'].min(), test_df['predicted_criticality'].max()
    test_df['criticality_score'] = (test_df['predicted_criticality'] - c_min) / (c_max - c_min)

    train_r2 = model.rsquared
    test_r2 = r2_score(test_df['actual_instability'], test_df['predicted_criticality'])
    correlation = test_df['actual_instability'].corr(test_df['predicted_criticality'])

    print(f"\nVALIDATION RESULTS:")
    print(f"   Train R²: {train_r2:.4f}")
    print(f"   Test R²:  {test_r2:.4f}")
    print(f"   Pred/Actual Correlation: {correlation:.4f}")

    test_df = test_df.sort_values('criticality_score', ascending=False)    
    test_df['intervention_priority'] = 'Medium'
    top_10 = int(len(test_df) * 0.1)
    test_df.iloc[:top_10, test_df.columns.get_loc('intervention_priority')] = 'High'
    test_df.iloc[-top_10:, test_df.columns.get_loc('intervention_priority')] = 'Low'

    high_targets = test_df[test_df['intervention_priority'] == 'High'].head(5)
    
    print("\nHIGH PRIORITY INTERVENTION TARGETS (Predicted to break reasoning):")
    for _, r in high_targets.iterrows():
        print(f"   - [{r['anchor_type']}] Prob: {r['problem_id']} | Chunk: {r['chunk_idx']} | Score: {r['criticality_score']:.2f}")
        print(f"     Text: {r['text'][:80]}...")

    output_cols = ['problem_id', 'chunk_idx', 'criticality_score', 'actual_instability', 'intervention_priority', 'text', 'anchor_type']
    test_df[output_cols].to_json('../results/intervention_targets.json', orient='records', indent=2)

    with open('../results/prediction_report.txt', 'w') as f:
        f.write("=== PREDICTIVE MODEL VALIDATION ===\n")
        f.write(f"Train R²: {train_r2:.4f}\nTest R²: {test_r2:.4f}\n")
        f.write(f"Correlation: {correlation:.4f}\n\n")
        f.write("MODELS HIGHLIGHTS:\n")
        f.write(model.summary().as_text())

    print("\nSaved targets to '../results/intervention_targets.json' and report to '../results/prediction_report.txt'")

if __name__ == "__main__":
    run_predictive_model()