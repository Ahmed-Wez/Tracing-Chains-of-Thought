import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import re

def classify_by_tags(chunk):
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

def run_causal_analysis():
    print("Initializing Cascade & Interaction Analysis...")
    with open('../data/problems_final.json', 'r') as f:
        problems = json.load(f)

    all_data = []

    for prob in problems:
        chunks = prob['anchor_data']['chunks_with_importance']
        num_chunks = len(chunks)

        for c in chunks:
            p = c.get('accuracy', 0)
            c['instability_val'] = p * (1 - p)

        for i, target_chunk in enumerate(chunks):
            target_idx = str(target_chunk['chunk_idx'])
            downstream_instabilities = []
            
            for follower_chunk in chunks[i+1:]:
                if target_idx in follower_chunk.get('depends_on', []):
                    downstream_instabilities.append(follower_chunk['instability_val'])
            
            cascade_score = np.mean(downstream_instabilities) if downstream_instabilities else 0
            
            row = {
                "problem_id": prob['id'],
                "chunk_idx": target_chunk['chunk_idx'],
                "anchor_type": classify_by_tags(target_chunk),
                "causal_instability": target_chunk['instability_val'],
                "path_shift_kl": target_chunk.get('resampling_importance_kl', 0),
                "rel_pos": target_chunk['chunk_idx'] / num_chunks,
                "dep_count": len(target_chunk.get('depends_on', [])),
                "cascade_influence": cascade_score
            }
            all_data.append(row)

    df = pd.DataFrame(all_data)

    print("\nRUNNING INTERACTION REGRESSION...")
    formula = "causal_instability ~ C(anchor_type) * rel_pos + dep_count + path_shift_kl + cascade_influence"
    model = smf.ols(formula=formula, data=df).fit()
    
    print(f"UPDATED R-SQUARED: {model.rsquared:.4f}")

    with open('../results/causal_final_report.txt', 'w') as f:
        f.write("=== CAUSAL VALIDATION ===\n\n")
        f.write(f"Final Model R-Squared: {model.rsquared:.4f}\n")
        f.write("-" * 40 + "\n")
        f.write("TOP CAUSAL DRIVERS (Significant Interactions):\n")
        for feat, p in model.pvalues.items():
            if p < 0.05:
                coef = model.params[feat]
                f.write(f"{feat:<40} | p={p:.2e} | Coef={coef:.4f}\n")
        
        f.write("\n" + "-" * 40 + "\n")
        f.write("CASCADE ANALYSIS (H4):\n")
        cascade_means = df.groupby('anchor_type')['cascade_influence'].mean().sort_values(ascending=False)
        f.write(cascade_means.to_string())

    print("\nAnalysis Complete! Results in '../results/causal_final_report.txt'")

if __name__ == "__main__":
    run_causal_analysis()