import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind
from sklearn.metrics import r2_score
import importlib
import os
import sys

def load_and_enrich_data():
    with open('../data/problems_final.json', 'r') as f:
        problems = json.load(f)

    all_data = []
    for prob in problems:
        chunks = prob['anchor_data']['chunks_with_importance']
        num_chunks = len(chunks)
        
        for i, target in enumerate(chunks):
            p = target.get('accuracy', 0)
            instability = p * (1 - p)
            t_idx = str(target['chunk_idx'])
            downs = [f.get('accuracy', 0) * (1-f.get('accuracy', 0)) for f in chunks[i+1:] if t_idx in f.get('depends_on', [])]
            cascade = np.mean(downs) if downs else 0
            
            classify_chunk = importlib.import_module("04_predictive_model").classify_chunk            

            all_data.append({
                "problem_id": prob['id'],
                "category": prob.get('type', 'Unknown'),
                "chunk_idx": target['chunk_idx'],
                "anchor_type": classify_chunk(target),
                "actual_instability": instability,
                "path_shift_kl": target.get('resampling_importance_kl', 0),
                "rel_pos": target['chunk_idx'] / num_chunks,
                "dep_count": len(target.get('depends_on', [])),
                "cascade_influence": cascade,
                "ans_diversity": target.get('accuracy', 0)
            })
    return pd.DataFrame(all_data)

def run_cross_domain_validation():
    print("Initializing Cross-Domain Validation...")
    df = load_and_enrich_data()
    
    train_df = df.sample(frac=0.7, random_state=42)
    test_df = df.drop(train_df.index).copy()
    
    formula = "actual_instability ~ C(anchor_type) * rel_pos + dep_count + path_shift_kl + cascade_influence"
    model = smf.ols(formula=formula, data=train_df).fit()
    test_df['pred_crit'] = model.predict(test_df)
    
    test_df['quartile'] = pd.qcut(test_df['pred_crit'], 4, labels=['Low', 'Mid-Low', 'Mid-High', 'High'])
    
    high_group = test_df[test_df['quartile'] == 'High']['actual_instability']
    low_group = test_df[test_df['quartile'] == 'Low']['actual_instability']
    
    t_stat, p_val = ttest_ind(high_group, low_group)
    cohen_d = (high_group.mean() - low_group.mean()) / np.sqrt((high_group.var() + low_group.var()) / 2)

    print(f"\nQUARTILE VALIDATION (High vs Low Predictions):")
    print(f"   T-Test p-value: {p_val:.4e}")
    print(f"   Effect Size (Cohen's d): {cohen_d:.4f}")
    
    print("\nRUNNING CROSS-DOMAIN GENERALIZATION (Leave-One-Category-Out)...")
    categories = df['category'].unique()
    generalization_results = []
    
    for cat in categories:
        loco_train = df[df['category'] != cat]
        loco_test = df[df['category'] == cat]
        
        if len(loco_test) < 10: continue
        
        loco_model = smf.ols(formula=formula, data=loco_train).fit()
        loco_preds = loco_model.predict(loco_test)
        loco_r2 = r2_score(loco_test['actual_instability'], loco_preds)
        
        generalization_results.append({"Domain": cat, "Out-of-Domain R²": loco_r2})
    
    gen_df = pd.DataFrame(generalization_results)
    print(gen_df.to_string(index=False))

    output = {
        "quartile_validation": {
            "p_value": p_val,
            "cohens_d": cohen_d,
            "high_mean": high_group.mean(),
            "low_mean": low_group.mean()
        },
        "generalization": generalization_results
    }
    
    with open('../results/prediction_validation_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\nVALIDATION COMPLETE. Results saved to '../results/prediction_validation_results.json'")

if __name__ == "__main__":
    run_cross_domain_validation()