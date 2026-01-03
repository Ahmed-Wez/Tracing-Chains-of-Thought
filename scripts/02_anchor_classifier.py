import json
import pandas as pd
import numpy as np
import re

def calculate_latex_density(text):
    if not text or len(text) == 0: return 0
    latex_chars = len(re.findall(r'[\$\\\{\}\_\^]', text))
    return latex_chars / len(text)

def classify_by_tags(chunk):
    tags = set(chunk.get('function_tags', []))
    text = chunk['chunk'].lower()
    
    # 1. Planning
    if tags.intersection({"plan_generation", "problem_setup"}):
        return "Planning"
    
    # 2. Error-Correction
    if tags.intersection({"uncertainty_management", "self_checking"}):
        return "Error-Correction"
    
    # 3. Decision
    if tags.intersection({"result_consolidation", "final_answer_emission"}):
        return "Decision"
    
    # 4. State-Transition
    depends_on = chunk.get('depends_on', [])
    is_jump = False
    try:
        if depends_on:
            max_dep = max([int(d) for d in depends_on])
            is_jump = (chunk['chunk_idx'] - max_dep) > 7
    except: pass
    
    if any(text.startswith(k) for k in ["now", "next", "moving on"]) or is_jump:
        return "State-Transition"

    # 5. Computational
    if tags.intersection({"active_computation", "computation"}) or calculate_latex_density(chunk['chunk']) > 0.1:
        return "Computational"

    return "Neutral/Elaboration"

def run_anchor_classifier():
    with open('../data/problems_final.json', 'r') as f:
        problems = json.load(f)

    all_rows = []
    for prob in problems:
        chunks = prob['anchor_data']['chunks_with_importance']
        num_chunks = len(chunks)
        
        prob_rows = []
        for chunk in chunks:
            row = {
                "problem_id": prob['id'],
                "chunk_idx": chunk['chunk_idx'],
                "text": chunk['chunk'],
                "anchor_type": classify_by_tags(chunk),
                "resamp_acc": chunk.get('resampling_importance_accuracy', 0),
                "forced_acc": chunk.get('forced_importance_accuracy', 0),
                "counter_acc": chunk.get('counterfactual_importance_accuracy', 0),
                "rel_pos": chunk['chunk_idx'] / num_chunks,
                "latex_density": calculate_latex_density(chunk['chunk']),
                "dep_count": len(chunk.get('depends_on', []))
            }
            prob_rows.append(row)
        
        p_df = pd.DataFrame(prob_rows)
        for col in ['resamp_acc', 'forced_acc', 'counter_acc']:
            if p_df[col].std() > 0:
                p_df[f'{col}_z'] = (p_df[col] - p_df[col].mean()) / p_df[col].std()
            else:
                p_df[f'{col}_z'] = 0
        
        all_rows.extend(p_df.to_dict('records'))

    df = pd.DataFrame(all_rows)
    df.to_json('../results/anchor_classifier.json', orient='records', indent=2)
    print(f"Anchor classifier completed successfully! Results in '../results/anchor_classifier.json'")

if __name__ == "__main__":
    run_anchor_classifier()