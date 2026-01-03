import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import re
import os
import sys

def get_math_symbols(text):
    math_content = re.findall(r'\\\(.*?\\\)' , text) + re.findall(r'\$.*?\$', text)
    symbols = set()
    for segment in math_content:
        found = re.findall(r'\\[a-z]+|[a-zA-Z]', segment)
        symbols.update(found)
    return symbols

def classify_chunk(chunk):
    tags = set(chunk.get('function_tags', []))
    text = chunk.get('chunk', '').lower()
    if tags.intersection({"plan_generation", "problem_setup"}): return "Planning"
    if tags.intersection({"uncertainty_management", "self_checking"}): return "Error-Correction"
    if tags.intersection({"result_consolidation", "final_answer_emission"}): return "Decision"
    
    depends_on = chunk.get('depends_on', [])
    try:
        max_dep = max([int(d) for d in depends_on]) if depends_on else 0
        if (chunk['chunk_idx'] - max_dep) > 7: return "State-Transition"
    except: pass
    
    if any(text.startswith(k) for k in ["now", "next", "moving on"]): return "State-Transition"
    if tags.intersection({"active_computation", "computation"}) or (len(re.findall(r'[\$\\]', text)) > 5):
        return "Computational"
    return "Neutral/Elaboration"

def run_hidden_anchor_detector():
    print("Entering Hidden Anchor Detection Phase...")
    
    if not os.path.exists('../data/problems_final.json'):
        print("Error: '../data/problems_final.json' not found. Please run 01_load_and_process.py first.")
        return

    with open('../data/problems_final.json', 'r') as f:
        problems = json.load(f)

    all_rows = []

    for prob in problems:
        chunks = prob['anchor_data']['chunks_with_importance']
        num_chunks = len(chunks)
        seen_symbols = set()
        prev_latex_density = 0
        
        for c in chunks:
            acc = c.get('accuracy', 0)
            c['instab'] = acc * (1 - acc)

        for i, chunk in enumerate(chunks):
            text = chunk.get('chunk', '')
            text_lower = text.lower()
            
            current_symbols = get_math_symbols(text)
            novel_symbols = len(current_symbols - seen_symbols)
            seen_symbols.update(current_symbols)
            
            def_chars = len(re.findall(r'[=≡≜]|\\approx', text))
            def_density = def_chars / len(text) if len(text) > 0 else 0
            
            found_keys = any(k in text_lower for k in ["let", "note that", "recall", "given that"])
            is_trans = any(k in text_lower for k in ["so", "thus", "therefore"]) and def_chars > 0
            
            latex_chars = len(re.findall(r'[\$\\]', text))
            curr_latex_density = latex_chars / len(text) if len(text) > 0 else 0
            abs_shift = abs(curr_latex_density - prev_latex_density)
            prev_latex_density = curr_latex_density
            
            foundation_score = (0.30 * novel_symbols + 0.25 * (def_density * 10) + 
                                0.20 * (1 if found_keys else 0) + 0.15 * (1 if is_trans else 0) + 
                                0.10 * abs_shift)

            t_idx = str(chunk['chunk_idx'])
            downs = [f['instab'] for f in chunks[i+1:] if t_idx in f.get('depends_on', [])]
            cascade = np.mean(downs) if downs else 0
            
            all_rows.append({
                "problem_id": prob['id'],
                "chunk_idx": chunk['chunk_idx'],
                "text": text,
                "anchor_type": classify_chunk(chunk),
                "actual_instability": chunk['instab'],
                "path_shift_kl": chunk.get('resampling_importance_kl', 0),
                "rel_pos": chunk['chunk_idx'] / num_chunks,
                "dep_count": len(chunk.get('depends_on', [])),
                "cascade_influence": cascade,
                "foundation_score": foundation_score
            })
            
    df = pd.DataFrame(all_rows)

    print("\nTraining Augmented Model for Hidden Anchor Detection...")
    formula = """actual_instability ~ C(anchor_type) * rel_pos + 
                 dep_count + path_shift_kl + cascade_influence + foundation_score + 
                 C(anchor_type):foundation_score"""
    
    model = smf.ols(formula=formula, data=df).fit()
    
    print(f"Augmented Model R-Squared: {model.rsquared:.4f}")
    
    df['pred_augmented'] = model.predict(df)
    hidden_anchors = df.sort_values('foundation_score', ascending=False).head(10)

    with open('../results/hidden_anchor_model_augmented.txt', 'w') as f:
        f.write("=== HIDDEN ANCHOR DETECTION MODEL PERFORMANCE ===\n")
        f.write(f"Augmented R-Squared: {model.rsquared:.4f}\n\n")
        f.write("TOP PREDICTORS:\n")
        f.write(model.pvalues.sort_values().to_string())

    with open('../results/hidden_anchor_safety_analysis.txt', 'w') as f:
        f.write("=== SAFETY ANALYSIS: DETECTING STEALTH ANCHORS ===\n\n")
        f.write("TOP 'FOUNDATIONAL' STEPS DETECTED:\n")
        for _, r in hidden_anchors.iterrows():
            f.write(f"Foundation Score: {r['foundation_score']:.2f} | Instability: {r['actual_instability']:.3f}\n")
            f.write(f"TEXT: {r['text'][:120]}...\n\n")

    print("\nHIDDEN ANCHOR DETECTION ANALYSIS COMPLETE.")
    print(f"Check '../results/hidden_anchor_model_augmented.txt' for the augmented model performance.")
    print(f"Check '../results/hidden_anchor_safety_analysis.txt' for the safety analysis.")

if __name__ == "__main__":
    run_hidden_anchor_detector()