import json
from datasets import load_dataset
from collections import defaultdict
from datetime import datetime
import time

def print_progress(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

print("="*60)
print_progress("Starting...")
print("="*60)

TARGET_PER_TYPE = 5
MAX_SAMPLES_TO_PROCESS = 30000
TIMEOUT_MINUTES = 160

print("="*60)
print("CONFIGURATION")
print("="*60)
print(f"  Target per type: {TARGET_PER_TYPE}")
print(f"  Max samples to check: {MAX_SAMPLES_TO_PROCESS}")
print(f"  Timeout: {TIMEOUT_MINUTES} minutes")
print(f"  Early stop: When all categories have {TARGET_PER_TYPE} problems")
print()

print("="*60)
print("[1/3] INITIALIZING STREAM")
print("="*60)
print_progress("Connecting to HuggingFace...")

try:
    dataset = load_dataset(
        "uzaymacar/math-rollouts",
        split="default",
        streaming=True
    )
    print_progress("Stream connected successfully!")
except Exception as e:
    print_progress(f"ERROR: {e}")
    exit(1)

print()
print("="*60)
print("[2/3] COLLECTING FILES AND GROUPING BY PROBLEM")
print("="*60)
print_progress("Starting file collection...")
print()

problem_files = defaultdict(dict)
processed = 0
files_collected = 0
errors = 0

start_time = time.time()

print("Progress (updates every 500 files):")
print("-" * 60)

problems_by_type_tracking = defaultdict(int)
goal_met = False

for item in dataset:
    if 'correct_base_solution' not in item.get('path', ''):
        continue
    
    elapsed_minutes = (time.time() - start_time) / 60
    if elapsed_minutes > TIMEOUT_MINUTES:
        print()
        print_progress(f"Timeout reached ({TIMEOUT_MINUTES} min)")
        break
    
    processed += 1
    if processed > MAX_SAMPLES_TO_PROCESS:
        print()
        print_progress(f"Max samples reached ({MAX_SAMPLES_TO_PROCESS})")
        break
    
    if processed % 500 == 0:
        problems_found = len([p for p in problem_files.values() if len(p) == 2])
        print_progress(
            f"Files: {processed} | Complete Problems: {problems_found} | "
            f"Time: {elapsed_minutes:.1f}min"
        )
        
        if problems_found > 0 and processed % 2000 == 0:
            print("   Categories so far: ", end="")
            for ptype, count in sorted(problems_by_type_tracking.items()):
                status = "✓" if count >= TARGET_PER_TYPE else f"{count}"
                print(f"{ptype[:8]}:{status} ", end="")
            print()
    
    try:
        path = item.get('path', '')
        filename = item.get('filename', '')
        content = item.get('content', '')
        
        if filename not in ['problem.json', 'chunks_labeled.json']:
            continue
        
        path_parts = path.split('/')
        problem_dir = None
        for part in path_parts:
            if part.startswith('problem_'):
                problem_dir = path.rsplit(part, 1)[0] + part
                break
        
        if not problem_dir:
            continue
        
        problem_files[problem_dir][filename] = content
        files_collected += 1
        
        if len(problem_files[problem_dir]) == 2:
            try:
                problem_data = json.loads(problem_files[problem_dir]['problem.json'])
                chunks_data = json.loads(problem_files[problem_dir]['chunks_labeled.json'])
                
                ptype = problem_data.get('type', 'General')
                
                reasoning_steps = [c.get('chunk', '') for c in chunks_data if len(c.get('chunk', '').strip()) > 10]
                
                if len(reasoning_steps) >= 5 and problems_by_type_tracking[ptype] < TARGET_PER_TYPE:
                    problems_by_type_tracking[ptype] += 1
                    
                    print_progress(
                        f"{ptype}: {problems_by_type_tracking[ptype]}/{TARGET_PER_TYPE} | "
                        f"{problem_data.get('nickname', problem_dir.split('/')[-1])[:40]}"
                    )
                    
                    if len(problems_by_type_tracking) >= 7:
                        all_full = all(count >= TARGET_PER_TYPE for count in problems_by_type_tracking.values())
                        if all_full and not goal_met:
                            goal_met = True
                            print()
                            print_progress(f"GOAL MET! All {len(problems_by_type_tracking)} categories have {TARGET_PER_TYPE}+ problems")
                            print_progress("Collecting a few more for buffer...")
                            MAX_SAMPLES_TO_PROCESS = min(MAX_SAMPLES_TO_PROCESS, processed + 2000)
                
            except:
                pass
        
    except Exception as e:
        errors += 1
        if errors <= 3:
            print_progress(f"Parse error: {str(e)[:50]}...")
        continue

complete_problems = {k: v for k, v in problem_files.items() 
                     if len(v) == 2 and 'problem.json' in v and 'chunks_labeled.json' in v}

print()
print("="*60)
print("FILE COLLECTION SUMMARY")
print("="*60)
print_progress(f"Total files processed: {processed}")
print_progress(f"Relevant files collected: {files_collected}")
print_progress(f"Complete problems: {len(complete_problems)}")
print_progress(f"Parse errors: {errors}")
print_progress(f"Time elapsed: {(time.time() - start_time) / 60:.1f} minutes")
print()

if not complete_problems:
    print_progress("ERROR: No complete problems collected!")
    exit(1)

print("="*60)
print("[3/3] PARSING PROBLEMS BY TYPE")
print("="*60)
print_progress("Parsing problem data...")

problems_by_type = defaultdict(list)
parse_errors = 0

for problem_path, files in complete_problems.items():
    try:
        problem_data = json.loads(files['problem.json'])
        chunks_data = json.loads(files['chunks_labeled.json'])
        
        ptype = problem_data.get('type', 'General')
        
        if len(problems_by_type[ptype]) >= TARGET_PER_TYPE:
            continue
        
        reasoning_steps = []
        for chunk_info in chunks_data:
            chunk_text = chunk_info.get('chunk', '')
            if chunk_text and len(chunk_text.strip()) > 10:
                reasoning_steps.append(chunk_text.strip())
        
        if len(reasoning_steps) < 5:
            continue
        
        problem_entry = {
            'problem': problem_data.get('problem', ''),
            'answer': problem_data.get('gt_answer', ''),
            'reasoning_steps': reasoning_steps,
            'num_steps': len(reasoning_steps),
            'level': problem_data.get('level', ''),
            'nickname': problem_data.get('nickname', problem_path.split('/')[-1]),
            'chunks_data': chunks_data,
        }
        
        problems_by_type[ptype].append(problem_entry)
        
    except Exception as e:
        parse_errors += 1
        if parse_errors <= 3:
            print_progress(f"Parse error: {str(e)[:50]}...")
        continue

print()
print("Problems by category:")
for ptype in sorted(problems_by_type.keys()):
    count = len(problems_by_type[ptype])
    status = "DONE" if count >= TARGET_PER_TYPE else "ERROR"
    print(f"  {status} {ptype}: {count}/{TARGET_PER_TYPE}")

if not problems_by_type:
    print_progress("ERROR: No problems parsed successfully!")
    exit(1)

print()
print("="*60)
print("[4/4] FORMATTING AND SAVING")
print("="*60)
print_progress("Creating output format...")

output_problems = []
problem_id = 1

for ptype in sorted(problems_by_type.keys()):
    for prob in problems_by_type[ptype]:
        full_solution = "\n\n".join(prob['reasoning_steps'])
        
        output_problems.append({
            'id': f'problem_{problem_id}',
            'problem': prob['problem'],
            'solution': full_solution,
            'answer': prob['answer'],
            'type': ptype,
            'level': prob['level'],
            'nickname': prob['nickname'],
            'source': 'Thought Anchors Dataset (streamed, deepseek-r1-distill)',
            'anchor_data': {
                'full_reasoning': full_solution,
                'reasoning_sentences': prob['reasoning_steps'],
                'num_sentences': prob['num_steps'],
                'final_answer': prob['answer'],
                'has_importance_scores': True,
                'chunks_with_importance': prob['chunks_data']
            }
        })
        problem_id += 1

output_file = 'data/problems_final.json'
print_progress(f"Saving to {output_file}...")

with open(output_file, 'w') as f:
    json.dump(output_problems, f, indent=2)

print_progress("Saved successfully!")

summary = {
    'collection_method': 'streaming_fixed_v2',
    'total_problems': len(output_problems),
    'collection_time_minutes': (time.time() - start_time) / 60,
    'files_processed': processed,
    'problems_by_type': {t: len(v) for t, v in problems_by_type.items()},
    'has_importance_scores': True,
    'importance_metrics_available': [
        'resampling_importance_accuracy',
        'resampling_importance_kl',
        'counterfactual_importance_accuracy',
        'counterfactual_importance_kl',
        'forced_importance_accuracy',
        'forced_importance_kl'
    ]
}

with open('collection_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print()
print("="*60)
print("SUCCESS!")
print("="*60)

type_counts = defaultdict(int)
for p in output_problems:
    type_counts[p['type']] += 1

print(f"\nFINAL STATISTICS:")
print(f"  • Total problems: {len(output_problems)}")
print(f"  • Categories: {len(type_counts)}")
print(f"  • Collection time: {(time.time() - start_time) / 60:.1f} minutes")

print("\nBy category:")
for ptype in sorted(type_counts.keys()):
    count = type_counts[ptype]
    status = "DONE" if count >= TARGET_PER_TYPE else "ERROR"
    print(f"  {status} {ptype}: {count}")

all_steps = [p['anchor_data']['num_sentences'] for p in output_problems]
avg_steps = sum(all_steps) / len(all_steps)
min_steps = min(all_steps)
max_steps = max(all_steps)

print(f"\nReasoning complexity:")
print(f"  - Average steps: {avg_steps:.1f}")
print(f"  - Range: {min_steps}-{max_steps}")

print("\n" + "="*60)
print("Ready for Step 2: Anchor Identification")
print("="*60)
print_progress("Collection completed successfully! Results in 'data/problems_final.json'")