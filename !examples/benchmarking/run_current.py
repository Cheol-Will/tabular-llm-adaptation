import pickle
from pathlib import Path

def load_all_results():
    base_path = Path("examples/benchmarking/results/FTTransformer/data/FTTransformer")
    all_data = {}
    
    pkl_paths = list(base_path.glob("*/*/results.pkl"))
    print(f"Total files found: {len(pkl_paths)}")

    for pkl_path in pkl_paths:
        try:
            seed = pkl_path.parent.name
            task_id = pkl_path.parent.parent.name
            
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            
            if task_id not in all_data:
                all_data[task_id] = {}
            
            all_data[task_id][seed] = data
            
        except Exception as e:
            print(f"Error loading {pkl_path}: {e}")

    return all_data

if __name__ == "__main__":
    results = load_all_results()
    
    for task_id in sorted(results.keys()):
        seeds = list(results[task_id].keys())
        print(f"Task: {task_id} | Seeds: {seeds}")