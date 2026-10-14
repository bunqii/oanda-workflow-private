import os
import json
import shutil
import requests
import subprocess
from datetime import datetime
from huggingface_hub import HfApi, create_repo
from download_history import bulk_download_and_split, OANDA_URL, HEADERS, env_vars

GH_TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("PAT_TOKEN") or env_vars.get("PAT_TOKEN") or env_vars.get("GITHUB_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN") or env_vars.get("HF_TOKEN")
GH_USERNAME = os.environ.get("GH_USERNAME") or env_vars.get("GH_USERNAME", "")

tf_env = os.environ.get("TIMEFRAMES")
if tf_env and tf_env.strip():
    TIMEFRAMES_TO_TEST = [t.strip() for t in tf_env.split(",") if t.strip()]
else:
    TIMEFRAMES_TO_TEST = ["D"]
# Mapping of Timeframe to Target Platform (True = HuggingFace, False = GitHub)
TF_TO_HF = {
    "5S": True, "5": True, "60": True,
    "1": False, "D": False
}

def get_all_instruments():
    account_id = os.environ.get('OANDA_ACCOUNT_ID') or env_vars.get('OANDA_ACCOUNT_ID')
    url = f"{OANDA_URL}/accounts/{account_id}/instruments"
    res = requests.get(url, headers=HEADERS)
    if res.status_code == 200:
        instruments = res.json().get("instruments", [])
        return {inst['name']: inst for inst in instruments}
    return {}

def run_cmd(cmd, cwd=None):
    res = subprocess.run(cmd, shell=True, cwd=cwd, text=True, capture_output=True)
    if res.returncode != 0 and "already exists" not in res.stderr:
        print(f"[!] Command failed: {cmd}\n{res.stderr}")
        return False
    return True

def get_gh_username():
    user = os.environ.get("GH_USERNAME") or env_vars.get("GH_USERNAME")
    if user:
        return user
    try:
        res = requests.get("https://api.github.com/user", headers={"Authorization": f"token {GH_TOKEN}"})
        if res.status_code == 200:
            return res.json().get("login", "")
    except Exception:
        pass
    return ""

def get_hf_username():
    user = os.environ.get("HF_USERNAME") or env_vars.get("HF_USERNAME")
    if user:
        return user
    try:
        api = HfApi(token=HF_TOKEN)
        return api.whoami().get("name", "")
    except Exception:
        pass
    return ""

def create_github_repo(repo_name, private=True):
    headers = {
        "Authorization": f"token {GH_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    payload = {
        "name": repo_name,
        "private": private,
        "auto_init": True
    }
    res = requests.post("https://api.github.com/user/repos", headers=headers, json=payload)
    if res.status_code in [201, 422]:  # 201 Created, 422 already exists
        return True
    print(f"[!] GitHub repo creation failed ({res.status_code}): {res.text}")
    return False

def push_to_github(repo_name, folder_to_push):
    gh_user = get_gh_username()
    # 1. Create repo if not exists via GitHub REST API
    create_github_repo(repo_name, private=True)
    
    # 2. Clone it
    clone_dir = f"gh_clone_{repo_name}"
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir, ignore_errors=True)
        
    run_cmd(f"git clone https://x-access-token:{GH_TOKEN}@github.com/{gh_user}/{repo_name}.git {clone_dir}")
    
    # 3. Copy files over (Creates D/ or 1/ folder inside clone_dir)
    tf_folder = os.path.basename(folder_to_push)
    dest_dir = os.path.join(clone_dir, tf_folder)
    os.makedirs(dest_dir, exist_ok=True)
    
    for item in os.listdir(folder_to_push):
        s = os.path.join(folder_to_push, item)
        d = os.path.join(dest_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
    
    # 4. Commit and push
    subprocess.run(["git", "config", "user.name", "github-actions[bot]"], cwd=clone_dir)
    subprocess.run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"], cwd=clone_dir)
    subprocess.run(["git", "add", "."], cwd=clone_dir)
    c_res = subprocess.run(["git", "commit", "-m", "Automated data update"], cwd=clone_dir, capture_output=True, text=True)
    if c_res.returncode == 0:
        subprocess.run(["git", "branch", "-M", "main"], cwd=clone_dir)
        p_res = subprocess.run(["git", "push", "-u", "origin", "main"], cwd=clone_dir, capture_output=True, text=True)
        if p_res.returncode != 0:
            print(f"[!] Push failed: {p_res.stderr}")
        else:
            print(f"     Successfully pushed {repo_name} to GitHub!")
    else:
        if "nothing to commit" not in c_res.stdout:
            print(f"[!] Commit failed: {c_res.stderr}")
    
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir, ignore_errors=True)

def push_to_huggingface(repo_name, folder_to_push):
    api = HfApi(token=HF_TOKEN)
    hf_user = get_hf_username()
    repo_id = f"{hf_user}/{repo_name}"
    
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=True)
    except Exception:
        pass # Already exists
        
    tf_folder = os.path.basename(folder_to_push)
    api.upload_folder(
        folder_path=folder_to_push,
        repo_id=repo_id,
        path_in_repo=tf_folder,
        repo_type="dataset",
        commit_message="Automated data update"
    )

def get_repo_name(inst_name, inst_dict):
    display = inst_dict[inst_name].get('displayName', '').replace('/', '').replace('\\', '').replace(' ', '-')
    name_no_us = inst_name.replace("_", "")
    return f"OANDA_{name_no_us}_{display}"

def get_chunk_size(tf):
    if tf == "D": return "10_year"
    if tf in ["1", "5", "60"]: return "month"
    return "week"

def main():
    if not GH_TOKEN or not HF_TOKEN:
        print("Missing GITHUB_TOKEN or HF_TOKEN. Make sure they are set in the environment.")
        return

    tracker_file = "tracker.json"
    if os.path.exists(tracker_file):
        with open(tracker_file, "r") as f:
            tracker = json.load(f)
    else:
        tracker = {}

    inst_dict = get_all_instruments()
    
    ticker_filter = os.environ.get("TICKER_FILTER")
    if ticker_filter:
        inst_dict = {k: v for k, v in inst_dict.items() if k == ticker_filter}
        print(f"[*] Filter applied: processing only {ticker_filter}")
        
    for inst_name, inst_info in inst_dict.items():
        repo_name = get_repo_name(inst_name, inst_dict)
        print(f"\n=========================================")
        print(f"Processing {inst_name} -> {repo_name}")
        
        if inst_name not in tracker:
            tracker[inst_name] = {}
            
        updated_any = False
        
        for tf in TIMEFRAMES_TO_TEST:
            print(f"  -> Timeframe {tf}")
            
            last_date_str = tracker[inst_name].get(tf, None)
            if last_date_str:
                start_dt = datetime.fromisoformat(last_date_str)
                print(f"     Resuming from {start_dt}")
            else:
                start_dt = None
                print(f"     Downloading from inception")
            
            save_dir = os.path.join("tmp_data", repo_name, tf)
            os.makedirs(save_dir, exist_ok=True)
            
            # Map friendly timeframe back to OANDA timeframe for the downloader
            oanda_tf = "D"
            if tf == "5S": oanda_tf = "S5"
            elif tf == "1": oanda_tf = "M1"
            elif tf == "5": oanda_tf = "M5"
            elif tf == "60": oanda_tf = "H1"
            
            # Run the synchronous downloader (saves CSV and POD files locally in tmp_data)
            bulk_download_and_split(inst_name, oanda_tf, start_dt, None, get_chunk_size(tf), save_dir, file_prefix=repo_name, file_tf=tf)
            
            csv_files = [f for f in os.listdir(save_dir) if f.endswith(".csv")]
            if not csv_files:
                print("     No new data to push.")
                continue
            
            # Check POD files to see what was VERIFIED or VERIFIED_PARTIAL
            pod_files = [f for f in os.listdir(save_dir) if f.endswith(".pod")]
            latest_end = None
            for pf in pod_files:
                with open(os.path.join(save_dir, pf)) as p:
                    data = json.load(p)
                    stat = data.get("status")
                    if stat in ["VERIFIED", "EMPTY"]:
                        c_end_str = data.get("chunk_end")
                        c_end_dt = datetime.strptime(c_end_str, "%Y-%m-%d")
                        if not latest_end or c_end_dt > latest_end:
                            latest_end = c_end_dt
                    elif stat == "VERIFIED_PARTIAL":
                        last_c = data.get("csv_last_candle")
                        if last_c and last_c.get("Time"):
                            time_str = last_c["Time"]
                            clean_str = time_str.replace("T", " ")[:19]
                            try:
                                c_end_dt = datetime.strptime(clean_str, "%Y-%m-%d %H:%M:%S")
                            except Exception:
                                c_end_dt = datetime.strptime(clean_str[:10], "%Y-%m-%d")
                            if not latest_end or c_end_dt > latest_end:
                                latest_end = c_end_dt
            
            if latest_end:
                if TF_TO_HF.get(tf, False):
                    print("     Pushing new data to HuggingFace...")
                    push_to_huggingface(repo_name, save_dir)
                else:
                    print("     Pushing new data to GitHub...")
                    push_to_github(repo_name, save_dir)
                
                tracker[inst_name][tf] = latest_end.isoformat()
                updated_any = True
                
        if updated_any:
            print(f"  -> Updating master tracker for {inst_name}")
            with open(tracker_file, "w") as f:
                json.dump(tracker, f, indent=2)
            
            # Commit and push tracker instantly if inside a git repository
            if os.path.exists(".git"):
                subprocess.run(["git", "config", "user.name", "github-actions[bot]"])
                subprocess.run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"])
                subprocess.run(["git", "add", "tracker.json"])
                subprocess.run(["git", "commit", "-m", f"Update tracker for {inst_name}"])
                subprocess.run(["git", "push", "origin", "main"])
            
        if os.path.exists("tmp_data"):
            shutil.rmtree("tmp_data", ignore_errors=True)

if __name__ == "__main__":
    main()
