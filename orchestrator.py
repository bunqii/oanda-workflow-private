import os
import json
import shutil
import requests
import subprocess
from datetime import datetime, timezone
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
    if isinstance(cmd, list):
        res = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    else:
        res = subprocess.run(cmd, shell=True, cwd=cwd, text=True, capture_output=True)
    if res.returncode != 0 and "already exists" not in res.stderr and "nothing to commit" not in res.stdout:
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

def get_repo_name(inst_name, inst_dict):
    display = inst_dict[inst_name].get('displayName', '').replace('/', '').replace('\\', '').replace(' ', '-')
    name_no_us = inst_name.replace("_", "")
    return f"OANDA_{name_no_us}_{display}"

def get_chunk_size(tf):
    if tf == "D": return "10_year"
    if tf in ["1", "5", "60"]: return "month"
    return "week"

def map_oanda_tf(tf):
    if tf == "5S": return "S5"
    if tf == "1": return "M1"
    if tf == "5": return "M5"
    if tf == "60": return "H1"
    return "D"

def sync_ticker_github(inst_name, repo_name, tracker):
    gh_user = get_gh_username()
    create_github_repo(repo_name, private=True)
    
    clone_dir = f"gh_clone_{repo_name}"
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir, ignore_errors=True)
        
    print(f"     Cloning repository {gh_user}/{repo_name}...")
    clone_ok = run_cmd(f"git clone https://x-access-token:{GH_TOKEN}@github.com/{gh_user}/{repo_name}.git {clone_dir}")
    if not clone_ok:
        print(f"[!] Could not clone {repo_name}. Skipping.")
        return False
        
    updated_any = False
    
    for tf in TIMEFRAMES_TO_TEST:
        if TF_TO_HF.get(tf, False):
            continue  # Handled on HuggingFace
            
        print(f"  -> Timeframe {tf} (Target: GitHub)")
        tf_dir = os.path.join(clone_dir, tf)
        os.makedirs(tf_dir, exist_ok=True)
        
        oanda_tf = map_oanda_tf(tf)
        chunk_size = get_chunk_size(tf)
        
        # Download directly into repo directory.
        # With start_dt=None, end_dt=None:
        # - Any completed chunks (status VERIFIED) in tf_dir are skipped instantly (0 seconds)
        # - Any incomplete or missing chunks are fetched with full historical data
        bulk_download_and_split(
            inst_name, oanda_tf, None, None, chunk_size, tf_dir,
            file_prefix=repo_name, file_tf=tf
        )
        
        pod_files = sorted([f for f in os.listdir(tf_dir) if f.endswith(".pod")])
        csv_files = sorted([f for f in os.listdir(tf_dir) if f.endswith(".csv")])
        
        if csv_files:
            total_rows = 0
            first_candle_time = None
            last_candle_time = None
            
            for pf in pod_files:
                try:
                    with open(os.path.join(tf_dir, pf)) as p:
                        pdata = json.load(p)
                        total_rows += pdata.get("csv_row_count", 0)
                        if not first_candle_time and pdata.get("csv_first_candle"):
                            first_candle_time = pdata["csv_first_candle"].get("Time")
                        if pdata.get("csv_last_candle"):
                            last_candle_time = pdata["csv_last_candle"].get("Time")
                except Exception:
                    pass
                    
            tracker[inst_name][tf] = {
                "first_candle": first_candle_time,
                "last_candle": last_candle_time,
                "total_rows": total_rows,
                "chunks_count": len(csv_files),
                "platform": "GitHub",
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            print(f"     Status: {len(csv_files)} chunks, {total_rows} total rows ({first_candle_time} -> {last_candle_time})")

    # Check if files changed in the repository
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=clone_dir, capture_output=True, text=True).stdout
    if git_status.strip():
        subprocess.run(["git", "config", "user.name", "github-actions[bot]"], cwd=clone_dir)
        subprocess.run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"], cwd=clone_dir)
        subprocess.run(["git", "add", "."], cwd=clone_dir)
        subprocess.run(["git", "commit", "-m", "Automated data update"], cwd=clone_dir)
        subprocess.run(["git", "branch", "-M", "main"], cwd=clone_dir)
        p_res = subprocess.run(["git", "push", "-u", "origin", "main"], cwd=clone_dir, capture_output=True, text=True)
        if p_res.returncode == 0:
            print(f"     Successfully pushed updates for {repo_name} to GitHub!")
            updated_any = True
        else:
            print(f"[!] Push failed: {p_res.stderr}")
    else:
        print(f"     All files for {repo_name} already up-to-date on GitHub. No push needed.")
        
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir, ignore_errors=True)
        
    return updated_any

def sync_ticker_huggingface(inst_name, repo_name, tracker):
    api = HfApi(token=HF_TOKEN)
    hf_user = get_hf_username()
    repo_id = f"{hf_user}/{repo_name}"
    
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=True)
    except Exception:
        pass
        
    hf_base_dir = os.path.join("tmp_hf", repo_name)
    os.makedirs(hf_base_dir, exist_ok=True)
    updated_any = False
    
    for tf in TIMEFRAMES_TO_TEST:
        if not TF_TO_HF.get(tf, False):
            continue  # Handled on GitHub
            
        print(f"  -> Timeframe {tf} (Target: HuggingFace)")
        tf_dir = os.path.join(hf_base_dir, tf)
        os.makedirs(tf_dir, exist_ok=True)
        
        oanda_tf = map_oanda_tf(tf)
        chunk_size = get_chunk_size(tf)
        
        bulk_download_and_split(
            inst_name, oanda_tf, None, None, chunk_size, tf_dir,
            file_prefix=repo_name, file_tf=tf
        )
        
        pod_files = sorted([f for f in os.listdir(tf_dir) if f.endswith(".pod")])
        csv_files = sorted([f for f in os.listdir(tf_dir) if f.endswith(".csv")])
        
        if csv_files:
            total_rows = 0
            first_candle_time = None
            last_candle_time = None
            
            for pf in pod_files:
                try:
                    with open(os.path.join(tf_dir, pf)) as p:
                        pdata = json.load(p)
                        total_rows += pdata.get("csv_row_count", 0)
                        if not first_candle_time and pdata.get("csv_first_candle"):
                            first_candle_time = pdata["csv_first_candle"].get("Time")
                        if pdata.get("csv_last_candle"):
                            last_candle_time = pdata["csv_last_candle"].get("Time")
                except Exception:
                    pass
                    
            print(f"     Uploading {tf} to Hugging Face {repo_id}...")
            api.upload_folder(
                folder_path=tf_dir,
                repo_id=repo_id,
                path_in_repo=tf,
                repo_type="dataset",
                commit_message="Automated data update"
            )
            
            tracker[inst_name][tf] = {
                "first_candle": first_candle_time,
                "last_candle": last_candle_time,
                "total_rows": total_rows,
                "chunks_count": len(csv_files),
                "platform": "HuggingFace",
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            updated_any = True
            print(f"     Successfully uploaded {tf} for {repo_name} to Hugging Face!")
            
    if os.path.exists(hf_base_dir):
        shutil.rmtree(hf_base_dir, ignore_errors=True)
        
    return updated_any

def main():
    if not GH_TOKEN or not HF_TOKEN:
        print("Missing GITHUB_TOKEN or HF_TOKEN. Make sure they are set in the environment.")
        return

    tracker_file = "tracker.json"
    if os.path.exists(tracker_file):
        with open(tracker_file, "r") as f:
            try:
                tracker = json.load(f)
            except Exception:
                tracker = {}
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
            
        has_gh = any(not TF_TO_HF.get(tf, False) for tf in TIMEFRAMES_TO_TEST)
        has_hf = any(TF_TO_HF.get(tf, False) for tf in TIMEFRAMES_TO_TEST)
        
        up_gh = False
        up_hf = False
        
        if has_gh:
            up_gh = sync_ticker_github(inst_name, repo_name, tracker)
        if has_hf:
            up_hf = sync_ticker_huggingface(inst_name, repo_name, tracker)
            
        if up_gh or up_hf:
            print(f"  -> Saving verified tracker metadata for {inst_name}")
            with open(tracker_file, "w") as f:
                json.dump(tracker, f, indent=2)
                
            if os.path.exists(".git"):
                subprocess.run(["git", "config", "user.name", "github-actions[bot]"])
                subprocess.run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"])
                subprocess.run(["git", "add", "tracker.json"])
                subprocess.run(["git", "commit", "-m", f"Update tracker for {inst_name}"])
                subprocess.run(["git", "push", "origin", "main"])

if __name__ == "__main__":
    main()
