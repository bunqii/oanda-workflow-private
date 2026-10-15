import os
import time
import json
import shutil
import base64
import requests
import subprocess
from datetime import datetime, timezone
from huggingface_hub import HfApi, hf_hub_download
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

def ensure_github_repo(repo_name, private=False):
    """
    Checks if repository exists first via GET to prevent hitting
    GitHub's secondary rate limits on repository creation.
    Ensures the repository is PUBLIC.
    """
    gh_user = get_gh_username()
    headers = {
        "Authorization": f"token {GH_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    check_url = f"https://api.github.com/repos/{gh_user}/{repo_name}"
    try:
        res = requests.get(check_url, headers=headers, timeout=10)
        if res.status_code == 200:
            repo_info = res.json()
            if repo_info.get("private") and not private:
                print(f"     Converting GitHub repository {repo_name} to public...")
                requests.patch(check_url, headers=headers, json={"private": False})
            return True  # Exists!
    except Exception as e:
        print(f"[!] Network error checking repo {repo_name}: {e}")
        return False

    payload = {
        "name": repo_name,
        "private": private,
        "auto_init": True
    }
    try:
        post_res = requests.post("https://api.github.com/user/repos", headers=headers, json=payload, timeout=15)
        if post_res.status_code in [201, 422]:
            return True
        if post_res.status_code == 403:
            print(f"[!] GitHub repo creation rate-limited (403). Secondary limit reached.")
            return False
        print(f"[!] GitHub repo creation failed ({post_res.status_code}): {post_res.text[:120]}")
    except Exception as e:
        print(f"[!] Exception creating repo {repo_name}: {e}")
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

def check_github_tf_uptodate(gh_user, repo_name, tf):
    """
    Checks if a timeframe is already up-to-date on GitHub by querying the pod file
    metadata directly through GitHub Contents API (without doing a heavy git clone).
    """
    headers = {
        "Authorization": f"token {GH_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    url = f"https://api.github.com/repos/{gh_user}/{repo_name}/contents/{tf}"
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200:
            return False, None
        items = res.json()
        if not isinstance(items, list) or not items:
            return False, None

        pod_items = sorted([item for item in items if item['name'].endswith('.pod')], key=lambda x: x['name'])
        data_items = [item for item in items if item['name'].endswith('.parquet') or item['name'].endswith('.csv')]
        
        if not pod_items or not data_items:
            return False, None

        # Fetch first pod for inception candle and latest pod for current candle
        first_pod_item = pod_items[0]
        latest_pod_item = pod_items[-1]
        
        def get_pod_json(item):
            if "content" in item and item["content"]:
                return json.loads(base64.b64decode(item["content"]).decode('utf-8'))
            r = requests.get(item["url"], headers=headers, timeout=10)
            if r.status_code == 200:
                return json.loads(base64.b64decode(r.json()["content"]).decode('utf-8'))
            return {}

        pod_data_last = get_pod_json(latest_pod_item)
        pod_data_first = get_pod_json(first_pod_item) if first_pod_item != latest_pod_item else pod_data_last
        
        status = pod_data_last.get("status")
        last_candle = pod_data_last.get("last_candle") or pod_data_last.get("csv_last_candle")
        first_candle = pod_data_first.get("first_candle") or pod_data_first.get("csv_first_candle")
        
        # Check if last candle is up to today or yesterday
        if last_candle and "Time" in last_candle:
            last_dt = datetime.strptime(last_candle["Time"][:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            now_dt = datetime.now(timezone.utc)
            # If latest candle is within 2 days (weekend / market close aware), chunk is up-to-date
            if (now_dt - last_dt).days <= 2 and status in ["VERIFIED", "VERIFIED_PARTIAL"]:
                return True, {
                    "first_candle": first_candle.get("Time") if first_candle else None,
                    "last_candle": last_candle.get("Time"),
                    "total_rows": pod_data_last.get("row_count") or pod_data_last.get("csv_row_count", 0),
                    "chunks_count": len(data_items),
                    "format": "parquet" if any(f['name'].endswith('.parquet') for f in data_items) else "csv"
                }
    except Exception:
        pass
    return False, None

def sync_ticker_github(inst_name, repo_name, tracker):
    gh_user = get_gh_username()
    if not ensure_github_repo(repo_name, private=False):
        print(f"     [!] Could not access or create repository {repo_name}. Skipping for now.")
        return False
    
    # Initialize separate GitHub tracker for this ticker
    if "GitHub" not in tracker[inst_name]:
        tracker[inst_name]["GitHub"] = {}

    timeframes_to_download = []
    
    # 1. Quick remote check: check which timeframes are already verified on GitHub
    for tf in TIMEFRAMES_TO_TEST:
        if TF_TO_HF.get(tf, False):
            continue  # Handled on Hugging Face
            
        is_up, pod_meta = check_github_tf_uptodate(gh_user, repo_name, tf)
        if is_up and pod_meta:
            print(f"  -> Timeframe {tf} (GitHub): Already up-to-date ({pod_meta['chunks_count']} chunks, last: {pod_meta['last_candle']}). Smart skip.")
            existing_total = tracker.get(inst_name, {}).get("GitHub", {}).get(tf, {}).get("total_rows")
            if existing_total and existing_total > pod_meta.get("total_rows", 0):
                pod_meta["total_rows"] = existing_total
            tracker[inst_name]["GitHub"][tf] = {
                **pod_meta,
                "platform": "GitHub",
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
        else:
            timeframes_to_download.append(tf)

    if not timeframes_to_download:
        return False  # All GitHub timeframes already up-to-date!

    # 2. Shallow clone (--depth 1) only when there are chunks to update
    clone_dir = f"gh_clone_{repo_name}"
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir, ignore_errors=True)
        
    print(f"     Cloning repository {gh_user}/{repo_name} (shallow --depth 1)...")
    clone_ok = run_cmd(f"git clone --depth 1 https://x-access-token:{GH_TOKEN}@github.com/{gh_user}/{repo_name}.git {clone_dir}")
    if not clone_ok:
        print(f"[!] Could not clone {repo_name}. Skipping.")
        return False
        
    updated_any = False
    
    for tf in timeframes_to_download:
        print(f"  -> Timeframe {tf} (Target: GitHub)")
        tf_dir = os.path.join(clone_dir, tf)
        os.makedirs(tf_dir, exist_ok=True)
        
        oanda_tf = map_oanda_tf(tf)
        chunk_size = get_chunk_size(tf)
        
        # Download directly into repo directory
        bulk_download_and_split(
            inst_name, oanda_tf, None, None, chunk_size, tf_dir,
            file_prefix=repo_name, file_tf=tf
        )
        
        pod_files = sorted([f for f in os.listdir(tf_dir) if f.endswith(".pod")])
        parquet_files = sorted([f for f in os.listdir(tf_dir) if f.endswith(".parquet")])
        csv_files = sorted([f for f in os.listdir(tf_dir) if f.endswith(".csv")])
        data_files = parquet_files if parquet_files else csv_files
        
        if data_files:
            total_rows = 0
            first_candle_time = None
            last_candle_time = None
            
            for pf in pod_files:
                try:
                    with open(os.path.join(tf_dir, pf)) as p:
                        pdata = json.load(p)
                        total_rows += pdata.get("row_count") or pdata.get("csv_row_count", 0)
                        fc = pdata.get("first_candle") or pdata.get("csv_first_candle")
                        lc = pdata.get("last_candle") or pdata.get("csv_last_candle")
                        if not first_candle_time and fc:
                            first_candle_time = fc.get("Time")
                        if lc:
                            last_candle_time = lc.get("Time")
                except Exception:
                    pass
                    
            tracker[inst_name]["GitHub"][tf] = {
                "first_candle": first_candle_time,
                "last_candle": last_candle_time,
                "total_rows": total_rows,
                "chunks_count": len(data_files),
                "format": "parquet" if parquet_files else "csv",
                "platform": "GitHub",
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            print(f"     Status: {len(data_files)} chunks, {total_rows} total rows ({first_candle_time} -> {last_candle_time})")

    # Check if files changed in the repository
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=clone_dir, capture_output=True, text=True).stdout
    if git_status.strip():
        subprocess.run(["git", "config", "user.name", "github-actions[bot]"], cwd=clone_dir)
        subprocess.run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"], cwd=clone_dir)
        subprocess.run(["git", "add", "-A"], cwd=clone_dir)
        subprocess.run(["git", "commit", "-m", "Automated parquet data update"], cwd=clone_dir)
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
    
    # 1. Ensure public dataset repo exists on Hugging Face
    try:
        r_info = api.repo_info(repo_id=repo_id, repo_type="dataset")
        if r_info.private:
            print(f"     Converting Hugging Face dataset {repo_id} to public...")
            api.update_repo_settings(repo_id=repo_id, private=False, repo_type="dataset")
    except Exception:
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=False)
        except Exception as e:
            print(f"[!] Hugging Face create_repo info: {e}")

    # Initialize separate Hugging Face tracker for this ticker
    if "HuggingFace" not in tracker[inst_name]:
        tracker[inst_name]["HuggingFace"] = {}

    try:
        existing_hf_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception:
        existing_hf_files = []

    hf_base_dir = os.path.join("tmp_hf", repo_name)
    os.makedirs(hf_base_dir, exist_ok=True)
    updated_any = False
    
    for tf in TIMEFRAMES_TO_TEST:
        if not TF_TO_HF.get(tf, False):
            continue  # Handled on GitHub
            
        print(f"  -> Timeframe {tf} (Target: HuggingFace)")
        
        # Check if already present on Hugging Face and up-to-date
        tf_prefix = f"{tf}/"
        tf_pods = sorted([f for f in existing_hf_files if f.startswith(tf_prefix) and f.endswith(".pod")])
        tf_data = [f for f in existing_hf_files if f.startswith(tf_prefix) and (f.endswith(".parquet") or f.endswith(".csv"))]
        
        skip_download = False
        if tf_pods and tf_data:
            latest_pod_file = tf_pods[-1]
            try:
                # Download ONLY the single 1KB pod file to check freshness
                local_pod_path = hf_hub_download(repo_id=repo_id, filename=latest_pod_file, repo_type="dataset", token=HF_TOKEN)
                with open(local_pod_path) as pf:
                    pdata = json.load(pf)
                    last_c = pdata.get("last_candle") or pdata.get("csv_last_candle")
                    first_c = pdata.get("first_candle") or pdata.get("csv_first_candle")
                    status = pdata.get("status")
                    if last_c and "Time" in last_c:
                        last_dt = datetime.strptime(last_c["Time"][:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                        now_dt = datetime.now(timezone.utc)
                        if (now_dt - last_dt).days <= 2 and status in ["VERIFIED", "VERIFIED_PARTIAL"]:
                            print(f"     Hugging Face {tf}: Already up-to-date ({len(tf_data)} files, last: {last_c['Time']}). Skipping.")
                            existing_total = tracker.get(inst_name, {}).get("HuggingFace", {}).get(tf, {}).get("total_rows")
                            tracker[inst_name]["HuggingFace"][tf] = {
                                "first_candle": first_c.get("Time") if first_c else None,
                                "last_candle": last_c.get("Time"),
                                "total_rows": existing_total if (existing_total and existing_total > (pdata.get("row_count") or pdata.get("csv_row_count", 0))) else (pdata.get("row_count") or pdata.get("csv_row_count", 0)),
                                "chunks_count": len(tf_data),
                                "format": "parquet" if any(f.endswith(".parquet") for f in tf_data) else "csv",
                                "platform": "HuggingFace",
                                "last_updated": datetime.now(timezone.utc).isoformat()
                            }
                            skip_download = True
            except Exception:
                pass

        if skip_download:
            continue

        tf_dir = os.path.join(hf_base_dir, tf)
        os.makedirs(tf_dir, exist_ok=True)
        
        oanda_tf = map_oanda_tf(tf)
        chunk_size = get_chunk_size(tf)
        
        bulk_download_and_split(
            inst_name, oanda_tf, None, None, chunk_size, tf_dir,
            file_prefix=repo_name, file_tf=tf
        )
        
        pod_files = sorted([f for f in os.listdir(tf_dir) if f.endswith(".pod")])
        parquet_files = sorted([f for f in os.listdir(tf_dir) if f.endswith(".parquet")])
        csv_files = sorted([f for f in os.listdir(tf_dir) if f.endswith(".csv")])
        data_files = parquet_files if parquet_files else csv_files
        
        if data_files:
            total_rows = 0
            first_candle_time = None
            last_candle_time = None
            
            for pf in pod_files:
                try:
                    with open(os.path.join(tf_dir, pf)) as p:
                        pdata = json.load(p)
                        total_rows += pdata.get("row_count") or pdata.get("csv_row_count", 0)
                        fc = pdata.get("first_candle") or pdata.get("csv_first_candle")
                        lc = pdata.get("last_candle") or pdata.get("csv_last_candle")
                        if not first_candle_time and fc:
                            first_candle_time = fc.get("Time")
                        if lc:
                            last_candle_time = lc.get("Time")
                except Exception:
                    pass
                    
            print(f"     Uploading {tf} to Hugging Face {repo_id}...")
            api.upload_folder(
                folder_path=tf_dir,
                repo_id=repo_id,
                path_in_repo=tf,
                repo_type="dataset",
                commit_message="Automated parquet data update"
            )
            
            tracker[inst_name]["HuggingFace"][tf] = {
                "first_candle": first_candle_time,
                "last_candle": last_candle_time,
                "total_rows": total_rows,
                "chunks_count": len(data_files),
                "format": "parquet" if parquet_files else "csv",
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
            tracker[inst_name] = {"GitHub": {}, "HuggingFace": {}}
        elif not isinstance(tracker[inst_name], dict):
            tracker[inst_name] = {"GitHub": {}, "HuggingFace": {}}
        else:
            if "GitHub" not in tracker[inst_name]:
                tracker[inst_name]["GitHub"] = {}
            if "HuggingFace" not in tracker[inst_name]:
                tracker[inst_name]["HuggingFace"] = {}
            
        has_gh = any(not TF_TO_HF.get(tf, False) for tf in TIMEFRAMES_TO_TEST)
        has_hf = any(TF_TO_HF.get(tf, False) for tf in TIMEFRAMES_TO_TEST)
        
        up_gh = False
        up_hf = False
        
        if has_gh:
            up_gh = sync_ticker_github(inst_name, repo_name, tracker)
        if has_hf:
            up_hf = sync_ticker_huggingface(inst_name, repo_name, tracker)
            
        # Update tracker after processing each ticker
        print(f"  -> Saving verified tracker metadata for {inst_name}")
        with open(tracker_file, "w") as f:
            json.dump(tracker, f, indent=2)
            
        if os.path.exists(".git") and (up_gh or up_hf):
            subprocess.run(["git", "config", "user.name", "github-actions[bot]"])
            subprocess.run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"])
            subprocess.run(["git", "add", "tracker.json"])
            commit_res = subprocess.run(["git", "commit", "-m", f"Update tracker for {inst_name}"], capture_output=True, text=True)
            if commit_res.returncode == 0:
                subprocess.run(["git", "pull", "--rebase", "origin", "main"])
                subprocess.run(["git", "push", "origin", "main"])

if __name__ == "__main__":
    main()
