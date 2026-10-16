import os
import sys
import time
import json
import shutil
import base64
import requests
import subprocess
from datetime import datetime, timezone

# Reconfigure stdout/stderr to UTF-8 if possible
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Disable Hugging Face tqdm progress bars to avoid log spam
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
try:
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
except Exception:
    pass

import gc
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from download_history import bulk_download_and_split, cleanup_memory, OANDA_URL, HEADERS, env_vars

GH_TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("PAT_TOKEN") or env_vars.get("PAT_TOKEN") or env_vars.get("GITHUB_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN") or env_vars.get("HF_TOKEN")
GH_USERNAME = os.environ.get("GH_USERNAME") or env_vars.get("GH_USERNAME", "")

TF_ALIASES = {
    "5S": "5S", "S5": "5S",
    "1": "1", "1M": "1", "M1": "1",
    "5": "5", "5M": "5", "M5": "5",
    "60": "60", "60M": "60", "H1": "60", "1H": "60",
    "D": "D", "DAILY": "D", "1D": "D"
}

tf_env = os.environ.get("TIMEFRAMES")
if tf_env and tf_env.strip():
    raw_tfs = [t.strip().upper() for t in tf_env.split(",") if t.strip()]
    TIMEFRAMES_TO_TEST = [TF_ALIASES.get(t, t) for t in raw_tfs]
else:
    TIMEFRAMES_TO_TEST = ["D", "60", "5", "1", "5S"]

# Platform routing:
# GitHub gets 1m ('1') and Daily ('D')
GITHUB_TIMEFRAMES = {"1", "D"}

# Hugging Face gets all timeframes: 5S, 1m ('1'), 5m ('5'), 60m ('60'), and Daily ('D')
HUGGINGFACE_TIMEFRAMES = {"5S", "1", "5", "60", "D"}

def get_all_instruments():
    account_id = os.environ.get('OANDA_ACCOUNT_ID') or env_vars.get('OANDA_ACCOUNT_ID')
    url = f"{OANDA_URL}/accounts/{account_id}/instruments"
    try:
        res = requests.get(url, headers=HEADERS, timeout=15)
        if res.status_code == 200:
            instruments = res.json().get("instruments", [])
            return {inst['name']: inst for inst in instruments}
    except Exception:
        pass
    return {}

def run_cmd(cmd, cwd=None):
    if isinstance(cmd, list):
        res = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    else:
        res = subprocess.run(cmd, shell=True, cwd=cwd, text=True, capture_output=True)
    if res.returncode != 0 and "already exists" not in res.stderr and "nothing to commit" not in res.stdout:
        print(f"  [!] Command failed: {cmd}\n{res.stderr.strip()}")
        return False
    return True

def get_gh_username():
    user = os.environ.get("GH_USERNAME") or env_vars.get("GH_USERNAME")
    if user:
        return user
    try:
        res = requests.get("https://api.github.com/user", headers={"Authorization": f"token {GH_TOKEN}"}, timeout=10)
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
                print(f"  * Converting GitHub repository {repo_name} to public...")
                requests.patch(check_url, headers=headers, json={"private": False})
            return True
    except Exception as e:
        print(f"  [!] Network error checking repo {repo_name}: {e}")
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
            print(f"  [!] GitHub repo creation rate-limited (403).")
            return False
        print(f"  [!] GitHub repo creation failed ({post_res.status_code}): {post_res.text[:120]}")
    except Exception as e:
        print(f"  [!] Exception creating repo {repo_name}: {e}")
    return False

def get_candle_dt(candle):
    if not candle:
        return None
    t = candle.get("Time") or candle.get("time")
    if t is None:
        return None
    if isinstance(t, (int, float)):
        return datetime.fromtimestamp(t, tz=timezone.utc)
    try:
        return datetime.strptime(str(t)[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:
        try:
            return datetime.strptime(str(t)[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            return None

def get_candle_time_str(candle):
    if not candle:
        return None
    if "Time" in candle and isinstance(candle["Time"], str):
        return candle["Time"]
    t = candle.get("time") or candle.get("Time")
    if isinstance(t, (int, float)):
        return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
    return str(t)

def clean_date_display(val):
    if not val:
        return "-"
    if isinstance(val, (int, float)):
        return datetime.fromtimestamp(val, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    s = str(val).strip()
    if len(s) >= 16 and "T" in s:
        return s[:16].replace("T", " ")
    return s[:10]

def get_repo_name(inst_name, inst_dict):
    display = inst_dict.get(inst_name, {}).get('displayName', '').replace('/', '').replace('\\', '').replace(' ', '-')
    name_no_us = inst_name.replace("_", "")
    return f"OANDA_{name_no_us}_{display}" if display else f"OANDA_{name_no_us}"

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

def format_duration(seconds):
    if seconds is None:
        return "-"
    if 0 < seconds < 0.05:
        return "<0.1s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds // 60)
    s = int(seconds % 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h = int(m // 60)
    m = int(m % 60)
    return f"{h}h {m:02d}m {s:02d}s"

# -----------------------------------------------------------------------------
# Summary Tracker
# -----------------------------------------------------------------------------
class SyncSummaryTracker:
    def __init__(self):
        self.records = []
        self.repo_links = {}     # ticker: {"GitHub": url, "HuggingFace": url}
        self.push_links = {}     # ticker: {"GitHub": [url], "HuggingFace": [url]}

    def add_record(self, ticker, platform, tf, chunks, rows, first_dt, last_dt, status, link, dl_time_s=None, push_time_s=None, is_reused=False):
        d_range = f"{clean_date_display(first_dt)} -> {clean_date_display(last_dt)}" if (first_dt and last_dt) else "-"
        # Update existing record if same ticker, platform, tf
        for r in self.records:
            if r["ticker"] == ticker and r["platform"] == platform and r["tf"] == tf:
                r.update({
                    "chunks": chunks or 0,
                    "rows": rows or 0,
                    "date_range": d_range,
                    "status": status,
                    "link": link,
                    "dl_time_s": dl_time_s,
                    "push_time_s": push_time_s,
                    "is_reused": is_reused
                })
                return
        self.records.append({
            "ticker": ticker,
            "platform": platform,
            "tf": tf,
            "chunks": chunks or 0,
            "rows": rows or 0,
            "date_range": d_range,
            "status": status,
            "link": link,
            "dl_time_s": dl_time_s,
            "push_time_s": push_time_s,
            "is_reused": is_reused
        })

    def record_repo(self, ticker, platform, url):
        if ticker not in self.repo_links:
            self.repo_links[ticker] = {}
        self.repo_links[ticker][platform] = url

    def record_push(self, ticker, platform, push_url):
        if ticker not in self.push_links:
            self.push_links[ticker] = {}
        if platform not in self.push_links[ticker]:
            self.push_links[ticker][platform] = []
        if push_url not in self.push_links[ticker][platform]:
            self.push_links[ticker][platform].append(push_url)

    def print_timeframe_summary(self, ticker, platform, tf, chunks, rows, first_dt, last_dt, status, link, dl_time_s=None, push_time_s=None, is_reused=False):
        pass

    def print_ticker_summary(self, ticker, repo_name, ticker_total_duration=None):
        ticker_records = [r for r in self.records if r["ticker"] == ticker]
        if not ticker_records:
            return

        tf_seen = set()
        total_chunks = 0
        total_rows = 0
        for r in ticker_records:
            tf = r.get("tf")
            if tf not in tf_seen:
                tf_seen.add(tf)
                total_chunks += r.get("chunks", 0)
                total_rows += r.get("rows", 0)

        dur_str = f"in {format_duration(ticker_total_duration)}" if ticker_total_duration else ""
        gh_url = self.repo_links.get(ticker, {}).get("GitHub", "")
        hf_url = self.repo_links.get(ticker, {}).get("HuggingFace", "")

        border = "=" * 120
        print(f"\n{border}")
        print(f"[COMPLETED] {ticker} ({repo_name}) {dur_str} | Total: {total_chunks:,d} chunks | {total_rows:,d} rows")
        if gh_url:
            print(f"  * GitHub:      {gh_url}")
        if hf_url:
            print(f"  * HuggingFace: {hf_url}")
        print(f"{border}\n")

# -----------------------------------------------------------------------------
# GitHub Sync
# -----------------------------------------------------------------------------
def check_github_tf_uptodate(gh_user, repo_name, tf):
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
        
        if last_candle:
            last_dt = get_candle_dt(last_candle)
            now_dt = datetime.now(timezone.utc)
            if last_dt and (now_dt - last_dt).days <= 2 and status in ["VERIFIED", "VERIFIED_PARTIAL"]:
                return True, {
                    "first_candle": get_candle_time_str(first_candle),
                    "last_candle": get_candle_time_str(last_candle),
                    "total_rows": pod_data_last.get("row_count") or pod_data_last.get("csv_row_count", 0),
                    "chunks_count": len(data_items),
                    "format": "parquet" if any(f['name'].endswith('.parquet') for f in data_items) else "csv"
                }
    except Exception:
        pass
    return False, None

# -----------------------------------------------------------------------------
# Hugging Face Remote Check & Helpers
# -----------------------------------------------------------------------------
def check_hf_tf_uptodate(api, repo_id, existing_hf_files, tf, tracker, inst_name):
    tf_prefix = f"{tf}/"
    tf_pods = sorted([f for f in existing_hf_files if f.startswith(tf_prefix) and f.endswith(".pod")])
    tf_data = [f for f in existing_hf_files if f.startswith(tf_prefix) and (f.endswith(".parquet") or f.endswith(".csv"))]
    
    if not tf_pods or not tf_data:
        return False, None
        
    latest_pod_file = tf_pods[-1]
    first_pod_file = tf_pods[0]
    try:
        local_pod_path = hf_hub_download(repo_id=repo_id, filename=latest_pod_file, repo_type="dataset", token=HF_TOKEN)
        with open(local_pod_path) as pf:
            pdata = json.load(pf)
            last_c = pdata.get("last_candle") or pdata.get("csv_last_candle")
            first_c = pdata.get("first_candle") or pdata.get("csv_first_candle")
            status = pdata.get("status")
            if last_c:
                last_dt = get_candle_dt(last_c)
                now_dt = datetime.now(timezone.utc)
                if last_dt and (now_dt - last_dt).days <= 2 and status in ["VERIFIED", "VERIFIED_PARTIAL"]:
                    existing_total = tracker.get(inst_name, {}).get("HuggingFace", {}).get(tf, {}).get("total_rows") or tracker.get(inst_name, {}).get("GitHub", {}).get(tf, {}).get("total_rows") or pdata.get("row_count") or pdata.get("csv_row_count", 0)
                    existing_first = tracker.get(inst_name, {}).get("HuggingFace", {}).get(tf, {}).get("first_candle")
                    if not existing_first and first_pod_file != latest_pod_file:
                        try:
                            first_pod_path = hf_hub_download(repo_id=repo_id, filename=first_pod_file, repo_type="dataset", token=HF_TOKEN)
                            with open(first_pod_path) as fpf:
                                first_pdata = json.load(fpf)
                                fc = first_pdata.get("first_candle") or first_pdata.get("csv_first_candle")
                                if fc:
                                    existing_first = get_candle_time_str(fc)
                        except Exception:
                            pass
                    first_candle_final = existing_first or get_candle_time_str(first_c)
                    return True, {
                        "first_candle": first_candle_final,
                        "last_candle": get_candle_time_str(last_c),
                        "total_rows": existing_total,
                        "chunks_count": len(tf_data),
                        "format": "parquet" if any(f.endswith(".parquet") for f in tf_data) else "csv"
                    }
    except Exception:
        pass
    return False, None

def ensure_hf_repo(api, repo_id):
    try:
        r_info = api.repo_info(repo_id=repo_id, repo_type="dataset")
        if r_info.private:
            api.update_repo_settings(repo_id=repo_id, private=False, repo_type="dataset")
        return True
    except Exception:
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=False)
            return True
        except Exception:
            return False

def push_timeframe_github(clone_dir, tf, gh_user, repo_name, gh_repo_url):
    t_start = time.time()
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=clone_dir, capture_output=True, text=True).stdout
    if not git_status.strip():
        return "Up-to-date", f"{gh_repo_url}/tree/main/{tf}", time.time() - t_start

    subprocess.run(["git", "config", "user.name", "github-actions[bot]"], cwd=clone_dir)
    subprocess.run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"], cwd=clone_dir)
    subprocess.run(["git", "add", tf], cwd=clone_dir)
    subprocess.run(["git", "commit", "-m", f"Automated {tf} parquet data update"], cwd=clone_dir, capture_output=True)
    subprocess.run(["git", "branch", "-M", "main"], cwd=clone_dir)
    commit_hash = subprocess.run(["git", "rev-parse", "HEAD"], cwd=clone_dir, capture_output=True, text=True).stdout.strip()
    p_res = subprocess.run(["git", "push", "--quiet", "-u", "origin", "main"], cwd=clone_dir, capture_output=True, text=True)
    if p_res.returncode != 0:
        subprocess.run(["git", "pull", "--quiet", "--rebase", "origin", "main"], cwd=clone_dir, capture_output=True)
        p_res = subprocess.run(["git", "push", "--quiet", "-u", "origin", "main"], cwd=clone_dir, capture_output=True, text=True)

    push_duration = time.time() - t_start
    if p_res.returncode == 0:
        return "Pushed", f"{gh_repo_url}/commit/{commit_hash}", push_duration
    else:
        return "Failed", f"{gh_repo_url}/tree/main/{tf}", push_duration

def push_timeframe_huggingface(api, tf_dir, hf_repo_id, tf, hf_repo_url):
    t_start = time.time()
    try:
        commit_info = api.upload_folder(
            folder_path=tf_dir,
            repo_id=hf_repo_id,
            path_in_repo=tf,
            repo_type="dataset",
            commit_message=f"Automated {tf} parquet data update"
        )
        push_duration = time.time() - t_start
        commit_oid = getattr(commit_info, 'oid', commit_info)
        commit_url = getattr(commit_info, 'commit_url', None) or f"{hf_repo_url}/commit/{commit_oid}"
        return "Pushed", commit_url, push_duration
    except Exception as e:
        push_duration = time.time() - t_start
        if "No files have been modified" in str(e):
            return "Up-to-date", f"{hf_repo_url}/tree/main/{tf}", push_duration
        return "Failed", f"{hf_repo_url}/tree/main/{tf}", push_duration

# -----------------------------------------------------------------------------
# Unified Timeframe Sync Engine
# -----------------------------------------------------------------------------
def sync_ticker(inst_name, repo_name, tracker, summary_tracker):
    gh_user = get_gh_username()
    hf_user = get_hf_username()
    api = HfApi(token=HF_TOKEN)
    hf_repo_id = f"{hf_user}/{repo_name}"
    gh_repo_url = f"https://github.com/{gh_user}/{repo_name}"
    hf_repo_url = f"https://huggingface.co/datasets/{hf_repo_id}"
    
    summary_tracker.record_repo(inst_name, "GitHub", gh_repo_url)
    summary_tracker.record_repo(inst_name, "HuggingFace", hf_repo_url)

    ensure_github_repo(repo_name, private=False)
    ensure_hf_repo(api, hf_repo_id)

    if "GitHub" not in tracker[inst_name]:
        tracker[inst_name]["GitHub"] = {}
    if "HuggingFace" not in tracker[inst_name]:
        tracker[inst_name]["HuggingFace"] = {}

    try:
        existing_hf_files = api.list_repo_files(repo_id=hf_repo_id, repo_type="dataset")
    except Exception:
        existing_hf_files = []

    clone_dir = f"gh_clone_{repo_name}"
    staging_base = os.path.join("tmp_staging", repo_name)
    os.makedirs(staging_base, exist_ok=True)
    
    updated_any = False

    for tf in TIMEFRAMES_TO_TEST:
        needs_gh = (tf in GITHUB_TIMEFRAMES)
        needs_hf = (tf in HUGGINGFACE_TIMEFRAMES)
        
        # 1. Fast remote checks (<1s)
        t_check_gh = time.time()
        is_up_gh, pod_gh = check_github_tf_uptodate(gh_user, repo_name, tf) if needs_gh else (True, None)
        check_gh_s = time.time() - t_check_gh

        t_check_hf = time.time()
        is_up_hf, pod_hf = check_hf_tf_uptodate(api, hf_repo_id, existing_hf_files, tf, tracker, inst_name) if needs_hf else (True, None)
        check_hf_s = time.time() - t_check_hf

        # 2. Both are already up-to-date!
        if is_up_gh and is_up_hf:
            meta = pod_gh or pod_hf or tracker.get(inst_name, {}).get("GitHub", {}).get(tf) or tracker.get(inst_name, {}).get("HuggingFace", {}).get(tf) or {}
            chunks = meta.get("chunks_count", 0)
            rows = tracker.get(inst_name, {}).get("GitHub", {}).get(tf, {}).get("total_rows") or tracker.get(inst_name, {}).get("HuggingFace", {}).get(tf, {}).get("total_rows") or meta.get("total_rows", 0)
            fc = meta.get("first_candle")
            lc = meta.get("last_candle")
            d_range = f"{clean_date_display(fc)} -> {clean_date_display(lc)}" if (fc and lc) else "-"

            gh_part = f"GitHub: Up-to-date ({format_duration(check_gh_s)})" if needs_gh else ""
            hf_part = f"HuggingFace: Up-to-date ({format_duration(check_hf_s)})" if needs_hf else ""
            status_inline = " | ".join([p for p in [gh_part, hf_part] if p])

            print(f"  * [{tf:<3}] {chunks:>3} chunks ({rows:>9,d} rows) | {d_range:<33} | {status_inline}")

            if needs_gh and pod_gh:
                tracker[inst_name]["GitHub"][tf] = {**pod_gh, "total_rows": rows, "platform": "GitHub", "last_updated": datetime.now(timezone.utc).isoformat()}
                summary_tracker.add_record(inst_name, "GitHub", tf, chunks, rows, fc, lc, "Up-to-date", f"{gh_repo_url}/tree/main/{tf}", dl_time_s=check_gh_s, push_time_s=0.0)
            if needs_hf and pod_hf:
                tracker[inst_name]["HuggingFace"][tf] = {**pod_hf, "total_rows": rows, "platform": "HuggingFace", "last_updated": datetime.now(timezone.utc).isoformat()}
                summary_tracker.add_record(inst_name, "HuggingFace", tf, chunks, rows, fc, lc, "Up-to-date", f"{hf_repo_url}/tree/main/{tf}", dl_time_s=check_hf_s, push_time_s=0.0)
            continue

        # 3. At least one platform needs data -> acquire files
        tf_staging = os.path.join(staging_base, tf)
        os.makedirs(tf_staging, exist_ok=True)
        t_acq = time.time()
        source = "OANDA"

        # Try reuse from GitHub (if GitHub is up-to-date and HF needs it)
        if is_up_gh and needs_gh and not is_up_hf:
            local_gh_dir = os.path.join(clone_dir, tf)
            if not os.path.exists(local_gh_dir) or not any(f.endswith(".parquet") for f in os.listdir(local_gh_dir)):
                run_cmd(f"git clone --quiet --depth 1 https://x-access-token:{GH_TOKEN}@github.com/{gh_user}/{repo_name}.git {clone_dir}")
            if os.path.exists(local_gh_dir) and any(f.endswith(".parquet") for f in os.listdir(local_gh_dir)):
                for f in os.listdir(local_gh_dir):
                    shutil.copy2(os.path.join(local_gh_dir, f), os.path.join(tf_staging, f))
                source = "GitHub"

        # Try reuse from Hugging Face (if HF is up-to-date and GitHub needs it)
        elif is_up_hf and needs_hf and not is_up_gh:
            try:
                hf_cache = snapshot_download(repo_id=hf_repo_id, repo_type="dataset", allow_patterns=f"{tf}/*", token=HF_TOKEN, max_workers=16)
                hf_tf = os.path.join(hf_cache, tf)
                if os.path.exists(hf_tf) and any(f.endswith(".parquet") for f in os.listdir(hf_tf)):
                    for f in os.listdir(hf_tf):
                        shutil.copy2(os.path.join(hf_tf, f), os.path.join(tf_staging, f))
                    source = "HuggingFace"
            except Exception:
                pass

        # Otherwise: download once from OANDA API
        if not any(f.endswith(".parquet") for f in os.listdir(tf_staging)):
            oanda_tf = map_oanda_tf(tf)
            chunk_size = get_chunk_size(tf)
            bulk_download_and_split(inst_name, oanda_tf, None, None, chunk_size, tf_staging, file_prefix=repo_name, file_tf=tf, is_standalone=False)
            source = "OANDA"

        acq_s = time.time() - t_acq

        # Read pod files to get accurate metadata
        pod_files = sorted([f for f in os.listdir(tf_staging) if f.endswith(".pod")])
        parquet_files = sorted([f for f in os.listdir(tf_staging) if f.endswith(".parquet")])
        csv_files = sorted([f for f in os.listdir(tf_staging) if f.endswith(".csv")])
        data_files = parquet_files if parquet_files else csv_files
        
        chunks = len(data_files)
        total_rows = 0
        first_candle = None
        last_candle = None
        for pf in pod_files:
            try:
                with open(os.path.join(tf_staging, pf)) as p:
                    pdata = json.load(p)
                    total_rows += pdata.get("row_count") or pdata.get("csv_row_count", 0)
                    fc = pdata.get("first_candle") or pdata.get("csv_first_candle")
                    lc = pdata.get("last_candle") or pdata.get("csv_last_candle")
                    if not first_candle and fc:
                        first_candle = get_candle_time_str(fc)
                    if lc:
                        last_candle = get_candle_time_str(lc)
            except Exception:
                pass
        d_range = f"{clean_date_display(first_candle)} -> {clean_date_display(last_candle)}" if (first_candle and last_candle) else "-"

        # 4. Push immediately for each platform!
        gh_status_str = ""
        hf_status_str = ""

        # --- GITHUB PUSH ---
        if needs_gh:
            if is_up_gh:
                gh_status_str = f"GitHub: Up-to-date ({format_duration(check_gh_s)})"
            else:
                if not os.path.exists(clone_dir):
                    run_cmd(f"git clone --quiet --depth 1 https://x-access-token:{GH_TOKEN}@github.com/{gh_user}/{repo_name}.git {clone_dir}")
                gh_tf_dir = os.path.join(clone_dir, tf)
                os.makedirs(gh_tf_dir, exist_ok=True)
                for f in os.listdir(tf_staging):
                    shutil.copy2(os.path.join(tf_staging, f), os.path.join(gh_tf_dir, f))
                
                status_gh, link_gh, push_gh_s = push_timeframe_github(clone_dir, tf, gh_user, repo_name, gh_repo_url)
                if status_gh == "Pushed":
                    updated_any = True
                    summary_tracker.record_push(inst_name, "GitHub", link_gh)
                    timing_lbl = f"Push {format_duration(push_gh_s)}" if source != "OANDA" else f"DL {format_duration(acq_s)} / Push {format_duration(push_gh_s)}"
                    gh_status_str = f"GitHub: Pushed in {format_duration(acq_s + push_gh_s)} ({timing_lbl}) -> {link_gh}"
                elif status_gh == "Up-to-date":
                    gh_status_str = f"GitHub: Up-to-date"
                else:
                    gh_status_str = f"GitHub: Push Failed ({format_duration(push_gh_s)})"

                summary_tracker.add_record(inst_name, "GitHub", tf, chunks, total_rows, first_candle, last_candle, status_gh, link_gh, dl_time_s=acq_s, push_time_s=push_gh_s, is_reused=(source != "OANDA"))
                tracker[inst_name]["GitHub"][tf] = {
                    "first_candle": first_candle, "last_candle": last_candle, "total_rows": total_rows,
                    "chunks_count": chunks, "format": "parquet" if parquet_files else "csv", "platform": "GitHub", "last_updated": datetime.now(timezone.utc).isoformat()
                }

        # --- HUGGING FACE PUSH ---
        if needs_hf:
            if is_up_hf:
                hf_status_str = f"HuggingFace: Up-to-date ({format_duration(check_hf_s)})"
            else:
                status_hf, link_hf, push_hf_s = push_timeframe_huggingface(api, tf_staging, hf_repo_id, tf, hf_repo_url)
                if status_hf == "Pushed":
                    updated_any = True
                    summary_tracker.record_push(inst_name, "HuggingFace", link_hf)
                    timing_lbl = f"Upload {format_duration(push_hf_s)}" if source != "OANDA" else f"DL {format_duration(acq_s)} / Up {format_duration(push_hf_s)}"
                    hf_status_str = f"HuggingFace: Pushed in {format_duration(acq_s + push_hf_s)} ({timing_lbl}) -> {link_hf}"
                elif status_hf == "Up-to-date":
                    hf_status_str = f"HuggingFace: Up-to-date"
                else:
                    hf_status_str = f"HuggingFace: Upload Failed ({format_duration(push_hf_s)})"

                summary_tracker.add_record(inst_name, "HuggingFace", tf, chunks, total_rows, first_candle, last_candle, status_hf, link_hf, dl_time_s=acq_s, push_time_s=push_hf_s, is_reused=(source != "OANDA"))
                tracker[inst_name]["HuggingFace"][tf] = {
                    "first_candle": first_candle, "last_candle": last_candle, "total_rows": total_rows,
                    "chunks_count": chunks, "format": "parquet" if parquet_files else "csv", "platform": "HuggingFace", "last_updated": datetime.now(timezone.utc).isoformat()
                }

        # 5. Print clean inline summary for this timeframe!
        status_line = " | ".join([p for p in [gh_status_str, hf_status_str] if p])
        print(f"  * [{tf:<3}] {chunks:>3} chunks ({total_rows:>9,d} rows) | {d_range:<33} | {status_line}")

        # 6. Save tracker immediately and clean up tf_staging
        try:
            with open("tracker.json", "w") as f:
                json.dump(tracker, f, indent=2)
        except Exception:
            pass
        if os.path.exists(tf_staging):
            shutil.rmtree(tf_staging, ignore_errors=True)

    return updated_any

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------
def cleanup_ticker_resources(repo_name, hf_user):
    """
    Frees disk storage and RAM immediately after a ticker is processed:
    1. Deletes the local git clone directory for this ticker.
    2. Deletes the Hugging Face staging folder for this ticker.
    3. Cleans up cached hub snapshots in ~/.cache/huggingface/hub for this ticker.
    4. Clears in-memory market caches and runs garbage collection.
    """
    # 1. Clean GitHub clone directory
    clone_dir = f"gh_clone_{repo_name}"
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir, ignore_errors=True)

    # 2. Clean temporary staging directory
    staging_dir = os.path.join("tmp_staging", repo_name)
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir, ignore_errors=True)
    if os.path.exists("tmp_staging"):
        try:
            if not os.listdir("tmp_staging"):
                shutil.rmtree("tmp_staging", ignore_errors=True)
        except Exception:
            pass

    if os.path.exists("tmp_hf"):
        shutil.rmtree("tmp_hf", ignore_errors=True)

    # 3. Clean Hugging Face cached snapshot for this repo to save disk
    try:
        from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
        cache_repo_dir = os.path.join(HUGGINGFACE_HUB_CACHE, f"datasets--{hf_user}--{repo_name}")
        if os.path.exists(cache_repo_dir):
            shutil.rmtree(cache_repo_dir, ignore_errors=True)
    except Exception:
        pass

    # 4. Clear memory caches and run garbage collector
    try:
        cleanup_memory()
    except Exception:
        pass
    gc.collect()

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
        
    summary_tracker = SyncSummaryTracker()

    for inst_name, inst_info in inst_dict.items():
        repo_name = get_repo_name(inst_name, inst_dict)
        ticker_start_time = time.time()
        print(f"\n==================================================")
        print(f"Processing: {inst_name} -> {repo_name}")
        print(f"==================================================")
        
        try:
            if inst_name not in tracker:
                tracker[inst_name] = {"GitHub": {}, "HuggingFace": {}}
            elif not isinstance(tracker[inst_name], dict):
                tracker[inst_name] = {"GitHub": {}, "HuggingFace": {}}
            else:
                if "GitHub" not in tracker[inst_name]:
                    tracker[inst_name]["GitHub"] = {}
                if "HuggingFace" not in tracker[inst_name]:
                    tracker[inst_name]["HuggingFace"] = {}
                
            updated_any = sync_ticker(inst_name, repo_name, tracker, summary_tracker)
                
            # Update tracker after processing each ticker
            with open(tracker_file, "w") as f:
                json.dump(tracker, f, indent=2)
                
            if os.path.exists(".git") and updated_any:
                subprocess.run(["git", "config", "user.name", "github-actions[bot]"], capture_output=True)
                subprocess.run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"], capture_output=True)
                subprocess.run(["git", "add", "tracker.json"], capture_output=True)
                commit_res = subprocess.run(["git", "commit", "-m", f"Update tracker for {inst_name}"], capture_output=True, text=True)
                if commit_res.returncode == 0:
                    subprocess.run(["git", "pull", "--quiet", "--rebase", "origin", "main"], capture_output=True)
                    subprocess.run(["git", "push", "--quiet", "origin", "main"], capture_output=True)

            # Print concise completion summary for this ticker
            ticker_total_duration = time.time() - ticker_start_time
            summary_tracker.print_ticker_summary(inst_name, repo_name, ticker_total_duration=ticker_total_duration)
        except Exception as ticker_err:
            print(f"  [!] Error processing ticker {inst_name}: {ticker_err}")
            ticker_total_duration = time.time() - ticker_start_time
            summary_tracker.print_ticker_summary(inst_name, repo_name, ticker_total_duration=ticker_total_duration)
        finally:
            hf_user = get_hf_username()
            cleanup_ticker_resources(repo_name, hf_user)
            print(f"  * Cleaned up local storage & memory for {inst_name}")

if __name__ == "__main__":
    main()
