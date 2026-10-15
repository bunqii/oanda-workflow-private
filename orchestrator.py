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
from huggingface_hub import HfApi, hf_hub_download
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
        d_range = f"{clean_date_display(first_dt)} -> {clean_date_display(last_dt)}" if (first_dt and last_dt) else "-"
        border = "+" + "-" * 118 + "+"
        
        # Build timing line
        if status == "Up-to-date":
            timing_line = f"Verified remote in {format_duration(dl_time_s)} | Status: Up-to-date (no upload needed)" if dl_time_s else "Verified remote | Status: Up-to-date (no upload needed)"
        elif is_reused:
            tot = (dl_time_s or 0.0) + (push_time_s or 0.0)
            timing_line = f"Download: Reused from GitHub ({format_duration(dl_time_s)}) | Upload: {format_duration(push_time_s)} | Total: {format_duration(tot)}"
        else:
            tot = (dl_time_s or 0.0) + (push_time_s or 0.0)
            dl_part = f"Download: {format_duration(dl_time_s)}" if dl_time_s is not None else "Download: -"
            push_part = f"Push: {format_duration(push_time_s)}" if push_time_s is not None else "Push: -"
            timing_line = f"{dl_part} | {push_part} | Total: {format_duration(tot)}"

        chunks_val = chunks or 0
        rows_val = rows or 0

        print(f"\n{border}")
        print(f"| [TIMEFRAME SYNC] {ticker} | {platform} (TF: {tf})")
        print(f"|   * Data:    {chunks_val:,d} chunk(s) | {rows_val:,d} rows | Range: {d_range}")
        print(f"|   * Timing:  {timing_line}")
        print(f"|   * Status:  {status} -> {link}")
        print(f"{border}\n")

    def print_ticker_summary(self, ticker, repo_name, ticker_total_duration=None):
        ticker_records = [r for r in self.records if r["ticker"] == ticker]
        if not ticker_records:
            return

        border = "=" * 136
        sep = "-" * 136
        title = f"DATA SYNC SUMMARY: {ticker} ({repo_name})"

        print(f"\n{border}")
        print(f"{title.center(136)}")
        print(border)
        print(f"{'Platform':<11} {'TF':<4} {'Chunks':>8} {'Total Rows':>14}   {'Date Range (UTC)':<37} {'Timing (DL / Push / Total)':<28} {'Status':<11} {'Direct Link'}")
        print(sep)

        tf_order = {"D": 1, "60": 2, "5": 3, "1": 4, "5S": 5}
        sorted_records = sorted(
            ticker_records,
            key=lambda r: (0 if r["platform"] == "GitHub" else 1, tf_order.get(r["tf"], 99))
        )

        for r in sorted_records:
            if r.get("status") == "Up-to-date":
                dl = r.get("dl_time_s")
                t_str = f"Verified ({format_duration(dl)})" if dl else "Verified (Up-to-date)"
            elif r.get("is_reused"):
                push = format_duration(r.get("push_time_s"))
                tot = format_duration((r.get("dl_time_s") or 0.0) + (r.get("push_time_s") or 0.0))
                t_str = f"Reused / {push} ({tot})"
            else:
                dl = format_duration(r.get("dl_time_s"))
                push = format_duration(r.get("push_time_s"))
                tot = format_duration((r.get("dl_time_s") or 0.0) + (r.get("push_time_s") or 0.0))
                t_str = f"{dl} / {push} ({tot})"

            print(f"{r['platform']:<11} {r['tf']:<4} {r['chunks']:>8,d} {r['rows']:>14,d}   {r['date_range']:<37} {t_str:<28} {r['status']:<11} {r['link']}")

        print(sep)
        total_chunks = sum(r.get("chunks", 0) for r in sorted_records)
        total_rows = sum(r.get("rows", 0) for r in sorted_records)

        print("Ticker Statistics:")
        if ticker_total_duration is not None:
            print(f"  * Total Ticker Runtime:      {format_duration(ticker_total_duration)}")
        print(f"  * Total Synchronized Chunks: {total_chunks:,d} chunks")
        print(f"  * Total Synchronized Rows:   {total_rows:,d} rows")

        gh_url = self.repo_links.get(ticker, {}).get("GitHub", "")
        hf_url = self.repo_links.get(ticker, {}).get("HuggingFace", "")
        gh_pushes = self.push_links.get(ticker, {}).get("GitHub", [])
        hf_pushes = self.push_links.get(ticker, {}).get("HuggingFace", [])

        print("Repository Links & Latest Pushes:")
        if gh_url:
            print(f"  * GitHub Repository:         {gh_url}")
            if gh_pushes:
                for p in gh_pushes:
                    print(f"    --> Pushed Commit:         {p}")
            else:
                print(f"    --> Push Status:           Up-to-date (No new commit)")
        if hf_url:
            print(f"  * HuggingFace Dataset:       {hf_url}")
            if hf_pushes:
                for p in hf_pushes:
                    print(f"    --> Pushed Commit:         {p}")
            else:
                print(f"    --> Push Status:           Up-to-date (No new commit)")
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

def sync_ticker_github(inst_name, repo_name, tracker, summary_tracker):
    gh_user = get_gh_username()
    repo_url = f"https://github.com/{gh_user}/{repo_name}"
    summary_tracker.record_repo(inst_name, "GitHub", repo_url)

    if not ensure_github_repo(repo_name, private=False):
        print(f"  [!] Could not access or create GitHub repository {repo_name}. Skipping GitHub sync.")
        return False
    
    if "GitHub" not in tracker[inst_name]:
        tracker[inst_name]["GitHub"] = {}

    timeframes_to_download = []
    
    # Quick remote check
    for tf in TIMEFRAMES_TO_TEST:
        if tf not in GITHUB_TIMEFRAMES:
            continue
            
        t_check_start = time.time()
        is_up, pod_meta = check_github_tf_uptodate(gh_user, repo_name, tf)
        check_duration = time.time() - t_check_start
        if is_up and pod_meta:
            existing_total = tracker.get(inst_name, {}).get("GitHub", {}).get(tf, {}).get("total_rows") or tracker.get(inst_name, {}).get("HuggingFace", {}).get(tf, {}).get("total_rows") or 0
            if existing_total < pod_meta.get("total_rows", 0):
                existing_total = pod_meta.get("total_rows", 0)
            print(f"  * [GitHub] {tf}: Up-to-date ({pod_meta['chunks_count']} chunks, {existing_total:,} rows)")
            tracker[inst_name]["GitHub"][tf] = {
                **pod_meta,
                "total_rows": existing_total,
                "platform": "GitHub",
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            tf_link = f"{repo_url}/tree/main/{tf}"
            summary_tracker.add_record(
                ticker=inst_name,
                platform="GitHub",
                tf=tf,
                chunks=pod_meta['chunks_count'],
                rows=existing_total,
                first_dt=pod_meta.get('first_candle'),
                last_dt=pod_meta.get('last_candle'),
                status="Up-to-date",
                link=tf_link,
                dl_time_s=check_duration,
                push_time_s=0.0
            )
            summary_tracker.print_timeframe_summary(
                ticker=inst_name,
                platform="GitHub",
                tf=tf,
                chunks=pod_meta['chunks_count'],
                rows=existing_total,
                first_dt=pod_meta.get('first_candle'),
                last_dt=pod_meta.get('last_candle'),
                status="Up-to-date",
                link=tf_link,
                dl_time_s=check_duration,
                push_time_s=0.0
            )
        else:
            timeframes_to_download.append(tf)

    if not timeframes_to_download:
        return False

    clone_dir = f"gh_clone_{repo_name}"
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir, ignore_errors=True)
        
    print(f"  * [GitHub] Fetching repository {gh_user}/{repo_name}...")
    clone_ok = run_cmd(f"git clone --quiet --depth 1 https://x-access-token:{GH_TOKEN}@github.com/{gh_user}/{repo_name}.git {clone_dir}")
    if not clone_ok:
        print(f"  [!] Could not clone {repo_name}. Skipping GitHub sync.")
        return False
        
    updated_any = False

    for tf in timeframes_to_download:
        print(f"  * [GitHub] Syncing timeframe {tf}...")
        tf_dir = os.path.join(clone_dir, tf)
        os.makedirs(tf_dir, exist_ok=True)
        
        oanda_tf = map_oanda_tf(tf)
        chunk_size = get_chunk_size(tf)
        
        t_dl_start = time.time()
        bulk_download_and_split(
            inst_name, oanda_tf, None, None, chunk_size, tf_dir,
            file_prefix=repo_name, file_tf=tf
        )
        dl_time_s = time.time() - t_dl_start
        
        pod_files = sorted([f for f in os.listdir(tf_dir) if f.endswith(".pod")])
        parquet_files = sorted([f for f in os.listdir(tf_dir) if f.endswith(".parquet")])
        csv_files = sorted([f for f in os.listdir(tf_dir) if f.endswith(".csv")])
        data_files = parquet_files if parquet_files else csv_files
        
        total_rows = 0
        first_candle_time = None
        last_candle_time = None
        
        if data_files:
            for pf in pod_files:
                try:
                    with open(os.path.join(tf_dir, pf)) as p:
                        pdata = json.load(p)
                        total_rows += pdata.get("row_count") or pdata.get("csv_row_count", 0)
                        fc = pdata.get("first_candle") or pdata.get("csv_first_candle")
                        lc = pdata.get("last_candle") or pdata.get("csv_last_candle")
                        if not first_candle_time and fc:
                            first_candle_time = get_candle_time_str(fc)
                        if lc:
                            last_candle_time = get_candle_time_str(lc)
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

        # Check if files changed for this timeframe
        git_status = subprocess.run(["git", "status", "--porcelain"], cwd=clone_dir, capture_output=True, text=True).stdout
        t_push_start = time.time()
        push_time_s = 0.0
        commit_url = f"{repo_url}/tree/main/{tf}"
        status_str = "Up-to-date"
        
        if git_status.strip():
            subprocess.run(["git", "config", "user.name", "github-actions[bot]"], cwd=clone_dir)
            subprocess.run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"], cwd=clone_dir)
            subprocess.run(["git", "add", "-A"], cwd=clone_dir)
            subprocess.run(["git", "commit", "-m", f"Automated {tf} parquet data update"], cwd=clone_dir, capture_output=True)
            subprocess.run(["git", "branch", "-M", "main"], cwd=clone_dir)
            commit_hash = subprocess.run(["git", "rev-parse", "HEAD"], cwd=clone_dir, capture_output=True, text=True).stdout.strip()
            p_res = subprocess.run(["git", "push", "--quiet", "-u", "origin", "main"], cwd=clone_dir, capture_output=True, text=True)
            push_time_s = time.time() - t_push_start
            
            if p_res.returncode != 0:
                subprocess.run(["git", "pull", "--quiet", "--rebase", "origin", "main"], cwd=clone_dir, capture_output=True)
                p_res = subprocess.run(["git", "push", "--quiet", "-u", "origin", "main"], cwd=clone_dir, capture_output=True, text=True)
                push_time_s = time.time() - t_push_start
                
            if p_res.returncode == 0:
                commit_url = f"{repo_url}/commit/{commit_hash}"
                print(f"  [OK] [GitHub] Pushed {tf} updates to GitHub: {commit_url}")
                summary_tracker.record_push(inst_name, "GitHub", commit_url)
                status_str = "Pushed"
                updated_any = True
            else:
                print(f"  [!] GitHub push failed for {tf}: {p_res.stderr.strip()[:150]}")
                status_str = "Failed"
        else:
            print(f"  * [GitHub] {tf} files already up-to-date on GitHub.")
            status_str = "Up-to-date"

        summary_tracker.add_record(
            ticker=inst_name,
            platform="GitHub",
            tf=tf,
            chunks=len(data_files),
            rows=total_rows,
            first_dt=first_candle_time,
            last_dt=last_candle_time,
            status=status_str,
            link=commit_url,
            dl_time_s=dl_time_s,
            push_time_s=push_time_s
        )
        summary_tracker.print_timeframe_summary(
            ticker=inst_name,
            platform="GitHub",
            tf=tf,
            chunks=len(data_files),
            rows=total_rows,
            first_dt=first_candle_time,
            last_dt=last_candle_time,
            status=status_str,
            link=commit_url,
            dl_time_s=dl_time_s,
            push_time_s=push_time_s
        )

    # Retain clone_dir temporarily so Hugging Face can reuse downloaded files if needed
    return updated_any

# -----------------------------------------------------------------------------
# Hugging Face Sync
# -----------------------------------------------------------------------------
def sync_ticker_huggingface(inst_name, repo_name, tracker, summary_tracker):
    api = HfApi(token=HF_TOKEN)
    hf_user = get_hf_username()
    repo_id = f"{hf_user}/{repo_name}"
    hf_repo_url = f"https://huggingface.co/datasets/{repo_id}"
    summary_tracker.record_repo(inst_name, "HuggingFace", hf_repo_url)
    
    # 1. Ensure public dataset repo exists on Hugging Face
    try:
        r_info = api.repo_info(repo_id=repo_id, repo_type="dataset")
        if r_info.private:
            api.update_repo_settings(repo_id=repo_id, private=False, repo_type="dataset")
    except Exception:
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=False)
        except Exception:
            pass

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
        if tf not in HUGGINGFACE_TIMEFRAMES:
            continue
            
        tf_prefix = f"{tf}/"
        tf_pods = sorted([f for f in existing_hf_files if f.startswith(tf_prefix) and f.endswith(".pod")])
        tf_data = [f for f in existing_hf_files if f.startswith(tf_prefix) and (f.endswith(".parquet") or f.endswith(".csv"))]
        
        skip_download = False
        t_check_hf = time.time()
        if tf_pods and tf_data:
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
                            check_duration = time.time() - t_check_hf
                            tf_link = f"{hf_repo_url}/tree/main/{tf}"
                            print(f"  * [HuggingFace] {tf}: Up-to-date ({len(tf_data)} chunks, {existing_total:,} rows)")
                            tracker[inst_name]["HuggingFace"][tf] = {
                                "first_candle": first_candle_final,
                                "last_candle": get_candle_time_str(last_c),
                                "total_rows": existing_total,
                                "chunks_count": len(tf_data),
                                "format": "parquet" if any(f.endswith(".parquet") for f in tf_data) else "csv",
                                "platform": "HuggingFace",
                                "last_updated": datetime.now(timezone.utc).isoformat()
                            }
                            summary_tracker.add_record(
                                ticker=inst_name,
                                platform="HuggingFace",
                                tf=tf,
                                chunks=len(tf_data),
                                rows=existing_total,
                                first_dt=first_candle_final,
                                last_dt=get_candle_time_str(last_c),
                                status="Up-to-date",
                                link=tf_link,
                                dl_time_s=check_duration,
                                push_time_s=0.0
                            )
                            summary_tracker.print_timeframe_summary(
                                ticker=inst_name,
                                platform="HuggingFace",
                                tf=tf,
                                chunks=len(tf_data),
                                rows=existing_total,
                                first_dt=get_candle_time_str(first_c),
                                last_dt=get_candle_time_str(last_c),
                                status="Up-to-date",
                                link=tf_link,
                                dl_time_s=check_duration,
                                push_time_s=0.0
                            )
                            skip_download = True
            except Exception:
                pass

        if skip_download:
            continue

        print(f"  * [HuggingFace] Syncing timeframe {tf}...")
        tf_dir = os.path.join(hf_base_dir, tf)
        os.makedirs(tf_dir, exist_ok=True)
        
        t_dl_start = time.time()
        is_reused = False
        clone_dir = f"gh_clone_{repo_name}"
        local_gh_dir = os.path.join(clone_dir, tf)
        
        has_gh_files = os.path.exists(local_gh_dir) and any(f.endswith(".parquet") for f in os.listdir(local_gh_dir))
        if not has_gh_files and (tf in GITHUB_TIMEFRAMES or bool(tracker.get(inst_name, {}).get("GitHub", {}).get(tf))):
            gh_user = get_gh_username()
            if not os.path.exists(clone_dir):
                print(f"    Fetching {repo_name} from GitHub ({tf}) to reuse files...")
                run_cmd(f"git clone --quiet --depth 1 https://x-access-token:{GH_TOKEN}@github.com/{gh_user}/{repo_name}.git {clone_dir}")
            has_gh_files = os.path.exists(local_gh_dir) and any(f.endswith(".parquet") for f in os.listdir(local_gh_dir))

        if has_gh_files:
            parquet_count = len([f for f in os.listdir(local_gh_dir) if f.endswith(".parquet")])
            print(f"    Reusing {tf} files from GitHub clone ({parquet_count} chunks)...")
            for f in os.listdir(local_gh_dir):
                shutil.copy2(os.path.join(local_gh_dir, f), os.path.join(tf_dir, f))
            is_reused = True
        else:
            oanda_tf = map_oanda_tf(tf)
            chunk_size = get_chunk_size(tf)
            bulk_download_and_split(
                inst_name, oanda_tf, None, None, chunk_size, tf_dir,
                file_prefix=repo_name, file_tf=tf
            )
        dl_time_s = time.time() - t_dl_start
        
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
                            first_candle_time = get_candle_time_str(fc)
                        if lc:
                            last_candle_time = get_candle_time_str(lc)
                except Exception:
                    pass
                    
            print(f"    Uploading {len(data_files)} file(s) for {tf} to Hugging Face...")
            t_up = time.time()
            try:
                commit_info = api.upload_folder(
                    folder_path=tf_dir,
                    repo_id=repo_id,
                    path_in_repo=tf,
                    repo_type="dataset",
                    commit_message=f"Automated {tf} parquet data update"
                )
                push_time_s = time.time() - t_up
                commit_oid = getattr(commit_info, 'oid', commit_info)
                commit_url = getattr(commit_info, 'commit_url', None) or f"{hf_repo_url}/commit/{commit_oid}"
                print(f"  [OK] [HuggingFace] Uploaded {tf} in {format_duration(push_time_s)}: {commit_url}")
                summary_tracker.record_push(inst_name, "HuggingFace", commit_url)
                
                tracker[inst_name]["HuggingFace"][tf] = {
                    "first_candle": first_candle_time,
                    "last_candle": last_candle_time,
                    "total_rows": total_rows,
                    "chunks_count": len(data_files),
                    "format": "parquet" if parquet_files else "csv",
                    "platform": "HuggingFace",
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }
                if is_reused and inst_name in tracker and "GitHub" in tracker[inst_name] and tf in tracker[inst_name]["GitHub"]:
                    if tracker[inst_name]["GitHub"][tf].get("total_rows", 0) < total_rows:
                        tracker[inst_name]["GitHub"][tf]["total_rows"] = total_rows
                summary_tracker.add_record(
                    ticker=inst_name,
                    platform="HuggingFace",
                    tf=tf,
                    chunks=len(data_files),
                    rows=total_rows,
                    first_dt=first_candle_time,
                    last_dt=last_candle_time,
                    status="Pushed",
                    link=commit_url,
                    dl_time_s=dl_time_s,
                    push_time_s=push_time_s,
                    is_reused=is_reused
                )
                summary_tracker.print_timeframe_summary(
                    ticker=inst_name,
                    platform="HuggingFace",
                    tf=tf,
                    chunks=len(data_files),
                    rows=total_rows,
                    first_dt=first_candle_time,
                    last_dt=last_candle_time,
                    status="Pushed",
                    link=commit_url,
                    dl_time_s=dl_time_s,
                    push_time_s=push_time_s,
                    is_reused=is_reused
                )
                updated_any = True
            except Exception as e:
                push_time_s = time.time() - t_up
                print(f"  [!] Hugging Face upload failed for {tf}: {e}")
                fail_url = f"{hf_repo_url}/tree/main/{tf}"
                summary_tracker.add_record(
                    ticker=inst_name,
                    platform="HuggingFace",
                    tf=tf,
                    chunks=len(data_files),
                    rows=total_rows,
                    first_dt=first_candle_time,
                    last_dt=last_candle_time,
                    status="Failed",
                    link=fail_url,
                    dl_time_s=dl_time_s,
                    push_time_s=push_time_s,
                    is_reused=is_reused
                )
                summary_tracker.print_timeframe_summary(
                    ticker=inst_name,
                    platform="HuggingFace",
                    tf=tf,
                    chunks=len(data_files),
                    rows=total_rows,
                    first_dt=first_candle_time,
                    last_dt=last_candle_time,
                    status="Failed",
                    link=fail_url,
                    dl_time_s=dl_time_s,
                    push_time_s=push_time_s,
                    is_reused=is_reused
                )
            
    if os.path.exists(hf_base_dir):
        shutil.rmtree(hf_base_dir, ignore_errors=True)
        
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

    # 2. Clean HF staging directory
    hf_base_dir = os.path.join("tmp_hf", repo_name)
    if os.path.exists(hf_base_dir):
        shutil.rmtree(hf_base_dir, ignore_errors=True)
    if os.path.exists("tmp_hf"):
        try:
            if not os.listdir("tmp_hf"):
                shutil.rmtree("tmp_hf", ignore_errors=True)
        except Exception:
            pass

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
                
            has_gh = any(tf in GITHUB_TIMEFRAMES for tf in TIMEFRAMES_TO_TEST)
            has_hf = any(tf in HUGGINGFACE_TIMEFRAMES for tf in TIMEFRAMES_TO_TEST)
            
            up_gh = False
            up_hf = False
            
            if has_gh:
                up_gh = sync_ticker_github(inst_name, repo_name, tracker, summary_tracker)
            if has_hf:
                up_hf = sync_ticker_huggingface(inst_name, repo_name, tracker, summary_tracker)
                
            # Update tracker after processing each ticker
            with open(tracker_file, "w") as f:
                json.dump(tracker, f, indent=2)
            print(f"  * Tracker metadata saved for {inst_name}")
                
            if os.path.exists(".git") and (up_gh or up_hf):
                subprocess.run(["git", "config", "user.name", "github-actions[bot]"], capture_output=True)
                subprocess.run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"], capture_output=True)
                subprocess.run(["git", "add", "tracker.json"], capture_output=True)
                commit_res = subprocess.run(["git", "commit", "-m", f"Update tracker for {inst_name}"], capture_output=True, text=True)
                if commit_res.returncode == 0:
                    subprocess.run(["git", "pull", "--quiet", "--rebase", "origin", "main"], capture_output=True)
                    subprocess.run(["git", "push", "--quiet", "origin", "main"], capture_output=True)

            # Print detailed, clean summary table for this ticker
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
