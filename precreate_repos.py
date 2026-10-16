import os
import time
import requests
from datetime import datetime, timezone
from orchestrator import (
    GH_TOKEN, HF_TOKEN, HfApi,
    get_all_instruments, get_repo_name,
    get_gh_username, get_hf_username
)

def precreate_all_repositories():
    if not GH_TOKEN or not HF_TOKEN:
        print("[!] Missing GH_TOKEN or HF_TOKEN.")
        return

    gh_user = get_gh_username()
    hf_user = get_hf_username()
    api = HfApi(token=HF_TOKEN)

    print("==================================================")
    print(f"Pre-creating All Repositories for GitHub ({gh_user}) & Hugging Face ({hf_user})")
    print("==================================================")

    # 1. Fetch all OANDA instruments
    inst_dict = get_all_instruments()
    if not inst_dict:
        print("[!] Could not fetch instruments from OANDA.")
        return
    print(f"[*] Total instruments: {len(inst_dict)}")

    all_targets = []
    for inst_name in inst_dict:
        repo_name = get_repo_name(inst_name, inst_dict)
        all_targets.append((inst_name, repo_name))

    # 2. Fetch existing Hugging Face repositories
    print("[*] Fetching existing Hugging Face dataset repositories...")
    try:
        hf_existing = set([r.id.split('/')[-1] for r in api.list_datasets(author=hf_user)])
    except Exception as e:
        print(f"[!] Error fetching HF datasets: {e}")
        hf_existing = set()

    # 3. Fetch existing GitHub repositories
    print("[*] Fetching existing GitHub repositories...")
    gh_headers = {
        "Authorization": f"token {GH_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    gh_existing = set()
    page = 1
    while True:
        try:
            r = requests.get(f"https://api.github.com/user/repos?per_page=100&page={page}", headers=gh_headers, timeout=15)
            if r.status_code != 200:
                print(f"[!] Warning: /user/repos returned {r.status_code}")
                break
            items = r.json()
            if not items or not isinstance(items, list):
                break
            for item in items:
                gh_existing.add(item["name"])
            page += 1
        except Exception as e:
            print(f"[!] Error fetching GitHub repos: {e}")
            break

    print(f"[*] Hugging Face: {len(hf_existing)} existing | GitHub: {len(gh_existing)} existing")
    print("--------------------------------------------------")

    # 4. Sync Hugging Face
    missing_hf = [target for target in all_targets if target[1] not in hf_existing]
    print(f"\n[Hugging Face] {len(missing_hf)} repositories to create (out of {len(all_targets)})")
    for inst_name, repo_name in missing_hf:
        repo_id = f"{hf_user}/{repo_name}"
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=False)
            print(f"  [+] Created HF dataset: {repo_id}")
            hf_existing.add(repo_name)
        except Exception as e:
            if "already exists" in str(e).lower() or "409" in str(e):
                print(f"  [*] Already exists: {repo_id}")
                hf_existing.add(repo_name)
            else:
                print(f"  [!] Failed to create HF {repo_id}: {e}")

    # 5. Sync GitHub
    missing_gh = [target for target in all_targets if target[1] not in gh_existing]
    print(f"\n[GitHub] {len(missing_gh)} repositories to create (out of {len(all_targets)})")

    created_gh_count = 0
    rate_limited = False

    for idx, (inst_name, repo_name) in enumerate(missing_gh, 1):
        if rate_limited:
            print(f"  [-] Skipped {repo_name} (rate-limit active)")
            continue

        payload = {
            "name": repo_name,
            "private": False,
            "auto_init": True
        }
        try:
            res = requests.post("https://api.github.com/user/repos", headers=gh_headers, json=payload, timeout=15)
            if res.status_code == 201:
                created_gh_count += 1
                gh_existing.add(repo_name)
                print(f"  [{idx}/{len(missing_gh)}] [+] Created GitHub repo: {repo_name}")
                # Gentle stagger to honor secondary rate limits
                time.sleep(1.5)
            elif res.status_code == 422:
                # Already exists
                print(f"  [{idx}/{len(missing_gh)}] [*] Already exists: {repo_name}")
                gh_existing.add(repo_name)
            elif res.status_code == 403:
                reset_ts = res.headers.get("x-ratelimit-reset")
                reset_str = ""
                if reset_ts:
                    try:
                        dt = datetime.fromtimestamp(int(reset_ts), tz=timezone.utc)
                        reset_str = f" (resets at {dt.strftime('%H:%M:%S UTC')})"
                    except Exception:
                        pass
                print(f"  [{idx}/{len(missing_gh)}] [!] GitHub rate limit reached (403){reset_str} on {repo_name}!")
                rate_limited = True
            else:
                print(f"  [{idx}/{len(missing_gh)}] [!] Failed {repo_name} ({res.status_code}): {res.text[:100]}")
        except Exception as e:
            print(f"  [{idx}/{len(missing_gh)}] [!] Exception on {repo_name}: {e}")

    print("\n==================================================")
    print("Pre-creation Summary:")
    print(f"* Hugging Face: {len(hf_existing)}/{len(all_targets)} available")
    print(f"* GitHub:       {len(gh_existing)}/{len(all_targets)} available (created {created_gh_count} new)")
    if rate_limited:
        remaining = len(all_targets) - len(gh_existing)
        print(f"[!] GitHub hit hourly rate limit. {remaining} repos remaining. Re-run after cooldown to finish.")
    print("==================================================")

if __name__ == "__main__":
    precreate_all_repositories()
