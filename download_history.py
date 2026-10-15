import os
import time
import asyncio
import httpx
import numpy as np
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
from dateutil import parser
from dateutil.tz import tzlocal

# =============================================================================
# CUSTOMIZABLE PARAMETERS
# =============================================================================
# Available CHUNK_SIZE: 'day', 'week', 'month', '3_month', '6_month', 'year', '10_year'
CHUNK_SIZE = "month"

# Available TIMEFRAME:
# 'S5', 'S10', 'S15', 'S30' (Seconds)
# 'M1', 'M2', 'M4', 'M5', 'M10', 'M15', 'M30' (Minutes)
# 'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12' (Hours)
# 'D' (Day), 'W' (Week), 'M' (Month)
TIMEFRAME  = "M1"

INSTRUMENT = "BTC_USD"
BATCH_SIZE = 100
STAGGER_MS = 11

# Optional: Set specific start and end dates (format: "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DD").
# Leave as None to download everything from inception up to the current moment.
USER_START_DATE = None  # e.g., "2020-01-01 00:00:00"
USER_END_DATE   = None  # e.g., "2024-01-01 15:30:00"
USER_TIMEZONE   = "UTC" # "local" or "UTC"
# =============================================================================

def _load_env_candidates():
    candidates = [
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "oanda_downloader", ".env"),
        r"c:\Users\admin\Documents\tvline\oanda_downloader\.env",
        r"c:\Users\admin\Documents\tvline\.env"
    ]
    vars_dict = {}
    for p in candidates:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if "=" in line and not line.strip().startswith("#"):
                        k, v = line.split("=", 1)
                        key = k.strip()
                        val = v.split("#")[0].strip()
                        if key and key not in vars_dict:
                            vars_dict[key] = val
    return vars_dict

env_vars = _load_env_candidates()
API_KEY     = os.environ.get("OANDA_API_KEY") or env_vars.get("OANDA_API_KEY")
ENVIRONMENT = (os.environ.get("OANDA_ENVIRONMENT") or env_vars.get("OANDA_ENVIRONMENT", "practice")).lower()
if not API_KEY:
    print("Error: OANDA_API_KEY not found in env vars or .env"); exit(1)

OANDA_URL = ("https://api-fxtrade.oanda.com/v3" if ENVIRONMENT == "live"
             else "https://api-fxpractice.oanda.com/v3")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Accept-Datetime-Format": "RFC3339"}

TIMEFRAME_MINUTES = {
    "S5": 5/60, "S10": 10/60, "S15": 15/60, "S30": 30/60,
    "M1": 1, "M2": 2, "M4": 4, "M5": 5, "M10": 10, "M15": 15, "M30": 30,
    "H1": 60, "H2": 120, "H3": 180, "H4": 240, "H6": 360, "H8": 480, "H12": 720,
    "D": 1440, "W": 10080, "M": 43200
}

def format_time_duration(seconds):
    if seconds is None or seconds < 0:
        return "0s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"

# -----------------------------------------------------------------------------
# Calendar helpers
# -----------------------------------------------------------------------------
def snap_start_date(dt, cs):
    if cs == 'day':      return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if cs == 'week':     return (dt - timedelta(days=dt.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    if cs == 'month':    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if cs == '3_month':  return (dt - relativedelta(months=(dt.month-1)%3)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if cs == '6_month':  return (dt - relativedelta(months=(dt.month-1)%6)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if cs == 'year':     return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    if cs == '10_year':  return dt.replace(year=dt.year-(dt.year%10), month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    raise ValueError(cs)

def next_chunk(dt, cs):
    if cs == 'day':      return dt + timedelta(days=1)
    if cs == 'week':     return dt + timedelta(days=7)
    if cs == 'month':    return dt + relativedelta(months=1)
    if cs == '3_month':  return dt + relativedelta(months=3)
    if cs == '6_month':  return dt + relativedelta(months=6)
    if cs == 'year':     return dt + relativedelta(years=1)
    if cs == '10_year':  return dt + relativedelta(years=10)
    raise ValueError(cs)

def gen_chunks(start, end, cs):
    chunks, cur = [], snap_start_date(start, cs)
    while cur < end:
        nxt = next_chunk(cur, cs)
        chunks.append((cur, nxt))
        cur = nxt
    return chunks

# -----------------------------------------------------------------------------
# API helpers
# -----------------------------------------------------------------------------
async def find_inception(client, instrument, timeframe):
    print("Finding inception date...")
    url = f"{OANDA_URL}/instruments/{instrument}/candles"
    try:
        r = await client.get(url, params={"granularity": timeframe, "from": "1971-01-01T00:00:00Z", "count": 1, "price": "M"}, headers=HEADERS, timeout=10.0)
        if r.status_code == 200:
            candles = r.json().get('candles', [])
            if candles:
                t = candles[0]['time']
                print(f"-> Inception: {t}")
                return datetime.strptime(t[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception as e:
        print(f"Inception error: {e}")
    return datetime(2005, 1, 1, tzinfo=timezone.utc)

_market_state_cache = {}

async def find_latest_complete_candle(client, instrument, timeframe, is_standalone=True):
    cache_key = f"{instrument}_{timeframe}"
    if cache_key in _market_state_cache:
        cached_time, cached_dt = _market_state_cache[cache_key]
        if time.time() - cached_time < 60:
            return cached_dt
            
    if is_standalone:
        print(f"Fetching current market state for {instrument} ({timeframe})...")
    url = f"{OANDA_URL}/instruments/{instrument}/candles"
    params = {"granularity": timeframe, "count": 2, "price": "M"}
    
    for attempt in range(1, 4):
        try:
            # Fetch last 2 candles of the exact timeframe to guarantee we find the last 'complete' candle
            r = await client.get(url, params=params, headers=HEADERS, timeout=10.0)
            if r.status_code == 200:
                candles = r.json().get('candles', [])
                for c in reversed(candles):
                    if c['complete']:
                        t = c['time']
                        dt = datetime.strptime(t[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
                        _market_state_cache[cache_key] = (time.time(), dt)
                        return dt
            elif r.status_code in [401, 403]:
                # Direct HTTP fallback if connection pool renegotiation caused a transient 401
                import requests
                resp = requests.get(url, params=params, headers=HEADERS, timeout=10.0)
                if resp.status_code == 200:
                    candles = resp.json().get('candles', [])
                    for c in reversed(candles):
                        if c['complete']:
                            t = c['time']
                            dt = datetime.strptime(t[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
                            _market_state_cache[cache_key] = (time.time(), dt)
                            return dt
                print(f"Market state fetch attempt {attempt} failed with status {r.status_code}")
            else:
                print(f"Market state fetch attempt {attempt} failed with status {r.status_code}")
        except Exception as e:
            print(f"Error fetching market state (Attempt {attempt}): {e}")
            
        if attempt < 3:
            await asyncio.sleep(2 ** attempt)
            
    print("Warning: Could not fetch global market state. Assuming market is active.")
    return datetime.now(timezone.utc)

async def fetch_block(client, symbol, timeframe, start_dt, end_dt, block_idx):
    url    = f"{OANDA_URL}/instruments/{symbol}/candles"
    params = {"granularity": timeframe, "from": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "to": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"), "price": "M"}
    try:
        r = await client.get(url, params=params, headers=HEADERS, timeout=30.0)
        if r.status_code == 200:
            try:
                candles = r.json().get('candles', [])
                return {"block_idx": block_idx, "ok": True, "candles": candles, "outcome": {"block_idx": block_idx, "code": 200, "msg": "OK", "params": params}}
            except Exception as json_err:
                return {"block_idx": block_idx, "ok": False, "retry": True, "outcome": {"block_idx": block_idx, "code": -2, "msg": f"JSON decode error: {json_err}", "params": params}}
        elif r.status_code in (429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 408):
            # Transient server / proxy / Cloudflare / rate-limit errors -> always retry
            return {"block_idx": block_idx, "ok": False, "retry": True, "outcome": {"block_idx": block_idx, "code": r.status_code, "msg": f"Transient HTTP {r.status_code}", "params": params}}
        elif r.status_code in (401, 403):
            # Transient session / connection renegotiation drops
            return {"block_idx": block_idx, "ok": False, "retry": True, "outcome": {"block_idx": block_idx, "code": r.status_code, "msg": f"Auth renegotiation ({r.status_code})", "params": params}}
        else:
            return {"block_idx": block_idx, "ok": False, "retry": False, "outcome": {"block_idx": block_idx, "code": r.status_code, "msg": r.text[:100], "params": params}}
    except Exception as e:
        return {"block_idx": block_idx, "ok": False, "retry": True, "outcome": {"block_idx": block_idx, "code": -1, "msg": str(e)[:100], "params": params}}

async def fetch_block_staggered(client, symbol, timeframe, start_dt, end_dt, block_idx, delay_s):
    if delay_s > 0:
        await asyncio.sleep(delay_s)
    return await fetch_block(client, symbol, timeframe, start_dt, end_dt, block_idx)

# -----------------------------------------------------------------------------
# Chunk writer
# -----------------------------------------------------------------------------
def write_chunk_to_disk(save_dir, symbol, timeframe, c_start, c_end, buf, now_utc,
                         chunk_outcomes, chunk_errors, total_blocks_in_run,
                         file_prefix=None, file_tf=None):
    start_str = c_start.strftime("%Y-%m-%d")
    end_str   = (c_end - timedelta(days=1)).strftime("%Y-%m-%d")
    
    p_sym = file_prefix if file_prefix else symbol
    p_tf  = file_tf if file_tf else timeframe
    
    parquet_file = os.path.join(save_dir, f"{p_sym}_{p_tf}_{start_str}_to_{end_str}.parquet")
    pod_file     = parquet_file.replace(".parquet", ".pod")

    times_l, opens_l, highs_l, lows_l, closes_l, vols_l = (
        buf["times"], buf["opens"], buf["highs"], buf["lows"], buf["closes"], buf["vols"])

    n = len(times_l)
    import pandas as pd

    if n > 0:
        df = pd.DataFrame({
            "time_str": times_l,
            "open": opens_l,
            "high": highs_l,
            "low": lows_l,
            "close": closes_l,
            "volume": vols_l
        })
        df.drop_duplicates(subset=["time_str"], inplace=True)
        df.sort_values("time_str", inplace=True)

        # Fast vectorized Unix timestamp conversion without Python strptime
        df["time"] = (pd.to_datetime(df["time_str"]).astype("int64") // 10**9).astype("int64")
        df["open"] = df["open"].astype("float64")
        df["high"] = df["high"].astype("float64")
        df["low"] = df["low"].astype("float64")
        df["close"] = df["close"].astype("float64")
        df["volume"] = df["volume"].astype("int64")

        n = len(df)
        first_row = df.iloc[0]
        last_row = df.iloc[-1]

        first_candle = {
            "time": int(first_row["time"]),
            "Time": str(first_row["time_str"]),
            "open": float(first_row["open"]),
            "high": float(first_row["high"]),
            "low": float(first_row["low"]),
            "close": float(first_row["close"]),
            "volume": int(first_row["volume"])
        }
        last_candle = {
            "time": int(last_row["time"]),
            "Time": str(last_row["time_str"]),
            "open": float(last_row["open"]),
            "high": float(last_row["high"]),
            "low": float(last_row["low"]),
            "close": float(last_row["close"]),
            "volume": int(last_row["volume"])
        }

        out_df = df[["time", "open", "high", "low", "close", "volume"]]
        out_df.to_parquet(parquet_file, compression="snappy", index=False)
    else:
        first_candle = None
        last_candle = None
        df = pd.DataFrame({
            "time": pd.Series(dtype="int64"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="int64")
        })
        df.to_parquet(parquet_file, compression="snappy", index=False)

    is_empty = (n == 0)
    if chunk_errors:
        status = "ERROR"
    else:
        status = "VERIFIED_PARTIAL" if c_end > now_utc else ("EMPTY" if is_empty else "VERIFIED")

    pod = {
        "pod_version": "4.0",
        "instrument": symbol, "granularity": timeframe,
        "chunk_start": start_str, "chunk_end": end_str,
        "parquet_file": os.path.basename(parquet_file),
        "data_file": os.path.basename(parquet_file),
        "row_count": n, "first_candle": first_candle, "last_candle": last_candle,
        "csv_file": os.path.basename(parquet_file),
        "csv_row_count": n, "csv_first_candle": first_candle, "csv_last_candle": last_candle,
        "status": status,
        "verified_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
        "download_request": {
            "total_api_blocks_in_run": total_blocks_in_run,
            "chunk_api_blocks": len(chunk_outcomes),
            "chunk_permanent_errors": len(chunk_errors),
            "permanent_error_details": chunk_errors,
            "all_block_outcomes": chunk_outcomes
        }
    }
    with open(pod_file, "w") as f:
        json.dump(pod, f, indent=2)

    return os.path.basename(parquet_file), n, status

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
async def _async_bulk_download(symbol, timeframe, start_dt, end_dt, chunk_size, save_dir, file_prefix=None, file_tf=None):
    os.makedirs(save_dir, exist_ok=True)
    now_utc = datetime.now(timezone.utc)

    limits = httpx.Limits(max_connections=200, max_keepalive_connections=100, keepalive_expiry=30.0)
    timeout = httpx.Timeout(30.0, connect=10.0)
    try:
        client_ctx = httpx.AsyncClient(http2=True, headers=HEADERS, limits=limits, timeout=timeout)
    except Exception:
        client_ctx = httpx.AsyncClient(headers=HEADERS, limits=limits, timeout=timeout)

    async with client_ctx as client:
        # Start date handling
        if not start_dt:
            start_dt = await find_inception(client, symbol, timeframe)
        else:
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            else:
                start_dt = start_dt.astimezone(timezone.utc)
                
        # End date handling
        if not end_dt:
            end_dt = now_utc
        else:
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)
            else:
                end_dt = end_dt.astimezone(timezone.utc)

        is_standalone = (__name__ == "__main__")
        from dateutil import tz
        ist_tz = tz.gettz('Asia/Kolkata')
        s_local = start_dt.astimezone(ist_tz).strftime("%Y-%m-%d %H:%M:%S")
        e_local = end_dt.astimezone(ist_tz).strftime("%Y-%m-%d %H:%M:%S")

        if is_standalone:
            print("==================================================")
            print(f"OANDA Fast Downloader - Streaming Hybrid v3.5 ({symbol})")
            print("==================================================")
            print(f"Target Range (IST): {s_local} to {e_local}")
        else:
            # Concise one-liner for server.py logs
            print(f"[OANDA] {symbol} | {s_local} -> {e_local} IST")

        global_latest_complete_dt = await find_latest_complete_candle(client, symbol, timeframe, is_standalone)

        # 1. All calendar chunks
        all_chunks = gen_chunks(start_dt, end_dt, chunk_size)

        # 2. POD-based skip
        unverified_chunks = []
        for c_start, c_end in all_chunks:
            ss = c_start.strftime("%Y-%m-%d")
            es = (c_end - timedelta(days=1)).strftime("%Y-%m-%d")
            
            p_sym = file_prefix if file_prefix else symbol
            p_tf  = file_tf if file_tf else timeframe
            pf = os.path.join(save_dir, f"{p_sym}_{p_tf}_{ss}_to_{es}.pod")
            parquet_f = os.path.join(save_dir, f"{p_sym}_{p_tf}_{ss}_to_{es}.parquet")
            csv_f = os.path.join(save_dir, f"{p_sym}_{p_tf}_{ss}_to_{es}.csv")
            data_file_exists = os.path.exists(parquet_f) or os.path.exists(csv_f)
            skip = False
            if os.path.exists(pf) and data_file_exists:
                try:
                    pod = json.load(open(pf))
                    status = pod.get("status")
                    if status in ("VERIFIED", "EMPTY"):
                        skip = True
                    elif status == "VERIFIED_PARTIAL":
                        first_candle = pod.get("first_candle") or pod.get("csv_first_candle")
                        last_candle = pod.get("last_candle") or pod.get("csv_last_candle")
                        if first_candle and "Time" in first_candle and last_candle and "Time" in last_candle:
                            t_start_str = first_candle["Time"]
                            first_dt = datetime.strptime(t_start_str[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
                            t_str = last_candle["Time"]
                            last_dt = datetime.strptime(t_str[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
                            
                            # Verify chunk has not been truncated (first candle matches expected start)
                            expected_start = max(c_start, start_dt)
                            if (first_dt - expected_start).days <= 10 and last_dt >= global_latest_complete_dt:
                                skip = True
                except Exception:
                    pass
            if not skip:
                unverified_chunks.append((c_start, c_end))

        skipped_count = len(all_chunks) - len(unverified_chunks)
        if not unverified_chunks:
            print(f"  * {symbol} ({timeframe}): All {len(all_chunks)} chunks up-to-date (cached)")
            return

        print(f"  * {symbol} ({timeframe}): {skipped_count} chunks cached | {len(unverified_chunks)} chunk(s) to download")

        fetch_start = max(unverified_chunks[0][0], start_dt)
        max_valid_fetch_end = global_latest_complete_dt + timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
        fetch_end   = min(unverified_chunks[-1][1], end_dt, max_valid_fetch_end)
        
        if fetch_start >= fetch_end:
            print(f"  * {symbol} ({timeframe}): All data up to {fetch_end.strftime('%Y-%m-%d')} already verified")
            return

        block_dur  = timedelta(minutes=5000 * TIMEFRAME_MINUTES[timeframe])
        api_blocks = []
        cur = fetch_start
        while cur < fetch_end:
            nxt = min(cur + block_dur, fetch_end)
            api_blocks.append((cur, nxt))
            cur = nxt
        print(f"    Fetching {len(api_blocks)} API blocks ({fetch_start.strftime('%Y-%m-%d')} -> {fetch_end.strftime('%Y-%m-%d')})...")

        # 3. Pre-compute block↔chunk overlap
        c_start_strs = [c.strftime("%Y-%m-%dT%H:%M:%S.000000000Z") for c, _ in unverified_chunks]
        c_end_strs   = [e.strftime("%Y-%m-%dT%H:%M:%S.000000000Z") for _, e in unverified_chunks]

        block_serves: dict[int, list[int]] = defaultdict(list)
        chunk_needs:  dict[int, set]       = defaultdict(set)

        for b_idx, (b_start, b_end) in enumerate(api_blocks):
            for c_idx, (c_start, c_end) in enumerate(unverified_chunks):
                if b_start < c_end and b_end > c_start:
                    block_serves[b_idx].append(c_idx)
                    chunk_needs[c_idx].add(b_idx)

        # Blocks that serve NO unverified chunks -> skip entirely
        skippable = {b_idx for b_idx in range(len(api_blocks)) if not block_serves[b_idx]}
        if skippable:
            print(f"    Skipping {len(skippable)} blocks (already verified ranges)")

        # 4. Per-chunk accumulators
        chunk_buffers: dict[int, dict] = {
            c_idx: {"times": [], "opens": [], "highs": [], "lows": [], "closes": [], "vols": []}
            for c_idx in range(len(unverified_chunks))
        }
        chunk_pending: dict[int, set] = {c_idx: set(chunk_needs[c_idx]) for c_idx in range(len(unverified_chunks))}
        all_block_outcomes: dict = {}
        permanent_failures: list = []

        # 5. Streaming fetch + write
        write_pool = ThreadPoolExecutor(max_workers=8)
        write_futures = []
        total_start = time.time()
        chunks_written = 0

        pending = [(b_idx, b_start, b_end) for b_idx, (b_start, b_end) in enumerate(api_blocks)
                   if b_idx not in skippable]
        total_active_blocks = len(pending)
        last_progress_time = 0.0

        block_attempts = defaultdict(int)
        loop_iteration = 1

        while pending:
            batch          = pending[:BATCH_SIZE]
            pending        = pending[BATCH_SIZE:]

            # Dispatch batch with smooth micro-stagger to honor OANDA 100 req/s limit and prevent 429 burst errors
            tasks   = [
                fetch_block_staggered(client, symbol, timeframe, b_start, b_end, b_idx, i * (STAGGER_MS / 1000.0))
                for i, (b_idx, b_start, b_end) in enumerate(batch)
            ]
            results = await asyncio.gather(*tasks)

            failed_retry = []
            for r in results:
                b_idx   = r["block_idx"]
                outcome = r["outcome"]
                all_block_outcomes[b_idx] = outcome

                if r["ok"]:
                    served = block_serves[b_idx]
                    for c in r["candles"]:
                        if not c['complete']:
                            continue
                        t = c['time']
                        m = c['mid']
                        for c_idx in served:
                            if c_start_strs[c_idx] <= t < c_end_strs[c_idx]:
                                buf = chunk_buffers[c_idx]
                                buf["times"].append(t)
                                buf["opens"].append(float(m['o']));  buf["highs"].append(float(m['h']))
                                buf["lows"].append(float(m['l']));   buf["closes"].append(float(m['c']))
                                buf["vols"].append(int(c['volume']))
                else:
                    if r.get("retry"):
                        block_attempts[b_idx] += 1
                        if block_attempts[b_idx] > 50:
                            print(f"    [!] Block {b_idx} failed after 50 retries: {outcome['code']} {outcome['msg']}")
                            permanent_failures.append(outcome)
                        else:
                            failed_retry.append((b_idx, *api_blocks[b_idx]))
                    else:
                        print(f"    [!] Block {b_idx} non-retryable error: {outcome['code']} {outcome['msg']}")
                        permanent_failures.append(outcome)

                # Process chunk completion only if the block is fully resolved (success or permanent failure)
                if r["ok"] or not r.get("retry") or block_attempts[b_idx] > 50:
                    for c_idx in block_serves[b_idx]:
                        chunk_pending[c_idx].discard(b_idx)
                        if not chunk_pending[c_idx] and c_idx in chunk_buffers:
                            c_start, c_end = unverified_chunks[c_idx]
                            buf  = chunk_buffers.pop(c_idx)
    
                            chunk_outcomes = []
                            chunk_errors   = []
                            for o in sorted(all_block_outcomes.values(), key=lambda x: x["block_idx"]):
                                bs = datetime.strptime(o["params"]["from"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                                be = datetime.strptime(o["params"]["to"],   "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                                if bs < c_end and be > c_start:
                                    fmt = {"from": o["params"]["from"], "to": o["params"]["to"],
                                           "code": o["code"], "msg": o["msg"]}
                                    chunk_outcomes.append(fmt)
                                    if o in permanent_failures:
                                        chunk_errors.append(fmt)
    
                            fut = write_pool.submit(
                                write_chunk_to_disk,
                                save_dir, symbol, timeframe, c_start, c_end, buf, now_utc,
                                chunk_outcomes, chunk_errors, len(api_blocks),
                                file_prefix, file_tf
                            )
                            write_futures.append((c_idx, fut))
                            chunks_written += 1

            pending = failed_retry + pending
            blocks_done = total_active_blocks - len(pending)

            now_t = time.time()
            if total_active_blocks > 0 and (now_t - last_progress_time >= 0.5 or not pending):
                pct = (blocks_done / total_active_blocks) * 100.0
                elapsed_s = now_t - total_start
                rate = blocks_done / elapsed_s if elapsed_s > 0 else 0.0
                eta_s = (total_active_blocks - blocks_done) / rate if rate > 0 else 0.0
                eta_str = format_time_duration(eta_s) if pending else "Done"
                prog_line = f"  [Progress] {blocks_done:,}/{total_active_blocks:,} ({pct:.1f}%) | Chunks: {chunks_written}/{len(unverified_chunks)} | {rate:.1f} blk/s | ETA: {eta_str}"
                sys.stdout.write(f"\r{prog_line.ljust(79)[:79]}")
                sys.stdout.flush()
                last_progress_time = now_t

            if failed_retry:
                print()  # Move past in-place line so retry message is on its own line
                max_attempts = max(block_attempts[b[0]] for b in failed_retry)
                sleep_time = min(60, 2 ** min(max_attempts, 6))
                codes = {all_block_outcomes[b[0]]["code"] for b in failed_retry if b[0] in all_block_outcomes}
                print(f"    [Retry] {len(failed_retry)} block(s) (codes: {sorted(list(codes), key=lambda x: str(x))}) waiting {sleep_time}s (attempt {max_attempts}/50)...")
                await asyncio.sleep(sleep_time)
                loop_iteration += 1

        print()  # Finalize the single in-place progress line with a newline

        total_rows_saved = 0
        chunks_saved = 0
        t_net = time.time() - total_start
        try:
            for c_idx, fut in write_futures:
                fname, rows, status = fut.result()
                total_rows_saved += rows
                chunks_saved += 1
                if len(write_futures) <= 3:
                    print(f"    Saved: {fname} ({rows:,} rows)")
            if len(write_futures) > 3:
                print(f"    Saved {chunks_saved} chunks ({total_rows_saved:,} rows) in {time.time()-total_start-t_net:.1f}s")
        except Exception as e:
            print(f"    [Error] Writing chunk: {e}")
        finally:
            write_pool.shutdown(wait=False)

        print(f"    Download finished in {time.time()-total_start:.1f}s")

def bulk_download_and_split(symbol, timeframe, start_dt, end_dt, chunk_size, save_dir, seconds_per_candle=None, file_prefix=None, file_tf=None):
    """
    Synchronous wrapper to safely run the asyncio downloader loop from inside
    a threaded web server environment (like Flask) without blocking the main event loop.
    """
    import threading

    def run_in_thread():
        asyncio.run(_async_bulk_download(symbol, timeframe, start_dt, end_dt, chunk_size, save_dir, file_prefix, file_tf))

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    thread.join()

def cleanup_memory():
    global _market_state_cache
    _market_state_cache.clear()
    import gc
    gc.collect()

if __name__ == "__main__":
    start_dt = None
    if USER_START_DATE:
        dt = parser.parse(USER_START_DATE)
        start_dt = dt.replace(tzinfo=tzlocal()).astimezone(timezone.utc) if USER_TIMEZONE.lower() == "local" else dt.replace(tzinfo=timezone.utc)
    end_dt = None
    if USER_END_DATE:
        dt = parser.parse(USER_END_DATE)
        end_dt = dt.replace(tzinfo=tzlocal()).astimezone(timezone.utc) if USER_TIMEZONE.lower() == "local" else dt.replace(tzinfo=timezone.utc)
        
    save_dir = f"oanda_downloader/data/{INSTRUMENT}/{TIMEFRAME}"
    bulk_download_and_split(INSTRUMENT, TIMEFRAME, start_dt, end_dt, CHUNK_SIZE, save_dir)
