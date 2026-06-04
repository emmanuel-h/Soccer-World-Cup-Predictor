#!/usr/bin/env python3
"""
mpp_push.py — Push WC2026 predictions to Mon Petit Prono (mpp.football).

Usage:
  python mpp_push.py predictions_mpp.json \\
      --championship-id <ID> \\
      [--token-file .mpp_tokens.json] \\
      [--dry-run]

Authentication:
  On first run (or if the token file is absent), you will be prompted for
  your MPP refresh token.  Extract it from your browser while logged in at
  https://mpp.football:

    1. Open DevTools (F12) → Application → Local Storage → https://mpp.football
    2. Find the key that contains "refresh_token" (look inside the Auth0 JSON blob)
    3. Paste the token value when prompted.

  The token file is updated automatically on each run (tokens rotate).

Finding your championship ID:
  Run with --list-championships to print your active contests and their IDs.
"""

import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from difflib import SequenceMatcher

# ── MPP API constants ──────────────────────────────────────────────────────────

AUTH0_TOKEN_URL = "https://connect.ligue1.fr/oauth/token"
AUTH0_CLIENT_ID = "grX5jWGWWQ4Uq91oe7KPNDZ96FS3jr0X"
MPP_API_BASE    = "https://api.mpp.football"

DEFAULT_TOKEN_FILE = pathlib.Path(".mpp_tokens.json")


# ── HTTP helper ────────────────────────────────────────────────────────────────

def _request_json(
    url: str,
    method: str = "GET",
    payload: dict | None = None,
    bearer: str | None = None,
) -> dict | list:
    headers: dict[str, str] = {"Accept": "application/json"}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


# ── Auth ───────────────────────────────────────────────────────────────────────

def load_tokens(token_file: pathlib.Path) -> dict:
    if token_file.exists():
        return json.loads(token_file.read_text(encoding="utf-8"))
    return {}


def save_tokens(token_file: pathlib.Path, tokens: dict):
    token_file.write_text(json.dumps(tokens, indent=2), encoding="utf-8")
    token_file.chmod(0o600)


def refresh_access_token(refresh_token: str) -> tuple[str, str]:
    """Exchange refresh token for (access_token, new_refresh_token). Tokens rotate."""
    resp = _request_json(AUTH0_TOKEN_URL, "POST", {
        "grant_type":    "refresh_token",
        "client_id":     AUTH0_CLIENT_ID,
        "refresh_token": refresh_token,
    })
    return resp["access_token"], resp.get("refresh_token", refresh_token)


def _prompt_for_refresh_token() -> str:
    print("\nNo MPP refresh token found.")
    print("Steps to get it:")
    print("  1. Log in at https://mpp.football")
    print("  2. Open DevTools → Application → Local Storage → https://mpp.football")
    print("  3. Find the Auth0 entry containing 'refresh_token' and copy its value")
    token = input("\nPaste your MPP refresh token: ").strip()
    if not token:
        sys.exit("No token provided. Aborting.")
    return token


def ensure_access_token(token_file: pathlib.Path) -> str:
    tokens = load_tokens(token_file)

    if tokens.get("access_token") and tokens.get("expires_at", 0) - time.time() > 60:
        print("  Using cached MPP access token (still valid).")
        return tokens["access_token"]

    if not tokens.get("refresh_token"):
        tokens["refresh_token"] = _prompt_for_refresh_token()

    print("  Refreshing MPP access token …", end=" ", flush=True)
    try:
        access, new_refresh = refresh_access_token(tokens["refresh_token"])
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        sys.exit(f"\nAuth failed ({e.code}): {body}")

    tokens["access_token"]  = access
    tokens["refresh_token"] = new_refresh
    tokens["expires_at"]    = time.time() + 432000  # 5 days, matching MPP's grant
    save_tokens(token_file, tokens)
    print("OK")
    return access


# ── MPP API calls ──────────────────────────────────────────────────────────────

def list_contests(access_token: str) -> list[dict]:
    data = _request_json(f"{MPP_API_BASE}/user-contests", bearer=access_token)
    if isinstance(data, dict):
        return data.get("contestsCards", [])
    return data


def get_calendar(access_token: str, championship_id: str) -> list[dict]:
    """Fetch group-stage matches with resolved en-GB team names.

    The API only stores match IDs in the calendar and club IDs in each match,
    so this function makes N+M calls (N matches + M unique clubs) to build
    the full list.  Expects a few seconds on first run.
    """
    cal = _request_json(
        f"{MPP_API_BASE}/championship-calendar/{championship_id}",
        bearer=access_token,
    )

    round1_groups = cal.get("rounds", {}).get("1", {}).get("groups", {})
    all_match_ids: list[str] = []
    group_for_match: dict[str, str] = {}
    for grp, gdata in round1_groups.items():
        for mid in gdata.get("matchesIds", []):
            all_match_ids.append(mid)
            group_for_match[mid] = grp

    club_ids: set[str] = set()
    raw_matches: dict[str, dict] = {}
    for mid in all_match_ids:
        m = _request_json(f"{MPP_API_BASE}/championship-match/{mid}", bearer=access_token)
        hcid = m["home"]["clubId"]
        acid = m["away"]["clubId"]
        club_ids.add(hcid)
        club_ids.add(acid)
        raw_matches[mid] = {"home_cid": hcid, "away_cid": acid}
        time.sleep(0.05)

    clubs: dict[str, str] = {}
    for cid in club_ids:
        c = _request_json(f"{MPP_API_BASE}/championship-club/{cid}", bearer=access_token)
        clubs[cid] = c.get("name", {}).get("en-GB") or c.get("name", {}).get("fr-FR", cid)
        time.sleep(0.05)

    return [
        {
            "id":    mid,
            "home":  clubs[m["home_cid"]],
            "away":  clubs[m["away_cid"]],
            "group": group_for_match.get(mid, "?"),
        }
        for mid, m in raw_matches.items()
    ]


def submit_prediction(
    access_token: str, scope: str, match_id: str,
    home_score: int, away_score: int,
) -> dict:
    url = f"{MPP_API_BASE}/user-match-forecasts/entity/{scope}/match/{match_id}"
    return _request_json(url, "PATCH", {
        "homeScore":  home_score,
        "awayScore":  away_score,
        "originPage": "home",
    }, access_token)


# ── Team name matching ─────────────────────────────────────────────────────────

# Predictor uses the FIFA/Wikipedia English names; MPP uses different conventions
_NAME_ALIASES: dict[str, str] = {
    "Czech Republic":         "Czechia",
    "Bosnia and Herzegovina": "Bosnia",
    "Turkey":                 "Türkiye",
}


def _normalize(name: str) -> str:
    return _NAME_ALIASES.get(name, name)


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_mpp_match(
    prediction: dict, mpp_matches: list[dict], threshold: float = 0.6
) -> dict | None:
    """Fuzzy-match a prediction's home/away teams to an MPP match dict."""
    pred_home = _normalize(prediction["homeTeam"])
    pred_away = _normalize(prediction["awayTeam"])

    best_score, best_match = 0.0, None
    for m in mpp_matches:
        mpp_home = m.get("home", "")
        mpp_away = m.get("away", "")
        score         = (_similarity(pred_home, mpp_home) + _similarity(pred_away, mpp_away)) * 0.5
        score_swapped = (_similarity(pred_home, mpp_away) + _similarity(pred_away, mpp_home)) * 0.5
        best = max(score, score_swapped)
        if best > best_score:
            best_score = best
            best_match = m

    return best_match if best_score >= threshold else None


# ── CLI commands ───────────────────────────────────────────────────────────────

def _cmd_list_championships(access_token: str):
    contests = list_contests(access_token)
    print(f"\n{'Title':<30} {'championshipId':>16}  contestId")
    print("─" * 70)
    for c in contests:
        print(f"  {c.get('title','?'):<28} {c.get('championshipId','?'):>16}  {c.get('contestId','?')}")


def _preview_submissions(matched: list[tuple]):
    print("\n  Preview of submissions:")
    print(f"  {'Home':<24} {'Away':<24} {'Score':>7}  MPP match ID")
    print(f"  {'─'*70}")
    for pred, mpp_m in matched:
        match_id = mpp_m.get("id") or mpp_m.get("matchId") or "?"
        mpp_home = mpp_m.get("home", "?")
        mpp_away = mpp_m.get("away", "?")
        print(f"  {pred['homeTeam']:<24} {pred['awayTeam']:<24} "
              f"{pred['homeScore']}-{pred['awayScore']}  "
              f"→ {mpp_home} vs {mpp_away}  (id={match_id})")


def _submit_all(access_token: str, scope: str, matched: list[tuple]):
    ok = failed = 0
    for pred, mpp_m in matched:
        match_id = str(mpp_m.get("id") or mpp_m.get("matchId", ""))
        try:
            submit_prediction(access_token, scope, match_id,
                              pred["homeScore"], pred["awayScore"])
            ok += 1
            print(f"  ✓  {pred['homeTeam']} {pred['homeScore']}-{pred['awayScore']} {pred['awayTeam']}")
        except urllib.error.HTTPError as e:
            failed += 1
            body = e.read().decode(errors="replace")
            print(f"  ✗  {pred['homeTeam']} vs {pred['awayTeam']}  ({e.code}): {body[:120]}")
        time.sleep(0.2)  # be polite to the API
    print(f"\n  Done: {ok} submitted, {failed} failed.")


def _cmd_push_predictions(args, access_token: str):
    if not args.predictions_file:
        sys.exit("Error: predictions_file is required unless --list-championships is used.")
    if not args.championship_id:
        sys.exit("Error: --championship-id is required. Use --list-championships to find it.")

    preds = json.loads(pathlib.Path(args.predictions_file).read_text(encoding="utf-8"))
    print(f"  Loaded {len(preds)} predictions from {args.predictions_file}")

    print(f"  Fetching MPP calendar for championship {args.championship_id} …")
    print("  (resolving team names — this takes ~15 s on first run)")
    try:
        mpp_matches = get_calendar(access_token, args.championship_id)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        sys.exit(f"\nFailed to fetch calendar ({e.code}): {body}")
    print(f"  {len(mpp_matches)} matches loaded")

    matched, unmatched = [], []
    for pred in preds:
        mpp_match = find_mpp_match(pred, mpp_matches)
        if mpp_match:
            matched.append((pred, mpp_match))
        else:
            unmatched.append(pred)

    print(f"\n  Matched: {len(matched)}  |  Unmatched: {len(unmatched)}")

    if unmatched:
        print("\n  WARNING — Could not match these predictions to MPP matches:")
        for p in unmatched:
            print(f"    {p['homeTeam']} vs {p['awayTeam']}  ({p.get('group', '')})")
        print()

    if not matched:
        sys.exit("No predictions matched. Aborting.")

    _preview_submissions(matched)

    if args.dry_run:
        print("\n  [DRY RUN] No predictions submitted.")
        return

    if not args.yes:
        confirm = input(f"\n  Submit {len(matched)} predictions to MPP? [y/N] ").strip().lower()
        if confirm != "y":
            print("  Aborted.")
            return

    _submit_all(access_token, args.scope, matched)


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Push WC2026 predictions to Mon Petit Prono",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("predictions_file", nargs="?",
                        help="MPP-format JSON file produced by predictor.py --output-format mpp")
    parser.add_argument("--championship-id", metavar="ID",
                        help="MPP championship/competition ID (use --list-championships to find it)")
    parser.add_argument("--scope", metavar="SCOPE", default="general",
                        help="Prediction scope (default: general). "
                             "Use 'general' or a contest ID from --list-championships.")
    parser.add_argument("--token-file", metavar="FILE",
                        default=str(DEFAULT_TOKEN_FILE),
                        help="Path to JSON file storing the MPP refresh/access tokens "
                             f"(default: {DEFAULT_TOKEN_FILE})")
    parser.add_argument("--list-championships", action="store_true",
                        help="Print your active MPP contests and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be submitted without actually pushing")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip the confirmation prompt and submit immediately")
    return parser.parse_args()


def main():
    args       = parse_args()
    token_file = pathlib.Path(args.token_file)

    print("\n=== MPP Push ===")
    access = ensure_access_token(token_file)

    if args.list_championships:
        _cmd_list_championships(access)
    else:
        _cmd_push_predictions(args, access)


if __name__ == "__main__":
    main()
