"""Schema exploration for Cricsheet T20 JSON data.

Read-only exploration script — no forecasting/EDA logic here. Run it to
regenerate the raw findings that back docs/schema_notes.md.

Usage:
    python scripts/explore_schema.py
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "t20s_json"


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_match_files() -> list[Path]:
    return sorted(DATA_DIR.glob("*.json"))


def inventory() -> None:
    print("=== Inventory ===")
    all_files = list(DATA_DIR.iterdir())
    ext_counts = Counter(f.suffix for f in all_files if f.is_file())
    print("Extension counts:", dict(ext_counts))
    print("Non-json files:", [f.name for f in all_files if f.is_file() and f.suffix != ".json"])
    print("Match json count:", len(list_match_files()))


def sample_files(n: int = 10, seed: int = 42) -> list[Path]:
    files = list_match_files()
    matches = [load(p) for p in files]

    # Try to get a spread across match_type / gender / year.
    buckets: dict[tuple[str, str, str], Path] = {}
    for p, m in zip(files, matches):
        info = m.get("info", {})
        year = str(info.get("dates", [None])[0])[:4]
        key = (info.get("match_type"), info.get("gender"), year)
        buckets.setdefault(key, p)

    rng = random.Random(seed)
    keys = list(buckets.keys())
    rng.shuffle(keys)
    chosen = [buckets[k] for k in keys[:n]]
    if len(chosen) < n:
        remaining = [p for p in files if p not in chosen]
        chosen += rng.sample(remaining, n - len(chosen))
    return chosen


def describe_structure(sample_paths: list[Path]) -> None:
    print("\n=== Sample structure dump ===")
    for p in sample_paths:
        m = load(p)
        info = m.get("info", {})
        print(f"\n--- {p.name} ---")
        print("meta:", m.get("meta"))
        print(
            "info keys:", sorted(info.keys()),
        )
        print(
            "match_type=%s gender=%s team_type=%s teams=%s dates=%s season=%s"
            % (
                info.get("match_type"),
                info.get("gender"),
                info.get("team_type"),
                info.get("teams"),
                info.get("dates"),
                info.get("season"),
            )
        )
        print("outcome:", info.get("outcome"))
        print("toss:", info.get("toss"))
        print("event:", info.get("event"))
        print("num innings:", len(m.get("innings", [])))
        for i, inn in enumerate(m.get("innings", [])):
            print(
                f"  innings[{i}] keys={sorted(inn.keys())} team={inn.get('team')} "
                f"target={inn.get('target')} super_over={inn.get('super_over')} "
                f"declared={inn.get('declared')} forfeited={inn.get('forfeited')} "
                f"penalty_runs={inn.get('penalty_runs')} absent_hurt={inn.get('absent_hurt')}"
            )
            overs = inn.get("overs", [])
            if overs:
                first_over = overs[0]
                print("    first over keys:", sorted(first_over.keys()))
                if first_over.get("deliveries"):
                    d = first_over["deliveries"][0]
                    print("    first delivery:", d)


def scan_all_matches() -> dict[str, Any]:
    files = list_match_files()

    outcome_shapes: Counter[str] = Counter()
    result_values: Counter[str] = Counter()
    method_values: Counter[str] = Counter()
    eliminator_count = 0
    winner_by_runs = 0
    winner_by_wickets = 0
    winner_by_innings = 0

    innings_count_dist: Counter[int] = Counter()
    super_over_innings_count = 0
    matches_with_super_over = 0

    target_present_2nd_innings = 0
    target_absent_2nd_innings = 0
    target_mismatch = 0
    target_mismatch_examples: list[dict[str, Any]] = []

    match_type_dist: Counter[str] = Counter()
    gender_dist: Counter[str] = Counter()
    team_type_dist: Counter[str] = Counter()
    year_dist: Counter[str] = Counter()
    balls_per_over_dist: Counter[int] = Counter()

    venue_raw_dist: Counter[str] = Counter()
    venue_city_pairs: dict[str, set[str]] = defaultdict(set)

    name_to_ids: dict[str, set[str]] = defaultdict(set)
    id_to_names: dict[str, set[str]] = defaultdict(set)
    files_without_registry = 0

    info_field_presence: Counter[str] = Counter()
    innings_field_presence: Counter[str] = Counter()
    season_format: Counter[str] = Counter()
    wicket_kinds: Counter[str] = Counter()
    extras_types: Counter[str] = Counter()
    fielder_shapes: Counter[str] = Counter()
    powerplay_types: Counter[str] = Counter()
    revision_dist: Counter[Any] = Counter()
    data_version_dist: Counter[str] = Counter()

    for p in files:
        m = load(p)
        info = m.get("info", {}) or {}
        outcome = info.get("outcome", {}) or {}

        meta = m.get("meta", {}) or {}
        data_version_dist[str(meta.get("data_version"))] += 1
        revision_dist[meta.get("revision")] += 1

        for field in (
            "city", "event", "player_of_match", "season", "match_type_number",
        ):
            if info.get(field) is not None:
                info_field_presence[field] += 1
        season_val = info.get("season")
        if season_val is not None:
            season_format["split_year" if "/" in str(season_val) else "single_year"] += 1

        shape_parts = sorted(outcome.keys())
        outcome_shapes[",".join(shape_parts) if shape_parts else "<empty>"] += 1

        if "result" in outcome:
            result_values[str(outcome.get("result"))] += 1
        if "method" in outcome:
            method_values[str(outcome.get("method"))] += 1
        if "eliminator" in outcome:
            eliminator_count += 1
        by = outcome.get("by", {}) or {}
        if "runs" in by:
            winner_by_runs += 1
        if "wickets" in by:
            winner_by_wickets += 1
        if "innings" in by:
            winner_by_innings += 1

        innings = m.get("innings", []) or []
        innings_count_dist[len(innings)] += 1
        has_super_over = any(inn.get("super_over") for inn in innings)
        if has_super_over:
            matches_with_super_over += 1
        super_over_innings_count += sum(1 for inn in innings if inn.get("super_over"))

        for inn in innings:
            for field in (
                "powerplays", "target", "super_over", "declared", "forfeited",
                "penalty_runs", "absent_hurt", "miscounted_overs",
            ):
                if inn.get(field) is not None:
                    innings_field_presence[field] += 1
            for pp in inn.get("powerplays", []) or []:
                powerplay_types[str(pp.get("type"))] += 1
            for over in inn.get("overs", []) or []:
                for delivery in over.get("deliveries", []) or []:
                    extras = delivery.get("extras", {}) or {}
                    for extra_key in extras:
                        extras_types[extra_key] += 1
                    for wicket in delivery.get("wickets", []) or []:
                        wicket_kinds[str(wicket.get("kind"))] += 1
                        for fielder in wicket.get("fielders", []) or []:
                            fielder_shapes[type(fielder).__name__ + (
                                ":" + ",".join(sorted(fielder.keys())) if isinstance(fielder, dict) else ""
                            )] += 1

        # target check on innings[1] (2nd innings), only for non-super-over innings
        non_super_innings = [inn for inn in innings if not inn.get("super_over")]
        if len(non_super_innings) >= 2:
            second = non_super_innings[1]
            first = non_super_innings[0]
            target = second.get("target")
            if target:
                target_present_2nd_innings += 1
                first_total = sum(
                    (d.get("runs", {}) or {}).get("total", 0) or 0
                    for over in first.get("overs", []) or []
                    for d in over.get("deliveries", []) or []
                )
                expected = first_total + 1
                actual = target.get("runs")
                if actual is not None and actual != expected:
                    target_mismatch += 1
                    if len(target_mismatch_examples) < 15:
                        target_mismatch_examples.append(
                            {
                                "match_id": p.stem,
                                "first_innings_total": first_total,
                                "target_runs": actual,
                                "target_overs": target.get("overs"),
                                "outcome_method": info.get("outcome", {}).get("method"),
                            }
                        )
            else:
                target_absent_2nd_innings += 1

        match_type_dist[str(info.get("match_type"))] += 1
        gender_dist[str(info.get("gender"))] += 1
        team_type_dist[str(info.get("team_type"))] += 1
        dates = info.get("dates") or [None]
        year_dist[str(dates[0])[:4]] += 1
        balls_per_over_dist[info.get("balls_per_over")] += 1

        venue = info.get("venue")
        city = info.get("city")
        if venue:
            venue_raw_dist[venue] += 1
            venue_city_pairs[venue].add(str(city))

        registry = (info.get("registry", {}) or {}).get("people", {}) or {}
        if not registry:
            files_without_registry += 1
        for name, pid in registry.items():
            name_to_ids[name].add(pid)
            id_to_names[pid].add(name)

    # Venue duplicate suspects: names that share a "base" token (crude heuristic)
    venue_names = sorted(venue_raw_dist.keys())
    suspected_dupes = []
    normalized: dict[str, list[str]] = defaultdict(list)
    for v in venue_names:
        base = v.split(",")[0].strip().lower()
        normalized[base].append(v)
    for base, variants in normalized.items():
        if len(variants) > 1:
            suspected_dupes.append(variants)

    names_with_multiple_ids = {n: ids for n, ids in name_to_ids.items() if len(ids) > 1}
    ids_with_multiple_names = {i: names for i, names in id_to_names.items() if len(names) > 1}

    return {
        "info_field_presence": info_field_presence,
        "innings_field_presence": innings_field_presence,
        "season_format": season_format,
        "wicket_kinds": wicket_kinds,
        "extras_types": extras_types,
        "fielder_shapes": fielder_shapes,
        "powerplay_types": powerplay_types,
        "revision_dist": revision_dist,
        "data_version_dist": data_version_dist,
        "n_files": len(files),
        "outcome_shapes": outcome_shapes,
        "result_values": result_values,
        "method_values": method_values,
        "eliminator_count": eliminator_count,
        "winner_by_runs": winner_by_runs,
        "winner_by_wickets": winner_by_wickets,
        "winner_by_innings": winner_by_innings,
        "innings_count_dist": innings_count_dist,
        "matches_with_super_over": matches_with_super_over,
        "super_over_innings_count": super_over_innings_count,
        "target_present_2nd_innings": target_present_2nd_innings,
        "target_absent_2nd_innings": target_absent_2nd_innings,
        "target_mismatch": target_mismatch,
        "target_mismatch_examples": target_mismatch_examples,
        "match_type_dist": match_type_dist,
        "gender_dist": gender_dist,
        "team_type_dist": team_type_dist,
        "year_dist": year_dist,
        "balls_per_over_dist": balls_per_over_dist,
        "venue_count": len(venue_raw_dist),
        "venue_suspected_dupes": suspected_dupes,
        "files_without_registry": files_without_registry,
        "names_with_multiple_ids": names_with_multiple_ids,
        "ids_with_multiple_names": ids_with_multiple_names,
    }


def print_scan_report(report: dict[str, Any]) -> None:
    print("\n=== Full scan ===")
    print("n_files:", report["n_files"])
    print("\noutcome shapes (sorted keys of info.outcome):")
    for shape, count in report["outcome_shapes"].most_common():
        print(f"  {shape or '<empty>'}: {count}")
    print("\nresult values:", dict(report["result_values"]))
    print("method values:", dict(report["method_values"]))
    print("eliminator present:", report["eliminator_count"])
    print(
        "winner+by.runs:", report["winner_by_runs"],
        " winner+by.wickets:", report["winner_by_wickets"],
        " winner+by.innings:", report["winner_by_innings"],
    )

    print("\ninnings count distribution:", dict(sorted(report["innings_count_dist"].items())))
    print("matches with >=1 super_over innings:", report["matches_with_super_over"])
    print("total super_over-flagged innings:", report["super_over_innings_count"])

    print("\n2nd innings target present:", report["target_present_2nd_innings"])
    print("2nd innings target absent:", report["target_absent_2nd_innings"])
    print("target != first_innings_total+1:", report["target_mismatch"])
    print("examples:")
    for ex in report["target_mismatch_examples"]:
        print("   ", ex)

    print("\nmatch_type distribution:", dict(report["match_type_dist"].most_common()))
    print("gender distribution:", dict(report["gender_dist"]))
    print("team_type distribution:", dict(report["team_type_dist"]))
    print("balls_per_over distribution:", dict(report["balls_per_over_dist"]))
    print("year distribution:", dict(sorted(report["year_dist"].items())))

    print("\nvenue count (distinct raw strings):", report["venue_count"])
    print("suspected venue duplicate groups (same prefix before comma):")
    for group in report["venue_suspected_dupes"][:40]:
        print("  ", group)

    print("\ndata_version distribution:", dict(report["data_version_dist"]))
    print("revision distribution:", dict(report["revision_dist"]))
    print("info optional-field presence counts (out of", report["n_files"], "):", dict(report["info_field_presence"]))
    print("season format:", dict(report["season_format"]))
    print("innings optional-field presence counts:", dict(report["innings_field_presence"]))
    print("powerplay types:", dict(report["powerplay_types"]))
    print("wicket kinds:", dict(report["wicket_kinds"].most_common()))
    print("delivery extras types seen:", dict(report["extras_types"]))
    print("fielder entry shapes:", dict(report["fielder_shapes"]))

    print("\nfiles without registry.people:", report["files_without_registry"])
    print("names mapping to >1 player id:", len(report["names_with_multiple_ids"]))
    for name, ids in list(report["names_with_multiple_ids"].items())[:20]:
        print("  ", name, "->", ids)
    print("ids mapping to >1 name:", len(report["ids_with_multiple_names"]))
    for pid, names in list(report["ids_with_multiple_names"].items())[:20]:
        print("  ", pid, "->", names)


if __name__ == "__main__":
    inventory()
    sample = sample_files(10)
    describe_structure(sample)
    report = scan_all_matches()
    print_scan_report(report)
