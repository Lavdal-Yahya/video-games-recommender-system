"""Phase 0 — quick look at the raw Steam dataset.

Loads each raw file, prints its shape + columns, prints a couple of sample
rows, and confirms that:
  1. games_metadata.json has tags + description (the content arm needs this),
  2. recommendations.csv has user-game interactions (the CF arm needs this),
  3. all three sources share the `app_id` key so we can join cleanly.

This script is read-only — it doesn't write any artifacts. It exists so a
human can eyeball the data once before Phase 1 starts cleaning it.
"""

from pathlib import Path
import json

import pandas as pd


RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def header(label: str) -> None:
    print("\n" + "=" * 72)
    print(label)
    print("=" * 72)


def peek_games_csv() -> pd.DataFrame:
    """games.csv — one row per game (title, price, rating, release date, …)."""
    header("games.csv")
    df = pd.read_csv(RAW / "games.csv")
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print("dtypes:")
    print(df.dtypes)
    print("\nsample rows:")
    print(df.head(3).to_string())
    return df


def peek_games_metadata_json() -> pd.DataFrame:
    """games_metadata.json — tags + description per app_id (content arm fuel).

    The Kaggle file is JSON-lines (one JSON object per line), not a single JSON
    array. pandas.read_json with lines=True handles that.
    """
    header("games_metadata.json")
    df = pd.read_json(RAW / "games_metadata.json", lines=True)
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print("dtypes:")
    print(df.dtypes)
    print("\nsample rows (tags truncated):")
    for _, row in df.head(3).iterrows():
        tags = row.get("tags")
        desc = row.get("description")
        if isinstance(desc, str) and len(desc) > 140:
            desc = desc[:140] + "…"
        print(
            f"  app_id={row.get('app_id')}  "
            f"tags={tags[:6] if isinstance(tags, list) else tags}  "
            f"description={desc!r}"
        )
    return df


def peek_recommendations_csv() -> pd.DataFrame:
    """recommendations.csv — user-game interactions (CF arm fuel).

    This is the big one (~2 GB). We only need the head to confirm the schema,
    so we read a small nrows sample instead of the whole file.
    """
    header("recommendations.csv (nrows=200_000 sample for inspection)")
    df = pd.read_csv(RAW / "recommendations.csv", nrows=200_000)
    print("shape (sample):", df.shape)
    print("columns:", list(df.columns))
    print("dtypes:")
    print(df.dtypes)
    print("\nsample rows:")
    print(df.head(5).to_string())
    return df


def peek_users_csv() -> pd.DataFrame:
    """users.csv — optional per CLAUDE/project.md; just confirm the schema."""
    header("users.csv")
    df = pd.read_csv(RAW / "users.csv", nrows=10)
    print("columns:", list(df.columns))
    print("\nsample rows:")
    print(df.head(5).to_string())
    return df


def confirm_signals_and_key(
    games: pd.DataFrame,
    metadata: pd.DataFrame,
    recs: pd.DataFrame,
) -> None:
    """Confirm the two signals + the shared join key."""
    header("Confirmations")

    # 1. content signal — tags + description present in metadata
    has_tags = "tags" in metadata.columns
    has_desc = "description" in metadata.columns
    n_with_tags = int(metadata["tags"].apply(lambda x: bool(x)).sum()) if has_tags else 0
    n_with_desc = (
        int(metadata["description"].apply(lambda x: isinstance(x, str) and bool(x.strip())).sum())
        if has_desc
        else 0
    )
    print(f"[content arm] tags column present: {has_tags}  "
          f"({n_with_tags}/{len(metadata)} rows non-empty)")
    print(f"[content arm] description column present: {has_desc}  "
          f"({n_with_desc}/{len(metadata)} rows non-empty)")

    # 2. interaction signal — recommendations has user_id + app_id
    has_user = "user_id" in recs.columns
    has_app = "app_id" in recs.columns
    print(f"[CF arm] user_id column present: {has_user}")
    print(f"[CF arm] app_id column present: {has_app}")
    if "is_recommended" in recs.columns:
        print(f"[CF arm] is_recommended dtype: {recs['is_recommended'].dtype}")
    if "hours" in recs.columns:
        print(f"[CF arm] hours present (could weight CF later) — "
              f"min/median/max in sample: "
              f"{recs['hours'].min():.1f} / {recs['hours'].median():.1f} / "
              f"{recs['hours'].max():.1f}")

    # 3. shared key — app_id in all three
    key = "app_id"
    in_games = key in games.columns
    in_meta = key in metadata.columns
    in_recs = key in recs.columns
    print(f"[join key] '{key}' in games.csv: {in_games}")
    print(f"[join key] '{key}' in games_metadata.json: {in_meta}")
    print(f"[join key] '{key}' in recommendations.csv: {in_recs}")

    # Quick overlap check: how many metadata app_ids match a games.csv app_id?
    if in_games and in_meta:
        overlap = len(set(metadata[key]).intersection(set(games[key])))
        print(f"[join key] metadata ∩ games on app_id: {overlap} games "
              f"(of {len(metadata)} metadata rows, {len(games)} games rows)")
    if in_games and in_recs:
        overlap_r = len(set(recs[key].unique()).intersection(set(games[key])))
        print(f"[join key] recommendations(sample) ∩ games on app_id: {overlap_r} games "
              f"(of {recs[key].nunique()} distinct app_ids in the sample)")


def main() -> None:
    print(f"Reading from: {RAW}")
    games = peek_games_csv()
    metadata = peek_games_metadata_json()
    recs = peek_recommendations_csv()
    peek_users_csv()
    confirm_signals_and_key(games, metadata, recs)
    print("\nDone.")


if __name__ == "__main__":
    main()
