"""
Recommender evaluation harness.
---------------------------------
Pulls the live library and reports a few diagnostics on /api/recommend's
current scoring behavior, so a constants/weights/ensemble change can be
validated against a consistent yardstick instead of eyeballing a few
examples by hand.

This intentionally runs its OWN copy of the scoring logic (fetch_data,
build_feature_matrix, and the recommend() scoring steps) rather than
importing backend/main.py directly, because that logic currently lives
inline inside the /api/recommend route function, not as a standalone
callable. Once the scoring approach stabilizes (we're still actively
comparing weighted-cosine vs. rank-fusion ensembles as of 2026-11), the
natural next step is to extract it into backend/scoring/ and have both
main.py and this script import from there instead of this script
shadowing it — tracked informally for now, not yet its own issue.

Usage:
    python -m scripts.eval_recommender

Reports, split out by whether the query album has any tags (tag coverage
is the single biggest driver of recommendation quality in this library):
  - same_artist_top1_rate: how often the #1 recommendation is just
    another album by the same artist
  - zero_tag_overlap_top1_rate: how often the #1 recommendation shares
    no tags at all with the query (only meaningful for tagged queries -
    it's 100% by definition for untagged ones)
  - mean_similarity: average displayed similarity score for the #1 rec
  - mean_ild: mean intra-list diversity of each query's top-N list (1 -
    mean pairwise cosine similarity among the N recommended albums,
    TOP_N_FOR_DIVERSITY below) — low ILD means a query's results tend to
    be near-duplicates of each other, not just of the query itself.
Plus one number for the whole run, not split by tagged/untagged (it's a
property of the library's reachability, not of any one query):
  - catalog_coverage: the fraction of the library that shows up in ANY
    query's top-N list across this whole run — a rough proxy for "is the
    long tail reachable at all," or whether the same handful of albums
    just keep winning every query.

── Goal (Building Intelligent Systems ch. 4's three-part test — states
the outcome, is achievable, is measurable with what already exists) ──
Reduce zero_tag_overlap_top1_rate among tagged albums, and the untagged
fraction of the library, without making same_artist_top1_rate worse.
Tag coverage, not the scoring formula, is this recommender's identified
bottleneck (issue #23) — this goal is written against that bottleneck on
purpose, not against mean_similarity or any other number a scoring-
formula change could move without actually improving anything a person
would notice.

── Discipline, before changing any constant in this file or main.py ──
Run this script and write down the actual numbers FIRST. Change one
thing. Run it again and compare. Don't iterate by eye against a single
run — that's tuning to the test set (Building Intelligent Systems'
"Independent Evaluation Data"; the Handbook calls the same mistake
"tuning to the test set"), and this script's numbers are the only
feedback loop available for a single-user library with no held-out
ground truth, so protecting them from that mistake is the whole point.
"""
import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# ── Scoring constants — keep these in sync with backend/main.py by hand   ──
# ── until the extraction described above happens.                        ──
TAG_WEIGHT = 2.0
ARTIST_WEIGHT = 1.0
PRODUCER_WEIGHT = 1.0
NUMERIC_WEIGHT = 0.5
TAG_CONFIDENCE_FLOOR = 0.4
MIN_CONFIDENT_TAG_OVERLAP = 2
SAME_ARTIST_PENALTY = 0.6

# How many results to look at for the diversity/coverage metrics below —
# matches /api/recommend's own default `n`, since that's what a real
# visitor actually sees per query, not an arbitrary evaluation-only
# number.
TOP_N_FOR_DIVERSITY = 5


def paginate(query_builder_fn, page_size=1000):
    all_rows, start = [], 0
    while True:
        resp = query_builder_fn().range(start, start + page_size - 1).execute()
        rows = resp.data or []
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        start += page_size
    return all_rows


def fetch_data(supabase: Client) -> pd.DataFrame:
    albums_data = paginate(lambda: supabase.table("albums").select(
        "id, title, mbid, release_year, avg_length, rating"
    ))
    albums_df = pd.DataFrame(albums_data)

    tags_data = paginate(lambda: supabase.table("album_tags").select("album_id, tags(name)"))
    tag_rows = [{"album_id": r["album_id"], "tag": r["tags"]["name"]} for r in tags_data]
    tags_df = (
        pd.DataFrame(tag_rows).groupby("album_id")["tag"].apply(list).reset_index().rename(columns={"tag": "tags"})
        if tag_rows else pd.DataFrame(columns=["album_id", "tags"])
    )

    artists_data = paginate(
        lambda: supabase.table("album_contributions").select("album_id, artists(id, mbid, name)").eq("role", "artist")
    )
    artist_rows = [
        {
            "album_id": r["album_id"],
            "artist_mbid": r["artists"]["mbid"] or f"noMbid:{r['artists']['id']}",
            "artist_name": r["artists"]["name"],
        }
        for r in artists_data
    ]
    artists_flat_df = pd.DataFrame(artist_rows)
    artists_df = (
        artists_flat_df.groupby("album_id")["artist_mbid"].apply(list).reset_index().rename(columns={"artist_mbid": "artist_mbids"})
        if not artists_flat_df.empty else pd.DataFrame(columns=["album_id", "artist_mbids"])
    )

    producers_data = paginate(
        lambda: supabase.table("album_contributions").select("album_id, artists(mbid, name)").eq("role", "producer")
    )
    producer_rows = [
        {"album_id": r["album_id"], "producer_mbid": r["artists"]["mbid"]}
        for r in producers_data
        if r.get("artists") and r["artists"].get("mbid")
    ]
    producers_df = (
        pd.DataFrame(producer_rows).groupby("album_id")["producer_mbid"].apply(list).reset_index().rename(columns={"producer_mbid": "producer_mbids"})
        if producer_rows else pd.DataFrame(columns=["album_id", "producer_mbids"])
    )

    df = albums_df.merge(tags_df, left_on="id", right_on="album_id", how="left").drop(columns="album_id", errors="ignore")
    df = df.merge(artists_df, left_on="id", right_on="album_id", how="left").drop(columns="album_id", errors="ignore")
    df = df.merge(producers_df, left_on="id", right_on="album_id", how="left").drop(columns="album_id", errors="ignore")

    df["tags"] = df["tags"].apply(lambda x: x if isinstance(x, list) else [])
    df["artist_mbids"] = df["artist_mbids"].apply(lambda x: x if isinstance(x, list) else [])
    df["producer_mbids"] = df["producer_mbids"].apply(lambda x: x if isinstance(x, list) else [])
    return df.reset_index(drop=True)


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    mlb = MultiLabelBinarizer()
    tag_bin = mlb.fit_transform(df["tags"])
    n_albums = max(tag_bin.shape[0], 1)
    tag_idf = np.log((n_albums + 1) / (tag_bin.sum(axis=0) + 1)) + 1
    tag_matrix = tag_bin * tag_idf

    numeric = df[["release_year", "avg_length"]].apply(pd.to_numeric, errors="coerce").fillna(0)
    numeric_matrix = MinMaxScaler().fit_transform(numeric)

    artist_matrix = MultiLabelBinarizer().fit_transform(df["artist_mbids"])
    producer_matrix = MultiLabelBinarizer().fit_transform(df["producer_mbids"])

    return np.hstack([
        tag_matrix * TAG_WEIGHT,
        artist_matrix * ARTIST_WEIGHT,
        producer_matrix * PRODUCER_WEIGHT,
        numeric_matrix * NUMERIC_WEIGHT,
    ])


def _scored_results(df, feature_matrix, idx):
    """The same scoring steps /api/recommend applies (TF-IDF cosine
    similarity, tag-confidence discount, same-artist penalty), returning
    every OTHER album sorted best-first with similarity/tag_overlap/
    same_artist columns attached. Factored out of top1_recommendation so
    the diversity/coverage metrics below can look at more than just the
    #1 result without duplicating this logic a second time."""
    matched_row = df.loc[idx]
    sim_scores = cosine_similarity([feature_matrix[idx]], feature_matrix)[0]
    results = df.copy()
    results["similarity"] = sim_scores

    query_tags = set(matched_row["tags"])
    results["tag_overlap"] = results["tags"].apply(lambda t: len(query_tags & set(t)))
    confidence = TAG_CONFIDENCE_FLOOR + (1 - TAG_CONFIDENCE_FLOOR) * (
        results["tag_overlap"].clip(upper=MIN_CONFIDENT_TAG_OVERLAP) / MIN_CONFIDENT_TAG_OVERLAP
    )
    results["similarity"] *= confidence

    results["same_artist"] = results["artist_mbids"].apply(lambda a: bool(set(matched_row["artist_mbids"]) & set(a)))
    results.loc[results["same_artist"], "similarity"] *= SAME_ARTIST_PENALTY

    return results[results.index != idx].sort_values("similarity", ascending=False)


def top1_recommendation(df, feature_matrix, idx):
    top = _scored_results(df, feature_matrix, idx).iloc[0]
    return {
        "same_artist": bool(top["same_artist"]),
        "tag_overlap": int(top["tag_overlap"]),
        "similarity": float(top["similarity"]),
    }


def topn_indices(df, feature_matrix, idx, n=TOP_N_FOR_DIVERSITY):
    """The query's top-n result indices (into df/feature_matrix), same
    ordering /api/recommend would return. Used by diversity_and_coverage_
    report below — kept separate from top1_recommendation so a caller
    that only needs the #1 result (the common case) doesn't pay for or
    depend on the rest of the list."""
    return _scored_results(df, feature_matrix, idx).head(n).index.tolist()


def intra_list_diversity(feature_matrix, result_indices):
    """1 - mean pairwise cosine similarity among a result list's own
    feature vectors — the standard ILD metric (Recommender Systems
    Handbook's novelty/diversity chapter). 0.0 means every result in the
    list is identical to every other; higher means the list spans more
    ground rather than being five variations on one theme. Undefined
    (None) for a list of fewer than 2 albums — nothing to pair up."""
    if len(result_indices) < 2:
        return None
    vectors = feature_matrix[result_indices]
    sims = cosine_similarity(vectors)
    n = len(result_indices)
    # Mean of the OFF-diagonal entries only (upper triangle, k=1) — the
    # diagonal is each album against itself (similarity 1.0 by
    # definition) and would just dilute the real pairwise comparisons.
    off_diagonal = sims[np.triu_indices(n, k=1)]
    return 1.0 - float(off_diagonal.mean())


def run_report(df, feature_matrix):
    has_tags = df["tags"].apply(len) > 0

    rows = []
    all_recommended = set()
    for idx in df.index:
        row = top1_recommendation(df, feature_matrix, idx)
        result_indices = topn_indices(df, feature_matrix, idx)
        row["ild"] = intra_list_diversity(feature_matrix, result_indices)
        all_recommended.update(result_indices)
        rows.append(row)

    report = pd.DataFrame(rows)
    report["has_tags"] = has_tags.values

    def summarize(subset, label):
        if subset.empty:
            return
        print(f"  [{label}] n={len(subset)}")
        print(f"    same_artist_top1_rate:      {subset['same_artist'].mean():.1%}")
        print(f"    zero_tag_overlap_top1_rate: {(subset['tag_overlap'] == 0).mean():.1%}")
        print(f"    mean_similarity:            {subset['similarity'].mean():.3f}")
        ild_values = subset["ild"].dropna()
        if not ild_values.empty:
            print(f"    mean_ild (top-{TOP_N_FOR_DIVERSITY}):           {ild_values.mean():.3f}")

    print(f"Library: {len(df)} albums ({has_tags.sum()} tagged, {(~has_tags).sum()} untagged)\n")
    summarize(report, "all")
    summarize(report[report["has_tags"]], "tagged only")
    summarize(report[~report["has_tags"]], "untagged only")

    coverage = len(all_recommended) / len(df) if len(df) else 0.0
    print(f"\n  catalog_coverage (any query's top-{TOP_N_FOR_DIVERSITY}): {coverage:.1%} "
          f"({len(all_recommended)}/{len(df)} albums ever recommended)")


def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("SUPABASE_URL / SUPABASE_KEY not set (check your .env).", file=sys.stderr)
        sys.exit(1)
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    df = fetch_data(supabase)
    feature_matrix = build_feature_matrix(df)
    run_report(df, feature_matrix)


if __name__ == "__main__":
    main()
