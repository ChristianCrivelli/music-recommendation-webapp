"""
Album Recommendations API
--------------------------
Wraps the existing Supabase-backed feature matrix + cosine-similarity
recommender (from ChristianCrivelli/Album-Recommendations) in a small
FastAPI service anyone can hit over HTTP.

Read-only by design: this service should be configured with a Supabase
key that only has SELECT access on albums / album_tags / tags /
album_contributions / artists. Ingestion (adding new albums) stays a
separate, local, write-key-only workflow — never exposed here.
"""

import os
import time
import difflib
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")  # must be a READ-ONLY key
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "*")
ADMIN_REFRESH_TOKEN = os.environ.get("ADMIN_REFRESH_TOKEN")  # optional
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", 3600))  # 1 hour default

app = FastAPI(title="Album Recommendations API")


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc: RuntimeError):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# Catch-all for anything not already handled above (e.g. a dependency-version
# break in fetch_data()/build_feature_matrix()). Without this, an unhandled
# exception surfaces to the client as a bare 500 with no detail, and the only
# way to diagnose it is guessing — this at least logs the real traceback
# server-side so it shows up in Render's log stream.
@app.exception_handler(Exception)
async def unhandled_error_handler(request, exc: Exception):
    import logging
    import traceback
    from fastapi.responses import JSONResponse
    logging.error("Unhandled error on %s: %s", request.url.path, traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN] if FRONTEND_ORIGIN != "*" else ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── In-memory cache ──────────────────────────────────────────────────────────
_cache = {
    "df": None,
    "feature_matrix": None,
    "built_at": 0.0,
}


def paginate(query_builder_fn, page_size: int = 1000):
    """
    Runs a Supabase select in pages of `page_size` rows via .range(), and
    returns the concatenated list of all rows. Supabase/PostgREST caps
    unpaginated selects at 1000 rows by default — without this, any table
    over 1000 rows gets silently truncated.

    query_builder_fn: a zero-arg function that returns a FRESH Supabase
    query builder each call (not a response), e.g.:
        lambda: supabase.table("albums").select("id, title")
    A fresh builder is needed each loop iteration because .range() must be
    applied to an unexecuted query, and builders are single-use once
    .execute() is called.
    """
    all_rows = []
    start = 0
    while True:
        resp = query_builder_fn().range(start, start + page_size - 1).execute()
        rows = resp.data or []
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        start += page_size
    return all_rows


def fetch_data() -> pd.DataFrame:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL / SUPABASE_KEY are not set")

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    albums_data = paginate(lambda: supabase.table("albums").select(
        "id, title, mbid, release_year, avg_length, rating, notion_created_at, notion_edited_at, "
        "spotify_album_id, spotify_cover_url"
    ))
    albums_df = pd.DataFrame(albums_data)

    if albums_df.empty:
        raise RuntimeError(
            "Supabase returned zero rows for 'albums'. This usually means "
            "Row Level Security is enabled without a SELECT policy for the "
            "key in use — check Supabase RLS policies on albums / album_tags "
            "/ tags / album_contributions / artists."
        )

    tags_data = paginate(lambda: supabase.table("album_tags").select("album_id, tags(name)"))
    tag_rows = [{"album_id": r["album_id"], "tag": r["tags"]["name"]} for r in tags_data]
    tags_df = (
        pd.DataFrame(tag_rows)
        .groupby("album_id")["tag"]
        .apply(list)
        .reset_index()
        .rename(columns={"tag": "tags"})
        if tag_rows else pd.DataFrame(columns=["album_id", "tags"])
    )

    artists_data = paginate(
        lambda: supabase.table("album_contributions")
        .select("album_id, artists(id, mbid, name)")
        .eq("role", "artist")
    )
    artist_rows = [
        {
            "album_id": r["album_id"],
            # Manually-entered artists (see manual_overrides / README's
            # "fully manual entry" mode) can lack an mbid entirely. 60 of the
            # 2053 artist-role contributions currently do. Falling back to
            # the artist's own row id keeps every album's artist list free
            # of None — mixing None into this column crashes
            # MultiLabelBinarizer's internal sort in build_feature_matrix()
            # (that's the "'<' not supported between instances of
            # 'NoneType' and 'str'" 500 on every endpoint) — while still
            # giving the recommender a stable, per-artist identifier so
            # albums by the same mbid-less artist still cluster as a match.
            "artist_mbid": r["artists"]["mbid"] or f"noMbid:{r['artists']['id']}",
            "artist_name": r["artists"]["name"],
        }
        for r in artists_data
    ]
    artists_flat_df = pd.DataFrame(artist_rows)
    if not artists_flat_df.empty:
        artists_df = (
            artists_flat_df.groupby("album_id")["artist_mbid"]
            .apply(list)
            .reset_index()
            .rename(columns={"artist_mbid": "artist_mbids"})
        )
        artist_names_df = (
            artists_flat_df.groupby("album_id")["artist_name"]
            .apply(lambda names: ", ".join(sorted(set(names))))
            .reset_index()
            .rename(columns={"artist_name": "artist_names"})
        )
        artist_credits_df = (
            artists_flat_df.groupby("album_id")
            .apply(lambda g: [{"mbid": m, "name": n} for m, n in zip(g["artist_mbid"], g["artist_name"])])
            .reset_index(name="artist_credits")
        )
    else:
        artists_df = pd.DataFrame(columns=["album_id", "artist_mbids"])
        artist_names_df = pd.DataFrame(columns=["album_id", "artist_names"])
        artist_credits_df = pd.DataFrame(columns=["album_id", "artist_credits"])

    # Producers are collected during ingestion but were previously unused by
    # the recommender. Two albums by different artists but the same producer
    # is often a genuinely useful "sounds/feels similar" signal — catches
    # style similarity that pure genre tags miss. Names (not just mbids) are
    # pulled too so /api/browse and /api/facets can offer "search by
    # producer" the same way they do for tags and artists.
    producers_data = paginate(
        lambda: supabase.table("album_contributions")
        .select("album_id, artists(mbid, name)")
        .eq("role", "producer")
    )
    producer_rows = [
        {
            "album_id": r["album_id"],
            "producer_mbid": r["artists"]["mbid"],
            "producer_name": r["artists"].get("name"),
        }
        for r in producers_data
        if r.get("artists") and r["artists"].get("mbid")
    ]
    producers_df = (
        pd.DataFrame(producer_rows)
        .groupby("album_id")["producer_mbid"]
        .apply(list)
        .reset_index()
        .rename(columns={"producer_mbid": "producer_mbids"})
        if producer_rows else pd.DataFrame(columns=["album_id", "producer_mbids"])
    )
    producer_names_df = (
        pd.DataFrame(producer_rows)
        .dropna(subset=["producer_name"])
        .groupby("album_id")["producer_name"]
        .apply(list)
        .reset_index()
        .rename(columns={"producer_name": "producer_names"})
        if producer_rows else pd.DataFrame(columns=["album_id", "producer_names"])
    )

    df = albums_df.merge(tags_df, left_on="id", right_on="album_id", how="left").drop(columns="album_id", errors="ignore")
    df = df.merge(artists_df, left_on="id", right_on="album_id", how="left").drop(columns="album_id", errors="ignore")
    df = df.merge(artist_names_df, left_on="id", right_on="album_id", how="left").drop(columns="album_id", errors="ignore")
    df = df.merge(artist_credits_df, left_on="id", right_on="album_id", how="left").drop(columns="album_id", errors="ignore")
    df = df.merge(producers_df, left_on="id", right_on="album_id", how="left").drop(columns="album_id", errors="ignore")
    df = df.merge(producer_names_df, left_on="id", right_on="album_id", how="left").drop(columns="album_id", errors="ignore")

    df["tags"] = df["tags"].apply(lambda x: x if isinstance(x, list) else [])
    df["artist_mbids"] = df["artist_mbids"].apply(lambda x: x if isinstance(x, list) else [])
    df["artist_names"] = df["artist_names"].fillna("Unknown artist")
    df["artist_credits"] = df["artist_credits"].apply(lambda x: x if isinstance(x, list) else [])
    df["producer_mbids"] = df["producer_mbids"].apply(lambda x: x if isinstance(x, list) else [])
    df["producer_names"] = df["producer_names"].apply(lambda x: x if isinstance(x, list) else [])

    # Defensive strip: some titles were stored with stray leading/trailing
    # whitespace (fixed at the ingestion source now, but this covers rows
    # already in the DB). Without this, /api/recommend's exact-match lookup
    # can silently fail for a title that still shows up fine in autocomplete,
    # since autocomplete just echoes back whatever string is stored.
    df["title"] = df["title"].astype(str).str.strip()

    df["created_at"] = pd.to_datetime(df["notion_created_at"], errors="coerce", utc=True)
    df["updated_at"] = pd.to_datetime(df["notion_edited_at"], errors="coerce", utc=True)

    return df

def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    mlb = MultiLabelBinarizer()
    tag_matrix_binary = mlb.fit_transform(df["tags"])

    # Rarity weighting (TF-IDF-style): a tag shared by half the library (e.g.
    # "hip-hop") should count for much less than a niche tag two albums
    # happen to share. Without this, common tags dilute genuinely
    # distinctive matches.
    n_albums = max(tag_matrix_binary.shape[0], 1)
    tag_doc_freq = tag_matrix_binary.sum(axis=0)
    tag_idf = np.log((n_albums + 1) / (tag_doc_freq + 1)) + 1  # smoothed, always positive
    tag_matrix = tag_matrix_binary * tag_idf

    numeric = df[["release_year", "avg_length"]].apply(pd.to_numeric, errors="coerce").fillna(0)
    scaler = MinMaxScaler()
    numeric_matrix = scaler.fit_transform(numeric)

    artist_mlb = MultiLabelBinarizer()
    artist_matrix = artist_mlb.fit_transform(df["artist_mbids"])

    # Producers are a secondary "sounds/feels similar" signal — the same
    # producer working with two different artists is often a genuinely
    # useful cross-reference that genre tags alone miss.
    producer_mlb = MultiLabelBinarizer()
    producer_matrix = producer_mlb.fit_transform(df["producer_mbids"])

    # Artist weight lowered from 1.5 → 1.0: it was previously strong enough,
    # combined with sparse/missing tags, to let a bare "same artist" match
    # dominate a comparison and read as a near-100% match. It's still a
    # useful signal (see the confidence discount in /api/recommend, which
    # handles the sparse-tag case directly), just no longer the loudest one.
    TAG_WEIGHT = 2.0
    ARTIST_WEIGHT = 1.0
    PRODUCER_WEIGHT = 1.0
    NUMERIC_WEIGHT = 0.5

    return np.hstack([
        tag_matrix * TAG_WEIGHT,
        artist_matrix * ARTIST_WEIGHT,
        producer_matrix * PRODUCER_WEIGHT,
        numeric_matrix * NUMERIC_WEIGHT,
    ])


def get_cache(force: bool = False):
    stale = (time.time() - _cache["built_at"]) > CACHE_TTL_SECONDS
    if force or stale or _cache["df"] is None:
        df = fetch_data()
        feature_matrix = build_feature_matrix(df)
        _cache["df"] = df
        _cache["feature_matrix"] = feature_matrix
        _cache["built_at"] = time.time()
    return _cache["df"], _cache["feature_matrix"]


# ── API models ───────────────────────────────────────────────────────────────

def cover_art_url(mbid: Optional[str]) -> Optional[str]:
    """Cover Art Archive serves images by MBID — no API key, no extra call.
    albums.mbid now stores the release-GROUP id (see the ingestion-side fix
    that stopped it from storing a specific, non-stable release id — that
    was the root cause of the mass album-duplication bug), so this hits the
    release-group endpoint rather than /release/. The URL may still 404 if
    no release in the group was ever scanned in; the frontend falls back to
    a placeholder sleeve in that case."""
    if not mbid:
        return None
    return f"https://coverartarchive.org/release-group/{mbid}/front-500"


def resolve_cover_url(row) -> Optional[str]:
    """Prefer the ingestion-precomputed Spotify fallback (issue #11) — set
    only when Cover Art Archive was probed at ingestion time and found to
    404 for this mbid — over the live-constructed CAA URL. Keeps this
    backend from ever needing its own Spotify credentials or a live API
    call on the request path; the ingestion pipeline already did the work."""
    return row.get("spotify_cover_url") or cover_art_url(row.get("mbid"))


def spotify_embed_url(row) -> Optional[str]:
    """Issue #12: the frontend embeds this directly (an <iframe> with no
    API key needed) to play an in-app preview. None when this album has no
    Spotify match on file — the frontend falls back to no preview."""
    album_id = row.get("spotify_album_id")
    return f"https://open.spotify.com/embed/album/{album_id}" if album_id else None


class Recommendation(BaseModel):
    title: str
    artist_names: str
    release_year: Optional[str] = None
    avg_length: Optional[float] = None
    rating: Optional[float] = None
    tags: list[str] = []
    similarity: float
    cover_url: Optional[str] = None
    spotify_embed_url: Optional[str] = None


class MatchedAlbum(BaseModel):
    """The album the person actually searched for — shown directly under the
    search bar, separately from the list of recommended cross-references."""
    title: str
    artist_names: str
    release_year: Optional[str] = None
    rating: Optional[float] = None
    tags: list[str] = []
    cover_url: Optional[str] = None
    spotify_embed_url: Optional[str] = None


class TitleMatch(BaseModel):
    """One of possibly several albums sharing an exact title — used to
    disambiguate when title alone isn't a unique key (two different artists
    can and do release albums with the same name)."""
    title: str
    artist_names: str


class RecommendResponse(BaseModel):
    query: str
    matched_title: Optional[str] = None
    matched_album: Optional[MatchedAlbum] = None
    suggestions: list[str] = []
    # Populated when `title` matched more than one album and `artist` either
    # wasn't provided or didn't narrow it down to exactly one — lets the
    # frontend show "which one did you mean?" instead of silently guessing.
    other_matches: list[TitleMatch] = []
    results: list[Recommendation] = []


class RecentAlbum(BaseModel):
    title: str
    artist_names: str
    release_year: Optional[str] = None
    rating: Optional[float] = None
    tags: list[str] = []
    cover_url: Optional[str] = None
    spotify_embed_url: Optional[str] = None
    added_at: Optional[str] = None
    edited: bool = False
    edited_at: Optional[str] = None


class RecentResponse(BaseModel):
    albums: list[RecentAlbum] = []


class LovedAlbum(BaseModel):
    title: str
    artist_names: str
    rating: float
    release_year: Optional[str] = None
    cover_url: Optional[str] = None


class LovedArtist(BaseModel):
    name: str
    album_count: int
    avg_rating: float
    weighted_score: float


class GenreQuality(BaseModel):
    tag: str
    album_count: int
    avg_rating: float
    weighted_score: float


class GenreFrequency(BaseModel):
    tag: str
    album_count: int


class RatingBucket(BaseModel):
    rating: int
    count: int


class DecadeStat(BaseModel):
    decade: str
    album_count: int
    avg_rating: Optional[float] = None


class StatsResponse(BaseModel):
    total_albums: int
    rated_albums: int
    untagged_albums: int
    unique_artists: int
    unique_tags: int
    avg_rating: Optional[float] = None
    most_loved_albums: list[LovedAlbum] = []
    most_loved_artists: list[LovedArtist] = []
    top_genres_by_quality: list[GenreQuality] = []
    top_genres_by_frequency: list[GenreFrequency] = []
    rating_distribution: list[RatingBucket] = []
    ratings_by_decade: list[DecadeStat] = []


class FacetsResponse(BaseModel):
    """Distinct tag / artist / producer names currently in the library, for
    populating the Browse tab's filter controls (issue #3: search by
    Genre/Tag/Artist/Producer instead of only exact-title search)."""
    tags: list[str] = []
    artists: list[str] = []
    producers: list[str] = []


class BrowseAlbum(BaseModel):
    title: str
    artist_names: str
    release_year: Optional[str] = None
    rating: Optional[float] = None
    tags: list[str] = []
    cover_url: Optional[str] = None
    spotify_embed_url: Optional[str] = None


class BrowseResponse(BaseModel):
    tag: Optional[str] = None
    artist: Optional[str] = None
    producer: Optional[str] = None
    total: int = 0
    offset: int = 0
    # Issue #14's "pagination fix": previously the only way to see more of
    # a large result set was raising `n` up to a hard 200-row ceiling —
    # anything past that was permanently unreachable, no error, no signal.
    # has_more tells the frontend whether a "Load more" fetch (same filters,
    # offset advanced by however many rows came back) would return anything.
    has_more: bool = False
    albums: list[BrowseAlbum] = []


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


class AlbumSummary(BaseModel):
    title: str
    artist_names: str


class AlbumsResponse(BaseModel):
    # Kept for backward compatibility with any existing frontend build —
    # plain title strings, deduplicated. Ambiguous for title collisions.
    titles: list[str] = []
    # Preferred going forward: title+artist pairs, so a frontend can tell
    # two same-titled albums apart in autocomplete instead of only ever
    # being able to reach whichever one the backend happens to pick first.
    albums: list[AlbumSummary] = []


@app.get("/api/albums", response_model=AlbumsResponse)
def list_albums():
    """Returns every album currently in the library (for autocomplete),
    as both a flat title list (legacy) and title+artist pairs (preferred)."""
    df, _ = get_cache()
    dedup = (
        df[["title", "artist_names"]]
        .fillna({"artist_names": "Unknown artist"})
        .drop_duplicates()
        .sort_values(["title", "artist_names"])
    )
    return AlbumsResponse(
        titles=sorted(df["title"].dropna().unique().tolist()),
        albums=[
            AlbumSummary(title=r["title"], artist_names=r["artist_names"])
            for _, r in dedup.iterrows()
        ],
    )


@app.get("/api/facets", response_model=FacetsResponse)
def facets():
    """Every distinct tag, artist, and producer name in the library — lets
    the Browse tab populate its filter controls without the frontend having
    to guess at what's available."""
    df, _ = get_cache()

    tags = sorted({t for tag_list in df["tags"] for t in tag_list})

    artist_names = set()
    for credits in df["artist_credits"]:
        for c in credits:
            name = c.get("name")
            if name:
                artist_names.add(name)

    producer_names = {name for names in df["producer_names"] for name in names}

    return FacetsResponse(
        tags=tags,
        artists=sorted(artist_names),
        producers=sorted(producer_names),
    )


@app.get("/api/browse", response_model=BrowseResponse)
def browse(
    tag: Optional[str] = None,
    artist: Optional[str] = None,
    producer: Optional[str] = None,
    n: int = 60,
    offset: int = 0,
):
    """Browse the library by genre/tag, artist, or producer instead of only
    by exact title (issue #3). Filters combine with AND when more than one
    is given. tag matches exactly (case-insensitive) against the controlled
    tag vocabulary from /api/facets; artist/producer match as a
    case-insensitive substring, same as /api/recommend's artist narrowing.

    Issue #14: `offset` pages through a result set larger than one page
    (still capped at 200 rows per request) instead of `n` alone silently
    truncating anything past its ceiling with no way to reach the rest."""
    if not (tag or artist or producer):
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of tag, artist, or producer.",
        )

    df, _ = get_cache()
    working = df

    if tag:
        tag_query = tag.strip().lower()
        working = working[
            working["tags"].apply(lambda ts: tag_query in {t.lower() for t in ts})
        ]

    if artist:
        artist_query = artist.strip().lower()
        working = working[
            working["artist_names"].str.lower().str.contains(artist_query, na=False, regex=False)
        ]

    if producer:
        producer_query = producer.strip().lower()
        working = working[
            working["producer_names"].apply(
                lambda names: any(producer_query in p.lower() for p in names)
            )
        ]

    total = len(working)
    working = working.sort_values("rating", ascending=False, na_position="last")

    page_size = min(max(n, 1), 200)
    offset = max(offset, 0)
    page = working.iloc[offset:offset + page_size]

    albums = [
        BrowseAlbum(
            title=row["title"],
            artist_names=row.get("artist_names", "Unknown artist"),
            release_year=str(row["release_year"]) if pd.notna(row["release_year"]) else None,
            rating=float(row["rating"]) if pd.notna(row["rating"]) else None,
            tags=row["tags"],
            cover_url=resolve_cover_url(row),
            spotify_embed_url=spotify_embed_url(row),
        )
        for _, row in page.iterrows()
    ]

    return BrowseResponse(
        tag=tag,
        artist=artist,
        producer=producer,
        total=total,
        offset=offset,
        has_more=(offset + len(albums)) < total,
        albums=albums,
    )


@app.get("/api/recommend", response_model=RecommendResponse)
def recommend(title: str, n: int = 5, artist: Optional[str] = None):
    df, feature_matrix = get_cache()

    matches = df[df["title"].str.lower() == title.strip().lower()]

    if matches.empty:
        # The query might be an artist's name rather than an album title (e.g.
        # searching "Birdman" the person) - matching only against df["title"]
        # otherwise falls straight through to difflib's close-title matches,
        # which produces meaningless "Did you mean" suggestions for a query
        # that was never a title to begin with (see issue #23). Try the
        # artist column first, the same substring match /api/browse already
        # uses for its artist filter.
        query_lower = title.strip().lower()
        artist_matches = df[df["artist_names"].str.lower().str.contains(query_lower, na=False, regex=False)]
        if not artist_matches.empty:
            artist_titles = artist_matches["title"].tolist()[:5]
            return RecommendResponse(query=title, suggestions=artist_titles, results=[])

        close = difflib.get_close_matches(title, df["title"].tolist(), n=5, cutoff=0.4)
        return RecommendResponse(query=title, suggestions=close, results=[])

    # Title alone isn't guaranteed unique — two different albums can share a
    # name. If more than one album matched, use the artist hint (when given)
    # to narrow it down; if that still leaves more than one candidate (or no
    # hint was given at all), don't guess — hand back the list of candidates
    # so the frontend can ask which one was meant.
    if len(matches) > 1 and artist:
        artist_query = artist.strip().lower()
        narrowed = matches[matches["artist_names"].str.lower().str.contains(artist_query, na=False, regex=False)]
        if not narrowed.empty:
            matches = narrowed

    if len(matches) > 1:
        other_matches = [
            TitleMatch(title=row["title"], artist_names=row.get("artist_names", "Unknown artist"))
            for _, row in matches.iterrows()
        ]
        return RecommendResponse(query=title, matched_title=title, other_matches=other_matches, results=[])

    idx = matches.index[0]
    matched_row = matches.iloc[0]
    matched_album = MatchedAlbum(
        title=matched_row["title"],
        artist_names=matched_row.get("artist_names", "Unknown artist"),
        release_year=str(matched_row["release_year"]) if pd.notna(matched_row["release_year"]) else None,
        rating=float(matched_row["rating"]) if pd.notna(matched_row["rating"]) else None,
        tags=matched_row["tags"],
        cover_url=resolve_cover_url(matched_row),
        spotify_embed_url=spotify_embed_url(matched_row),
    )

    sim_scores = cosine_similarity([feature_matrix[idx]], feature_matrix)[0]

    results = df.copy()
    results["similarity"] = sim_scores

    # Confidence discount: cosine similarity can't tell "genuinely similar
    # across several shared tags" apart from "shares an artist but has
    # little/no tag data to compare" — if both albums have empty tags, that
    # segment contributes nothing to the comparison either way, so a bare
    # artist match can still score ~1.0. Discount any candidate that shares
    # fewer than MIN_CONFIDENT_TAG_OVERLAP tags with the searched album, so
    # a same-artist-no-data match can't display as a "100% match" the way a
    # real cross-genre overlap would.
    TAG_CONFIDENCE_FLOOR = 0.4
    MIN_CONFIDENT_TAG_OVERLAP = 2
    query_tags = set(matched_row["tags"])
    tag_overlap = results["tags"].apply(lambda t: len(query_tags & set(t)))
    confidence = TAG_CONFIDENCE_FLOOR + (1 - TAG_CONFIDENCE_FLOOR) * (
        tag_overlap.clip(upper=MIN_CONFIDENT_TAG_OVERLAP) / MIN_CONFIDENT_TAG_OVERLAP
    )
    results["similarity"] = results["similarity"] * confidence

    results = (
        results[results.index != idx]
        .sort_values("similarity", ascending=False)
        .head(min(max(n, 1), 20))
    )

    recs = [
        Recommendation(
            title=row["title"],
            artist_names=row.get("artist_names", "Unknown artist"),
            release_year=str(row["release_year"]) if pd.notna(row["release_year"]) else None,
            avg_length=float(row["avg_length"]) if pd.notna(row["avg_length"]) else None,
            rating=float(row["rating"]) if pd.notna(row["rating"]) else None,
            tags=row["tags"],
            similarity=round(float(row["similarity"]), 3),
            cover_url=resolve_cover_url(row),
            spotify_embed_url=spotify_embed_url(row),
        )
        for _, row in results.iterrows()
    ]

    return RecommendResponse(
        query=title,
        matched_title=matched_row["title"],
        matched_album=matched_album,
        results=recs,
    )


RECENT_WINDOW_DAYS = 14


@app.get("/api/recent", response_model=RecentResponse)
def recent():
    """Albums added/edited in Notion within the last RECENT_WINDOW_DAYS days,
    newest first. Based on Notion's own created_time/last_edited_time (see
    NotionCreatedAt/NotionEditedAt in pull_albums.py) rather than Supabase's
    created_at/updated_at, since the latter reflects pipeline processing
    time, not when the entry was actually touched in Notion."""
    df, _ = get_cache()

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=RECENT_WINDOW_DAYS)

    working = df.copy()
    working["_sort_ts"] = working[["created_at", "updated_at"]].max(axis=1)
    working = working[working["_sort_ts"].notna() & (working["_sort_ts"] >= cutoff)]
    working = working.sort_values("_sort_ts", ascending=False)

    albums = []
    for _, row in working.iterrows():
        created = row["created_at"]
        updated = row["updated_at"]
        edited = bool(pd.notna(created) and pd.notna(updated) and updated > created)
        albums.append(RecentAlbum(
            title=row["title"],
            artist_names=row.get("artist_names", "Unknown artist"),
            release_year=str(row["release_year"]) if pd.notna(row["release_year"]) else None,
            rating=float(row["rating"]) if pd.notna(row["rating"]) else None,
            tags=row["tags"],
            cover_url=resolve_cover_url(row),
            spotify_embed_url=spotify_embed_url(row),
            added_at=created.isoformat() if pd.notna(created) else None,
            edited=edited,
            edited_at=updated.isoformat() if edited else None,
        ))

    return RecentResponse(albums=albums)


# Damping constant for the weighted "most loved" scores below — same idea as
# IMDB's top-250 formula: weighted_score = (sum_ratings + C * global_mean) / (n + C).
# Pulls low-sample-size artists/genres toward the library-wide average until
# they've earned enough data to stand on their own, instead of letting one
# 10/10 album crown an artist or genre outright.
DAMPING_C = 3.0


@app.get("/api/stats", response_model=StatsResponse)
def stats():
    df, _ = get_cache()

    total_albums = len(df)
    untagged_albums = int((df["tags"].apply(len) == 0).sum())

    # Unique artists/tags computed over the full library, not just rated
    # albums, since these are structural counts, not quality measures.
    all_artist_keys = set()
    for credits in df["artist_credits"]:
        for c in credits:
            key = c.get("mbid") or c.get("name")
            if key:
                all_artist_keys.add(key)
    unique_artists = len(all_artist_keys)

    all_tags_flat = [t for tags in df["tags"] for t in tags]
    unique_tags = len(set(all_tags_flat))

    rated = df[df["rating"].apply(pd.notna)].copy()
    rated["rating"] = rated["rating"].astype(float)
    rated_albums = len(rated)
    global_mean = float(rated["rating"].mean()) if rated_albums else 0.0

    # --- Most loved albums: straightforward top-N, no weighting needed ---
    top_albums_df = rated.sort_values("rating", ascending=False).head(10)
    most_loved_albums = [
        LovedAlbum(
            title=row["title"],
            artist_names=row.get("artist_names", "Unknown artist"),
            rating=float(row["rating"]),
            release_year=str(row["release_year"]) if pd.notna(row["release_year"]) else None,
            cover_url=resolve_cover_url(row),
        )
        for _, row in top_albums_df.iterrows()
    ]

    # --- Most loved artists: damped mean over exploded per-artist credits ---
    # Explodes on mbid (falling back to name only if mbid is missing) rather
    # than splitting the display string on commas, since an artist name
    # could legitimately contain one.
    artist_rows = [
        {"key": c.get("mbid") or c.get("name"), "name": c.get("name", "Unknown artist"), "rating": row["rating"]}
        for _, row in rated.iterrows()
        for c in row["artist_credits"]
        if c.get("mbid") or c.get("name")
    ]
    most_loved_artists = []
    if artist_rows:
        artists_flat = pd.DataFrame(artist_rows)
        grouped = artists_flat.groupby("key").agg(
            name=("name", "first"),
            album_count=("rating", "count"),
            rating_sum=("rating", "sum"),
        ).reset_index()
        grouped["avg_rating"] = grouped["rating_sum"] / grouped["album_count"]
        grouped["weighted_score"] = (grouped["rating_sum"] + DAMPING_C * global_mean) / (grouped["album_count"] + DAMPING_C)
        top_artists_df = grouped.sort_values("weighted_score", ascending=False).head(10)
        most_loved_artists = [
            LovedArtist(
                name=r["name"],
                album_count=int(r["album_count"]),
                avg_rating=round(float(r["avg_rating"]), 2),
                weighted_score=round(float(r["weighted_score"]), 2),
            )
            for _, r in top_artists_df.iterrows()
        ]

    # --- Genres: quality (damped mean, rated albums only) vs frequency
    # (raw count, full library) — kept as two separate views rather than one
    # blended score, since conflating them lets a single outlier album make
    # a genre look "beloved" off one data point. ---
    tag_rows = [
        {"tag": tag, "rating": row["rating"]}
        for _, row in rated.iterrows()
        for tag in row["tags"]
    ]
    top_genres_by_quality = []
    if tag_rows:
        tags_flat = pd.DataFrame(tag_rows)
        tag_grouped = tags_flat.groupby("tag").agg(
            album_count=("rating", "count"),
            rating_sum=("rating", "sum"),
        ).reset_index()
        tag_grouped["avg_rating"] = tag_grouped["rating_sum"] / tag_grouped["album_count"]
        tag_grouped["weighted_score"] = (tag_grouped["rating_sum"] + DAMPING_C * global_mean) / (tag_grouped["album_count"] + DAMPING_C)
        top_quality_df = tag_grouped.sort_values("weighted_score", ascending=False).head(10)
        top_genres_by_quality = [
            GenreQuality(
                tag=r["tag"],
                album_count=int(r["album_count"]),
                avg_rating=round(float(r["avg_rating"]), 2),
                weighted_score=round(float(r["weighted_score"]), 2),
            )
            for _, r in top_quality_df.iterrows()
        ]

    top_genres_by_frequency = []
    if all_tags_flat:
        freq_series = pd.Series(all_tags_flat).value_counts().head(10)
        top_genres_by_frequency = [
            GenreFrequency(tag=t, album_count=int(c)) for t, c in freq_series.items()
        ]

    # --- Rating distribution (1-10 histogram) ---
    rating_distribution = []
    if rated_albums:
        buckets = rated["rating"].round().astype(int).clip(lower=1, upper=10)
        bucket_counts = buckets.value_counts().sort_index()
        rating_distribution = [
            RatingBucket(rating=int(r), count=int(c)) for r, c in bucket_counts.items()
        ]

    # --- Ratings by decade ---
    ratings_by_decade = []
    with_year = rated.copy()
    with_year["release_year_num"] = pd.to_numeric(with_year["release_year"], errors="coerce")
    with_year = with_year.dropna(subset=["release_year_num"])
    if not with_year.empty:
        with_year["decade"] = (with_year["release_year_num"] // 10 * 10).astype(int).astype(str) + "s"
        decade_grouped = with_year.groupby("decade").agg(
            album_count=("rating", "count"),
            avg_rating=("rating", "mean"),
        ).reset_index().sort_values("decade")
        ratings_by_decade = [
            DecadeStat(
                decade=r["decade"],
                album_count=int(r["album_count"]),
                avg_rating=round(float(r["avg_rating"]), 2),
            )
            for _, r in decade_grouped.iterrows()
        ]

    return StatsResponse(
        total_albums=total_albums,
        rated_albums=rated_albums,
        untagged_albums=untagged_albums,
        unique_artists=unique_artists,
        unique_tags=unique_tags,
        avg_rating=round(global_mean, 2) if rated_albums else None,
        most_loved_albums=most_loved_albums,
        most_loved_artists=most_loved_artists,
        top_genres_by_quality=top_genres_by_quality,
        top_genres_by_frequency=top_genres_by_frequency,
        rating_distribution=rating_distribution,
        ratings_by_decade=ratings_by_decade,
    )


@app.post("/api/refresh")
def refresh(authorization: Optional[str] = Header(None)):
    """Force-rebuild the cached feature matrix (e.g. after adding new albums)."""
    if ADMIN_REFRESH_TOKEN:
        if authorization != f"Bearer {ADMIN_REFRESH_TOKEN}":
            raise HTTPException(status_code=401, detail="Invalid or missing token")
    get_cache(force=True)
    return {"status": "refreshed", "albums": len(_cache["df"])}