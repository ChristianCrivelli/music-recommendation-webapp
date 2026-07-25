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


def fetch_data() -> pd.DataFrame:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL / SUPABASE_KEY are not set")

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    albums_resp = supabase.table("albums").select(
        "id, title, mbid, release_year, avg_length, rating, created_at, updated_at"
    ).execute()
    albums_df = pd.DataFrame(albums_resp.data)

    if albums_df.empty:
        raise RuntimeError(
            "Supabase returned zero rows for 'albums'. This usually means "
            "Row Level Security is enabled without a SELECT policy for the "
            "key in use — check Supabase RLS policies on albums / album_tags "
            "/ tags / album_contributions / artists."
        )

    tags_resp = (
        supabase.table("album_tags").select("album_id, tags(name)").execute()
    )
    tag_rows = [{"album_id": r["album_id"], "tag": r["tags"]["name"]} for r in tags_resp.data]
    tags_df = (
        pd.DataFrame(tag_rows)
        .groupby("album_id")["tag"]
        .apply(list)
        .reset_index()
        .rename(columns={"tag": "tags"})
        if tag_rows else pd.DataFrame(columns=["album_id", "tags"])
    )

    artists_resp = (
        supabase.table("album_contributions")
        .select("album_id, artists(mbid, name)")
        .eq("role", "artist")
        .execute()
    )
    artist_rows = [
        {
            "album_id": r["album_id"],
            "artist_mbid": r["artists"]["mbid"],
            "artist_name": r["artists"]["name"],
        }
        for r in artists_resp.data
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
    else:
        artists_df = pd.DataFrame(columns=["album_id", "artist_mbids"])
        artist_names_df = pd.DataFrame(columns=["album_id", "artist_names"])

    df = albums_df.merge(tags_df, left_on="id", right_on="album_id", how="left").drop(columns="album_id", errors="ignore")
    df = df.merge(artists_df, left_on="id", right_on="album_id", how="left").drop(columns="album_id", errors="ignore")
    df = df.merge(artist_names_df, left_on="id", right_on="album_id", how="left").drop(columns="album_id", errors="ignore")

    df["tags"] = df["tags"].apply(lambda x: x if isinstance(x, list) else [])
    df["artist_mbids"] = df["artist_mbids"].apply(lambda x: x if isinstance(x, list) else [])
    df["artist_names"] = df["artist_names"].fillna("Unknown artist")

    # Defensive strip: some titles were stored with stray leading/trailing
    # whitespace (fixed at the ingestion source now, but this covers rows
    # already in the DB). Without this, /api/recommend's exact-match lookup
    # can silently fail for a title that still shows up fine in autocomplete,
    # since autocomplete just echoes back whatever string is stored.
    df["title"] = df["title"].astype(str).str.strip()

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce", utc=True)

    return df


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    mlb = MultiLabelBinarizer()
    tag_matrix = mlb.fit_transform(df["tags"])

    numeric = df[["release_year", "avg_length"]].apply(pd.to_numeric, errors="coerce").fillna(0)
    scaler = MinMaxScaler()
    numeric_matrix = scaler.fit_transform(numeric)

    artist_mlb = MultiLabelBinarizer()
    artist_matrix = artist_mlb.fit_transform(df["artist_mbids"])

    TAG_WEIGHT = 2.0
    ARTIST_WEIGHT = 1.5
    NUMERIC_WEIGHT = 0.5

    return np.hstack([
        tag_matrix * TAG_WEIGHT,
        artist_matrix * ARTIST_WEIGHT,
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
    """Cover Art Archive serves images by release MBID — no API key, no extra call.
    The URL may 404 if that release was never scanned in; the frontend falls
    back to a placeholder sleeve in that case."""
    if not mbid:
        return None
    return f"https://coverartarchive.org/release/{mbid}/front-500"


class Recommendation(BaseModel):
    title: str
    artist_names: str
    release_year: Optional[str] = None
    avg_length: Optional[float] = None
    rating: Optional[float] = None
    tags: list[str] = []
    similarity: float
    cover_url: Optional[str] = None


class MatchedAlbum(BaseModel):
    """The album the person actually searched for — shown directly under the
    search bar, separately from the list of recommended cross-references."""
    title: str
    artist_names: str
    release_year: Optional[str] = None
    rating: Optional[float] = None
    tags: list[str] = []
    cover_url: Optional[str] = None


class RecommendResponse(BaseModel):
    query: str
    matched_title: Optional[str] = None
    matched_album: Optional[MatchedAlbum] = None
    suggestions: list[str] = []
    results: list[Recommendation] = []


class RecentAlbum(BaseModel):
    title: str
    artist_names: str
    release_year: Optional[str] = None
    rating: Optional[float] = None
    tags: list[str] = []
    cover_url: Optional[str] = None
    added_at: Optional[str] = None
    edited: bool = False
    edited_at: Optional[str] = None


class RecentResponse(BaseModel):
    albums: list[RecentAlbum] = []


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/albums")
def list_albums():
    """Returns every album title currently in the library (for autocomplete)."""
    df, _ = get_cache()
    return {"titles": sorted(df["title"].dropna().unique().tolist())}


@app.get("/api/recommend", response_model=RecommendResponse)
def recommend(title: str, n: int = 5):
    df, feature_matrix = get_cache()

    matches = df[df["title"].str.lower() == title.strip().lower()]

    if matches.empty:
        close = difflib.get_close_matches(title, df["title"].tolist(), n=5, cutoff=0.4)
        return RecommendResponse(query=title, suggestions=close, results=[])

    idx = matches.index[0]
    matched_row = matches.iloc[0]
    matched_album = MatchedAlbum(
        title=matched_row["title"],
        artist_names=matched_row.get("artist_names", "Unknown artist"),
        release_year=str(matched_row["release_year"]) if pd.notna(matched_row["release_year"]) else None,
        rating=float(matched_row["rating"]) if pd.notna(matched_row["rating"]) else None,
        tags=matched_row["tags"],
        cover_url=cover_art_url(matched_row.get("mbid")),
    )

    sim_scores = cosine_similarity([feature_matrix[idx]], feature_matrix)[0]

    results = df.copy()
    results["similarity"] = sim_scores
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
            cover_url=cover_art_url(row.get("mbid")),
        )
        for _, row in results.iterrows()
    ]

    return RecommendResponse(
        query=title,
        matched_title=matched_row["title"],
        matched_album=matched_album,
        results=recs,
    )


@app.get("/api/recent", response_model=RecentResponse)
def recent(n: int = 20):
    """Recently added/edited albums, newest first. "Edited" means updated_at
    is meaningfully after created_at (see the DB trigger in
    schema_additions_updated_at.sql, which skips the bump for no-op updates)."""
    df, _ = get_cache()

    working = df.copy()
    working["_sort_ts"] = working[["created_at", "updated_at"]].max(axis=1)
    working = working.sort_values("_sort_ts", ascending=False, na_position="last")
    working = working.head(min(max(n, 1), 100))

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
            cover_url=cover_art_url(row.get("mbid")),
            added_at=created.isoformat() if pd.notna(created) else None,
            edited=edited,
            edited_at=updated.isoformat() if edited else None,
        ))

    return RecentResponse(albums=albums)


@app.post("/api/refresh")
def refresh(authorization: Optional[str] = Header(None)):
    """Force-rebuild the cached feature matrix (e.g. after adding new albums)."""
    if ADMIN_REFRESH_TOKEN:
        if authorization != f"Bearer {ADMIN_REFRESH_TOKEN}":
            raise HTTPException(status_code=401, detail="Invalid or missing token")
    get_cache(force=True)
    return {"status": "refreshed", "albums": len(_cache["df"])}