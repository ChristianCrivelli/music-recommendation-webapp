# Album Recommendations — public app

![License](https://img.shields.io/badge/license-MIT-blue.svg)

Wraps your existing Supabase-backed recommender (feature matrix + cosine
similarity) in a small FastAPI service, with a static frontend styled as a
library card catalog. Two independent pieces:

```
backend/    FastAPI service — /api/albums, /api/recommend, /api/refresh
frontend/   Single static HTML file (no build step) — the public UI
```

## 1. Lock down Supabase access first

Before deploying anything publicly, create a **read-only** key/role:

- In Supabase, add Row Level Security (RLS) policies on `albums`,
  `album_tags`, `tags`, `album_contributions`, `artists` that only allow
  `SELECT`.
- Use the `anon` key (with those RLS policies) for this app's
  `SUPABASE_KEY` — never your service-role / write key.
- Keep your existing ingestion scripts (`album_finder.py`, `pull_albums.py`,
  `album_push_logic.py`) on your machine only, using your write key. They
  never need to touch the deployed app.

## 2. Run the backend locally

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in SUPABASE_URL / SUPABASE_KEY
uvicorn main:app --reload
```

Check it: `curl http://localhost:8000/api/albums`

## 3. Run the frontend locally

`frontend/index.html` is a single static file — just open it, or serve it:

```bash
cd frontend
python -m http.server 5500
```

By default it points at `http://localhost:8000`. To point it at a deployed
backend, set this before the app runs (e.g. in a small inline `<script>`
tag you add at the top of `index.html`, or by editing the constant):

```html
<script>window.ALBUM_API_BASE = "https://your-backend.onrailway.app";</script>
```

## 4. Deploy

**Backend — Render (free tier, no card required):**
- Easiest: this repo includes `render.yaml`, so on [render.com](https://render.com) choose
  **New → Blueprint**, point it at this repo, and Render reads the file and
  sets up the service (root dir `backend`, build/start commands) automatically.
- Or manually: **New → Web Service** → select this repo → root directory
  `backend` → build command `pip install -r requirements.txt` → start command
  `uvicorn main:app --host 0.0.0.0 --port $PORT` → plan **Free**.
- Either way, set the env vars in the Render dashboard: `SUPABASE_URL`,
  `SUPABASE_KEY` (read-only!), `FRONTEND_ORIGIN` (your deployed frontend's
  exact URL, once you have it — locks down CORS), optionally
  `ADMIN_REFRESH_TOKEN`.
- Heads up on the free tier: it spins down after 15 minutes of no traffic,
  so the first request after a quiet period takes ~30-60 seconds to wake
  back up. Fine for a personal project; if that ever bugs you, Render's
  paid tier removes it.

**Frontend** — any static host (Vercel, Netlify, GitHub Pages):
- No build step needed — just deploy the `frontend/` folder.
- Set `window.ALBUM_API_BASE` to your backend's URL.

## 5. Keeping recommendations fresh

The backend caches the feature matrix in memory and rebuilds it
automatically every `CACHE_TTL_SECONDS` (default: 1 hour). After adding
albums with your ingestion scripts, you can also force an immediate
rebuild:

```bash
curl -X POST https://your-backend.onrailway.app/api/refresh \
  -H "Authorization: Bearer YOUR_ADMIN_REFRESH_TOKEN"
```

(Skip the header if you didn't set `ADMIN_REFRESH_TOKEN`.)

## Notes on the matching

`/api/recommend?title=...` requires an exact (case-insensitive) title
match today, same as the original script — but now falls back to fuzzy
suggestions if nothing matches exactly, and `/api/albums` gives the
frontend a full title list to drive the autocomplete dropdown, so visitors
never have to guess your exact spelling/capitalization.
