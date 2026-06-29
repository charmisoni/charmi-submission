# Villa Community Staffing Survey — Analytics Dashboard (single-file version)

Everything (data loading, feature engineering, descriptive/diagnostic stats, classification,
clustering, and the Streamlit UI) lives in **one file: `app.py`**. No `utils/` package to forget
when uploading to GitHub — that's exactly what caused the `ModuleNotFoundError` last time.

## Files you need in the repo (only 3 things)

```
your-repo/
├── app.py
├── requirements.txt
└── data/
    └── sample_data.csv
```

That's it. No other folders.

## Deploy on Streamlit Community Cloud via GitHub

1. **Create a new GitHub repo and push exactly these 3 items** (easiest: use git from a terminal
   rather than the GitHub web "Upload files" button, which is the most common way a subfolder like
   `data/` silently gets left out):
   ```bash
   cd streamlit_villa_app_standalone
   git init
   git add .
   git commit -m "Single-file villa staffing survey dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
   If you don't have git set up locally and must use the GitHub website: create the repo, then use
   **"Add file" → "Upload files"**, and drag the `app.py`, `requirements.txt`, and the **entire**
   `data` folder (with `sample_data.csv` inside it) all in the **same upload**, not one at a time —
   uploading folders one level at a time is the most common way a file gets dropped.

2. **Verify on GitHub.com** that all 3 items actually show up in the repo's root listing before
   deploying. If `data/sample_data.csv` isn't visible there, it won't be visible to Streamlit Cloud
   either.

3. Go to **[share.streamlit.io](https://share.streamlit.io)** → sign in with GitHub → **"New app"**:
   - Repository: `<your-username>/<your-repo-name>`
   - Branch: `main`
   - Main file path: `app.py`
4. Click **Deploy**. Future `git push` to `main` redeploys automatically.

## Run locally first (recommended sanity check before deploying)

```bash
pip install -r requirements.txt
streamlit run app.py
```

If it works locally with this exact folder structure, it will work on Streamlit Cloud too.

## Using your own (real) survey data

Once you have real responses, export to a CSV with the same column names as `data/sample_data.csv`
and upload it via the **file uploader in the sidebar** — no code or redeployment needed.
