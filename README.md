# Villa Community Staffing Survey — Analytics Dashboard

A Streamlit dashboard for the villa-community shared-staffing subscription survey. Covers:

- **Descriptive & diagnostic analysis** — summary stats, cross-tabulations, chi-square/Cramer's V and
  ANOVA/eta-squared driver ranking
- **Classification** — KNN, Decision Tree, Random Forest, Gradient Boosting (with grid-search tuning),
  train vs. test accuracy/precision/recall/F1, confusion matrices, multi-class ROC curves
- **Clustering** — K-means with elbow chart + silhouette score, 3D PCA visualization, cluster profiling,
  and "best cluster" identification
- **Findings** — a guide to interpreting all of the above

## Project structure

```
streamlit_villa_app/
├── app.py                        # main Streamlit app (run this)
├── requirements.txt
├── data/
│   └── sample_data.csv           # bundled synthetic data so the app works out of the box
├── utils/
│   ├── data_processing.py        # data loading + feature engineering
│   ├── descriptive_diagnostic.py # cross-tabs, chi-square, ANOVA
│   ├── classification_models.py  # KNN / DT / RF / GBM training + metrics + plots
│   └── clustering_analysis.py    # K-means + elbow + PCA 3D + cluster profiling
└── README.md
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Deploy on Streamlit Community Cloud (free) via GitHub

1. **Push this folder to a new GitHub repository.**
   ```bash
   cd streamlit_villa_app
   git init
   git add .
   git commit -m "Initial commit: villa staffing survey analytics dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with your GitHub account.
3. Click **"New app"**, then select:
   - **Repository:** `<your-username>/<your-repo-name>`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy**. The first build takes a few minutes (installing `requirements.txt`); after that,
   any future `git push` to `main` redeploys automatically.
5. Once live, you'll get a shareable URL like `https://<your-app-name>.streamlit.app`.

## Using your own (real) survey data

Once you've collected real responses, export them to a CSV with the **same column names** as
`data/sample_data.csv` (i.e. matching the survey questionnaire: `Q2_community`, `Q3_nationality_region`,
... `Q25_interest`), and upload it via the **file uploader in the sidebar** of the running app — no code
changes needed. The bundled `sample_data.csv` is only a fallback default for demoing the dashboard before
real data exists.

## Notes on the analysis choices

- **"Super learning"** in the original request is implemented here as a side-by-side comparison of four
  classifiers (KNN, Decision Tree, Random Forest, Gradient Boosting) rather than the formal *SuperLearner*
  stacking algorithm (which trains a meta-model on top of these four). If you'd like the formal stacking
  version too, `sklearn.ensemble.StackingClassifier` can be dropped into `classification_models.py` using
  the same four pipelines as base estimators — ask if you'd like this added.
- **Clustering k selection:** the dashboard shows both the elbow knee-point and the pure silhouette-best k,
  since they can disagree (silhouette tends to favor one big coarse split on high-dimensional one-hot survey
  data, while the knee-point tends to better match a business-meaningful number of personas). A slider lets
  you pick k interactively and see how the resulting segments look before committing to a number.
- **"Best cluster"** is defined as the segment ranking highest on a combined score of average
  willingness-to-pay and % saying "Yes" to subscribing — a business judgment call, not a statistical one.
  Change the definition in `identify_best_cluster()` if your priorities differ (e.g. weighting cluster size
  more heavily if you care about total addressable revenue rather than per-respondent value).
