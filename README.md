PCA Analyzer (Streamlit)

Quick steps:

1) Install dependencies: pip install -r requirements.txt
2) Run app: streamlit run streamlit_app.py

One-click: optional run.sh provided to start Streamlit quickly (make it executable):

How to prepare Excel file for upload:
- First row: feature names (column headers). Include only numeric feature columns; you can add one categorical column (e.g., 'Group') to color points.
- Each subsequent row: one sample/observation.
- Example CSV content (save as .csv or .xlsx):

SampleID,Feature_A,Feature_B,Feature_C,Group
S1,1.2,3.4,2.1,A
S2,2.3,3.8,1.8,B
S3,1.9,4.0,2.5,A

Upload that file in the app.

Deployment to Streamlit Community Cloud (persistent public URL):

1) Create a public GitHub repository and push this project:
   - git remote add origin https://github.com/<your-username>/<repo-name>.git
   - git branch -M main
   - git push -u origin main

2) On https://streamlit.io/cloud, sign in and choose 'New app'.
   - Connect your GitHub repo, select branch 'main' and the file 'streamlit_app.py'.
   - Add required secrets (if any) in the app settings.
   - Click deploy — Streamlit will build and provide a public URL.

Notes:
- Keep requirements.txt updated so the cloud builder installs dependencies.
- The app will be publicly accessible once deployed. Remove secrets from code before pushing.
