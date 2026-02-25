# Team T

A collaborative project site built for [GitHub Pages](https://BaileyMeche.github.io/team_t/).

## 🚀 Live Site

Visit: **https://BaileyMeche.github.io/team_t/**

## 📁 Structure

```
team_t/
├── index.html          # Main landing page
├── styles.css          # Site styles
└── .github/
    └── workflows/
        └── deploy.yml  # GitHub Actions deploy workflow
```

## 🛠 Deployment

The site is automatically deployed to GitHub Pages whenever changes are pushed to the `main` branch via the GitHub Actions workflow in `.github/workflows/deploy.yml`.

To enable GitHub Pages manually:
1. Go to **Settings → Pages** in the repository.
2. Under **Build and deployment**, select **Source: GitHub Actions**.
3. Push to `main` — the workflow will handle the rest.