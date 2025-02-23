# **MkDocs: Installing, Configuring, and Automating it with GitHub Pages**

This tutorial provides a step-by-step walkthrough to **install, configure, and deploy MkDocs** manually and set up an **automated deployment pipeline** using GitHub Actions.

## Step 1: Install MkDocs with Poetry

I use **Poetry** as a package manager to keep dependencies organized.

### 1. Navigate to your project folder:

```bash
cd /path/to/general_projects
```

### 2. Initialize Poetry (if not already set up):

```bash
poetry init
```

(You can leave dependencies empty for now.)

### 3. Install MkDocs and the Material theme:

```bash
poetry add --group dev mkdocs mkdocs-material
```

### 4. Verify the installation:

```bash
poetry run mkdocs --version
```

Expected output:

```bash
mkdocs, version X.X.X from ...
```

## Step 2: Create MkDocs File Structure

MkDocs expects content inside a `docs/` directory.

### 1. Initialize MkDocs

```bash
mkdocs new .
```

This creates:

```bash
.
├── docs
│   └── index.md  # Homepage content
├── mkdocs.yml  # Configuration file
```

We can add extra folders following the same template:

```bash
├── docs
│   ├── guides
│   │   ├── deep_learning
│   │   │   └── overview.md
│   │   ├── projects
│   │   │   └── overview.md
│   │   ├── software_architecture
│   │   │   ├── SOLID_principles.md
│   │   │   └── overview.md
│   │   └── unit_testing
│   │       └── overview.md
│   └── index.md
├── mkdocs.yml
```



### 2. Modify `**mkdocs.yml**` to use the Material theme and ensure proper navigation:

```yaml
site_name: My Software & Data Science Portfolio
site_url: https://yourusername.github.io/repository-name/

theme:
  name: material
  features:
    - navigation.instant
    - navigation.tabs
    - navigation.top
    - search.suggest
    - search.highlight
  palette:
    scheme: default
  icon:
    logo: material/book

nav:
  - Home: index.md
  - Software Architecture:
      - Overview: guides/software_architecture/overview.md
      - SOLID Principles: guides/software_architecture/SOLID_principles.md
  - Unit Testing:
      - Overview: guides/unit_testing/overview.md
  - Deep Learning:
      - Overview: guides/deep_learning/overview.md
  - Projects:
      - Overview: guides/projects/overview.md

extra_css:
  - assets/stylesheets/extra.css

extra_javascript:
  - assets/javascripts/extra.js

markdown_extensions:
  - admonition
  - pymdownx.highlight
  - pymdownx.superfences
  - toc:
      permalink: true

plugins:
  - search

```

### 3. Create the necessary folders and files:

```bash
mkdir -p docs/guides/software_architecture docs/guides/unit_testing docs/guides/deep_learning docs/guides/projects
```

## Step 3: Run MkDocs Locally

To preview the documentation before deploying:

```bash
poetry run mkdocs build --clean
poetry run mkdocs serve --dev-addr=0.0.0.0:8080

```

Open **http://0.0.0.0:8080/** in your browser to verify.

## Step 4: Manual Deployment to GitHub Pages

### 1. Ensure GitHub Pages is enabled:

- Go to **Settings > Pages** in your repo.
- Set **Branch** to `gh-pages`, and select **/ (root)`(NOT/docs` !!!).

### 2. Disable Jekyll to prevent conflicts:

```bash
touch docs/.nojekyll
```

### 3. Commit and push all changes:

```bash
git add .
git commit -m "Initial MkDocs setup"
git push origin main
```

### 4. Deploy manually:

```bash
poetry run mkdocs gh-deploy --force
```

### 5. Verify deployment at:

```bash
https://yourusername.github.io/repository-name/
```

## Step 5: Set Up GitHub Actions for Automatic Deployment

Instead of running `mkdocs gh-deploy` manually, automate deployment when merging into `main`.

### 1. Create the workflow file:

```bash
mkdir -p .github/workflows
nano .github/workflows/deploy-mkdocs.yml
```

### 2. Add the following configuration:

```yaml
name: Deploy MkDocs to GitHub Pages

on:
  push:
    branches:
      - main

permissions:
  contents: write

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install Poetry
        run: pip install poetry

      - name: Install dependencies
        run: poetry install --no-root

      - name: Deploy MkDocs
        run: poetry run mkdocs gh-deploy --force
```

### 3. Commit and push the workflow:

```bash
git add .github/workflows/deploy-mkdocs.yml
git commit -m "Add GitHub Actions for MkDocs deployment"
git push origin main
```

### 4. Test Automatic Deployment:

- Go to **GitHub > Actions tab**.
- Ensure the workflow **runs successfully after merging a branch into** `**main**`.
- Once finished, visit your site to check the updates.


## **Common Errors and Fixes**

### ** Error: OSError: [Errno 98] Address already in use**

**Fix:** If MkDocs fails to start due to a port conflict, find and kill the process occupying the port.

```bash
lsof -i :8080
kill -9 <PID>
```

Then restart MkDocs:

```bash
poetry run mkdocs serve --dev-addr=0.0.0.0:8080
```


## Summary: Quick Commands

### **Run MkDocs Locally**

```bash
poetry run mkdocs serve
```

### **Deploy Manually**

```bash
poetry run mkdocs gh-deploy --force
```

### **Enable GitHub Actions for Automatic Deployment**

```bash
git push origin main
```

 **Now you have a fully automated MkDocs site deployed on GitHub Pages!** 