# **Guide: Setting Up Automatic Versioning in GitHub with GitHub Actions**

This guide explains how to implement **automatic versioning** for your projects using **GitHub Actions**. This setup ensures that:

- **A new Git tag is created** whenever a version change is detected.
- **A GitHub Release is automatically generated**, attaching the `CHANGELOG.md`.
- **Versioning works dynamically for multiple projects**, avoiding hardcoded project names.

------

## Step 1: Create the Necessary Files

### **1. Add a `VERSION` File for Your Package**

Each package should have its own `VERSION` file.

Example for `database_toolkit/`:

```sh
echo "1.0.0" > database_toolkit/VERSION
```

### **2. Add a `CHANGELOG.md` File for Your Package**

```sh
touch database_toolkit/CHANGELOG.md
```

Example content:

```md
# **Changelog**

All notable changes to this project will be documented in this file.

## **[Unreleased]**
- Describe upcoming changes here.

## **[0.1.0]** - YYYY-MM-DD
### **Added**
- Initial release of ` project_name `.
- Implemented core utilities.
- Added unit tests.

### **Fixed**
- N/A

### **Changed**
- N/A

### **Removed**
- N/A

---
```

### **3. Create a `PROJECTS` File**

This file will list all the packages that require versioning. Heads-up: the PROJECT file should live in the root of the repository.

```sh
echo "database_toolkit" > PROJECTS
```

If you have multiple projects, list them one per line:

```
database_toolkit
another_project
```

------

## Step 2: Create the GitHub Actions Workflow

### **1.  Create the Workflow File**

In the root of your repository, create the following Workflow File:

```sh
mkdir -p .github/workflows
nano .github/workflows/versioning.yml
```

### **2. Add the Following Configuration**

```yaml
name: Automatic Versioning

on:
  push:
    branches:
      - main  # Runs when merging into main

permissions:
  contents: write

jobs:
  tag_version:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Read project names
        id: read_projects
        run: echo "PROJECTS=$(cat PROJECTS)" >> $GITHUB_ENV

      - name: Process Each Project
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}  # Set the token for authentication
        run: |
          for project in ${{ env.PROJECTS }}; do
            echo "Processing project: $project"

            # Read version file
            VERSION=$(cat $project/VERSION)
            echo "Detected version $VERSION for $project"

            # Check if the tag already exists in remote
            if git ls-remote --tags origin | grep -q "refs/tags/v${VERSION}"; then
              echo "Tag v${VERSION} already exists for $project, skipping..."
              continue
            fi

            # Create a Git tag
            git config --global user.name "github-actions"
            git config --global user.email "github-actions@github.com"
            git tag -a "v${VERSION}" -m "Release version ${VERSION} for ${project}"
            git push origin "v${VERSION}"

            # Create GitHub release
            gh release create "v${VERSION}" --title "Release ${VERSION}" --notes-file $project/CHANGELOG.md
          done
```

------

## Step 3: Commit and Push the Workflow

```sh
git add PROJECTS .github/workflows/versioning.yml database_toolkit/VERSION database_toolkit/CHANGELOG.md
git commit -m "Add automatic versioning workflow"
git push origin adding_versioning
```

------

## Step 4: Merge the Workflow into `main`

```sh
git checkout main
git merge adding_versioning
git push origin main
```

------

## Step 5: Test the Workflow

### **1. Update the Version in `VERSION` File**

```sh
echo "1.0.1" > database_toolkit/VERSION
git add database_toolkit/VERSION
git commit -m "Bump version to 1.0.1"
git push origin main
```

### **2.  Check the GitHub Actions Execution**

- Go to **GitHub > Actions**.
- The workflow should run and:
  - **Detect the version change**.
  - **Create a new Git tag** (`v1.0.1`).
  - **Publish a GitHub Release**, attaching `CHANGELOG.md`.

------



**Now you have a fully automated versioning system in your GitHub repository!** 