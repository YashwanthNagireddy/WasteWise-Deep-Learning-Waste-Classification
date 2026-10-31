# GitHub Upload Guide (Windows)

Repository name: `WasteWise-Deep-Learning-Waste-Classification`

## A. Create the repository on GitHub
1. Log in at https://github.com -> click **+** (top right) -> **New repository**.
2. Name: `WasteWise-Deep-Learning-Waste-Classification`. Choose Public.
3. Do NOT tick "Add README/.gitignore/license" (the project already has them). Click **Create repository**.

## B. Upload from the terminal (inside the WasteWise folder)
```
git --version
git init
git branch -M main
git add .
git status
git commit -m "Initial commit: WasteWise waste classification project"
git remote add origin https://github.com/YOUR_USERNAME/WasteWise-Deep-Learning-Waste-Classification.git
git push -u origin main
```
First time only, set your identity:
```
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```
When asked to log in, use the browser sign-in or a Personal Access Token (GitHub -> Settings -> Developer settings -> Tokens) as the password.

Check `git status` before committing: no `venv/`, `.keras` model or dataset images should be listed.

## C. Upload from VS Code
1. Open the WasteWise folder (File -> Open Folder).
2. Click the **Source Control** icon -> **Initialize Repository**.
3. Type a commit message, click **Commit** (stage all if asked).
4. Click **Publish Branch** -> choose the repo name above -> sign in to GitHub when prompted.

## D. Later updates
```
git add .
git commit -m "Describe your change"
git push
```
Add screenshots and your own results to the README, then push again.
