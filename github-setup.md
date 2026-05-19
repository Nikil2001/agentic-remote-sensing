# GitHub Repository Setup Guide

This guide walks you through setting up a GitHub repository, creating SSH keys for authentication, and pushing your source code.

---


## 1. Create a GitHub Repository

1. Go to [github.com](https://github.com) and sign in (or create an account)
2. Click the **+** icon in the top-right corner → **New repository**
3. Configure your repository:
   - **Repository name**: `Querydriven` (or your preferred name)
   - **Description**: Query-Driven Remote Sensing Data Interpretation
   - **Visibility**: Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we'll push existing code)
4. Click **Create repository**

---

## 2. Generate SSH Key

SSH keys provide secure, password-less authentication with GitHub.

### Check for Existing Keys

```bash
ls -la ~/.ssh
```

Look for files like `id_ed25519` or `id_rsa`. If they exist, you can skip to Step 3.

### Create New SSH Key

```bash
# Generate a new SSH key (recommended: ed25519)
ssh-keygen -t ed25519 -C "your_email@example.com"

# When prompted:
# - Press Enter to accept default file location (~/.ssh/id_ed25519)
# - Enter a passphrase (optional but recommended)
```

### Start SSH Agent & Add Key

```bash
# Start the SSH agent
eval "$(ssh-agent -s)"

# Add your SSH key to the agent
ssh-add ~/.ssh/id_ed25519
```

### Copy Your Public Key

```bash
# Copy to clipboard (macOS)
pbcopy < ~/.ssh/id_ed25519.pub

# Or display it to copy manually
cat ~/.ssh/id_ed25519.pub
```

---

## 3. Add SSH Key to GitHub

1. Go to [GitHub SSH Settings](https://github.com/settings/keys)
2. Click **New SSH key**
3. Configure:
   - **Title**: `MacBook` (or any descriptive name)
   - **Key type**: Authentication Key
   - **Key**: Paste your public key
4. Click **Add SSH key**

### Verify Connection

```bash
ssh -T git@github.com
```

Expected output: `Hi username! You've successfully authenticated...`

---

## 4. Initialize & Push Your Repository

### Initialize Git (if not already done)

```bash
cd /Users/sanjeevmurthy/Documents/Nikil/Querydriven

# Initialize git repository
git init

# Create .gitignore file
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
venv/
.env

# Model weights (large files)
*.pth

# IDE
.vscode/
.idea/

# OS files
.DS_Store
Thumbs.db

# Uploaded images
static/uploaded*.png
static/mask*.png
EOF
```

### Configure Git Identity

```bash
git config user.name "Your Name"
git config user.email "your_email@example.com"
```

### Add, Commit, and Push

```bash
# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Query-Driven Remote Sensing Application"

# Add remote repository (replace with your repository URL)
git remote add origin git@github.com:YOUR_USERNAME/Querydriven.git

# Push to GitHub
git push -u origin main
```

> **Note**: If your branch is named `master` instead of `main`, use:
>
> ```bash
> git push -u origin master
> ```

---

## 5. Subsequent Pushes

After initial setup, push changes with:

```bash
git add .
git commit -m "Your commit message"
git push
```

---

## Quick Reference

| Task                | Command                        |
| ------------------- | ------------------------------ |
| Check git status    | `git status`                   |
| View commit history | `git log --oneline`            |
| Pull latest changes | `git pull`                     |
| Create new branch   | `git checkout -b feature-name` |
| Switch branch       | `git checkout branch-name`     |
| View remotes        | `git remote -v`                |

---

## Troubleshooting

### "Permission denied (publickey)"

- Verify SSH key is added: `ssh-add -l`
- Check key is on GitHub: [github.com/settings/keys](https://github.com/settings/keys)
- Test connection: `ssh -T git@github.com`

### "Repository not found"

- Verify repository URL is correct
- Ensure you have access to the repository
- Check if using SSH URL (not HTTPS): `git@github.com:user/repo.git`

### "Failed to push some refs"

- Pull first: `git pull --rebase origin main`
- Then push again: `git push`

---

_Last updated: February 2026_
