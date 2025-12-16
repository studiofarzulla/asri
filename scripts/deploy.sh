#!/bin/bash
# Deploy ASRI static site
#
# 1. Rsync to K3s cluster (immediate update)
# 2. Copy to resurrexi-io repo and git push (Cloudflare Pages)
# 3. Git commit/push asri repo changes

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOCAL_DIR="${1:-$PROJECT_DIR/static_site}"

# Paths
REMOTE_HOST="sudosenpai@192.168.2.49"
REMOTE_PATH="/mnt/storage/resurrexi-io/public/asri"
RESURREXI_IO_REPO="/home/purrpower/Resurrexi/projects/websites/resurrexi-io"
ASRI_REPO="$PROJECT_DIR"

echo "=== ASRI Deploy ==="
echo "Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

# Check if source exists
if [ ! -f "$LOCAL_DIR/index.html" ]; then
    echo "Error: $LOCAL_DIR/index.html not found"
    echo "Run the static generator first"
    exit 1
fi

# Extract ASRI value for commit message
ASRI_VALUE=$(grep -oP '(?<=asri-value">)[0-9.]+' "$LOCAL_DIR/index.html" | head -1)
COMMIT_MSG="ASRI update: ${ASRI_VALUE:-unknown} - $(date -u '+%Y-%m-%d %H:%M UTC')"

echo "ASRI Value: $ASRI_VALUE"
echo ""

# Step 1: Rsync to K3s (immediate update)
echo "📤 [1/3] Syncing to K3s cluster..."
ssh "$REMOTE_HOST" "mkdir -p $REMOTE_PATH"
rsync -avz --delete "$LOCAL_DIR/" "$REMOTE_HOST:$REMOTE_PATH/"
echo "✅ K3s updated"
echo ""

# Step 2: Update resurrexi-io repo
echo "📤 [2/3] Updating resurrexi-io repo..."
mkdir -p "$RESURREXI_IO_REPO/public/asri"
cp -r "$LOCAL_DIR/"* "$RESURREXI_IO_REPO/public/asri/"

cd "$RESURREXI_IO_REPO"
if git diff --quiet && git diff --staged --quiet; then
    echo "   No changes to commit in resurrexi-io"
else
    git add public/asri/
    git commit -m "$COMMIT_MSG

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
    git push origin main
    echo "✅ resurrexi-io pushed (Cloudflare Pages will auto-deploy)"
fi
echo ""

# Step 3: Update asri repo
echo "📤 [3/3] Updating asri repo..."
cd "$ASRI_REPO"
if git diff --quiet && git diff --staged --quiet; then
    echo "   No changes to commit in asri"
else
    # Add relevant files (not the static site itself, just code changes)
    git add -A
    # Check again after staging
    if git diff --staged --quiet; then
        echo "   No staged changes in asri"
    else
        git commit -m "Daily ASRI calculation - $ASRI_VALUE

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
        git push origin main
        echo "✅ asri repo pushed"
    fi
fi
echo ""

echo "=========================================="
echo "✅ Deployment complete!"
echo "   K3s:             https://resurrexi.io/asri/"
echo "   Cloudflare:      (auto-deploys from git)"
echo "   ASRI Value:      $ASRI_VALUE"
echo "=========================================="
