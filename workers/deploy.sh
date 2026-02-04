#!/bin/bash
# ASRI API Deployment Script for Cloudflare Workers

set -e

echo "=== ASRI API Deployment ==="

# Check for wrangler
if ! command -v wrangler &> /dev/null; then
    echo "Installing wrangler..."
    npm install -g wrangler
fi

# Login check
echo "Checking Cloudflare authentication..."
wrangler whoami || wrangler login

# Create D1 database if it doesn't exist
echo "Setting up D1 database..."
DB_ID=$(wrangler d1 list | grep "asri-db" | awk '{print $1}' || echo "")

if [ -z "$DB_ID" ]; then
    echo "Creating D1 database 'asri-db'..."
    wrangler d1 create asri-db

    # Get the new database ID
    DB_ID=$(wrangler d1 list | grep "asri-db" | awk '{print $1}')
    echo "Database created with ID: $DB_ID"

    # Update wrangler.toml with database ID
    sed -i "s/database_id = \"\"/database_id = \"$DB_ID\"/" wrangler.toml
fi

# Apply schema
echo "Applying database schema..."
wrangler d1 execute asri-db --file=schema.sql

# Deploy the worker
echo "Deploying worker..."
wrangler deploy

echo ""
echo "=== Deployment Complete ==="
echo "API will be available at: https://api.dissensus.ai"
echo "Docs at: https://api.dissensus.ai/docs"
