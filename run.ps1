# =============================================================================
# run.ps1 - Simplified Docker and Project Management Commands (Windows)
# =============================================================================
# Usage: .\run.ps1 <target>
# Example: .\run.ps1 build, .\run.ps1 up, .\run.ps1 logs
# To list all available targets: .\run.ps1 help  or  .\run.ps1
#
# Note: you may need to allow script execution first:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
# =============================================================================

param(
    [Parameter(Position = 0)]
    [string]$Target = "help"
)

$ErrorActionPreference = "Stop"

function Write-Blue($msg)   { Write-Host $msg -ForegroundColor Blue }
function Write-Green($msg)  { Write-Host $msg -ForegroundColor Green }
function Write-Red($msg)    { Write-Host $msg -ForegroundColor Red }
function Write-Yellow($msg) { Write-Host $msg -ForegroundColor Yellow }

# -----------------------------------------------------------------------------
# Target descriptions (like ## comments in the Makefile)
# -----------------------------------------------------------------------------
$Targets = [ordered]@{
    "help"                        = "Show this help message"
    "build"                       = "Build all Docker images"
    "build-no-cache"              = "Build all images without using cache"
    "build-python-base"           = "Build the base Python image (required first)"
    "build-python-base-no-cache"  = "Build base Python image without cache"
    "build-base"                  = "Build only the base image"
    "build-ml"                    = "Build the ML image"
    "build-frontend"              = "Build the frontend image"
    "up"                          = "Start all services"
    "up-build"                    = "Build and start all services"
    "down"                        = "Stop all services"
    "down-volumes"                = "Stop all services and remove volumes"
    "restart"                     = "Restart all services"
    "ps"                          = "Show running services"
    "stop"                        = "Stop all services (without removing)"
    "start"                       = "Start stopped services"
    "dev"                         = "Start in development mode with hot reload"
    "dev-build"                   = "Build and start in development mode"
    "logs"                        = "Show logs from all services"
    "logs-api"                    = "Show logs from API gateway"
    "logs-retrieval"              = "Show logs from retrieval service"
    "logs-frontend"               = "Show logs from frontend"
    "health"                      = "Check health status of all services"
    "db-shell"                    = "Open PostgreSQL shell"
    "db-backup"                   = "Backup database to the backups folder"
    "db-restore"                  = "Restore database from latest backup"
    "shell-api"                   = "Open shell in API gateway container"
    "shell-retrieval"             = "Open shell in retrieval container"
    "shell-preprocessing"         = "Open shell in preprocessing container"
    "test"                        = "Run tests in containers"
    "test-integration"            = "Run integration tests"
    "build-indices"               = "Build all indices"
    "build-tfidf"                 = "Build TF-IDF index only"
    "build-bm25"                  = "Build BM25 index only"
    "build-embeddings"            = "Build embedding index only"
    "refresh-views"               = "Refresh materialized views"
    "clear-cache"                 = "Clear all Redis cache"
    "cache-stats"                 = "Show cache statistics"
    "init-db"                     = "Initialize database schema (run once)"
    "check-data"                  = "Check status of datasets, indices, and database"
    "download-dataset"            = "Download dataset using ir-datasets"
    "preprocess-data"             = "Preprocess downloaded dataset"
    "load-to-db"                  = "Load processed data to database"
    "evaluate"                    = "Run full evaluation"
    "clean"                       = "Remove stopped containers"
    "clean-all"                   = "Remove containers, images, and volumes"
    "clean-cache"                 = "Clean Python cache files"
    "clean-logs"                  = "Clean log files"
    "stats"                       = "Show container resource usage"
    "inspect"                     = "Show detailed system information"
    "quickstart"                  = "Quick start: build, start, and show logs"
    "prod-up"                     = "Start in production mode"
    "prod-build"                  = "Build for production"
    "init"                        = "Initialize project (first time setup)"
}

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
function Show-Help {
    Write-Blue "IR System - Docker Management (Windows)"
    Write-Host ""
    Write-Green "Available targets:"
    foreach ($key in $Targets.Keys) {
        $desc = $Targets[$key]
        $padded = $key.PadRight(28)
        Write-Host ("  {0}  {1}" -f $padded, $desc)
    }
    Write-Host ""
}

# -----------------------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------------------
switch ($Target) {

    "help" { Show-Help }

    # --- Build ---
    "build-python-base" {
        Write-Blue "Building Python base image..."
        docker build -f docker/python-base.Dockerfile -t ir-python-base:latest .
    }
    "build-python-base-no-cache" {
        Write-Blue "Building Python base image (no cache)..."
        docker build --no-cache -f docker/python-base.Dockerfile -t ir-python-base:latest .
    }
    "build" {
        & $PSCommandPath build-python-base
        Write-Blue "Building all Docker images..."
        docker compose build
    }
    "build-no-cache" {
        & $PSCommandPath build-python-base-no-cache
        Write-Blue "Building all images (no cache)..."
        docker compose build --no-cache
    }
    "build-base" {
        Write-Blue "Building base image..."
        docker build -f docker/base.Dockerfile -t ir-base:latest .
    }
    "build-ml" {
        Write-Blue "Building ML image..."
        docker build -f docker/ml.Dockerfile -t ir-ml:latest .
    }
    "build-frontend" {
        Write-Blue "Building frontend image..."
        docker build -f docker/frontend.Dockerfile -t ir-frontend:latest .
    }

    # --- Compose ---
    "up" {
        Write-Green "Starting all services..."
        docker compose up -d
    }
    "up-build" {
        Write-Green "Building and starting all services..."
        docker compose up -d --build
    }
    "down" {
        Write-Red "Stopping all services..."
        docker compose down
    }
    "down-volumes" {
        Write-Red "Stopping services and removing volumes..."
        docker compose down -v
    }
    "restart" {
        Write-Yellow "Restarting all services..."
        docker compose restart
    }
    "ps" { docker compose ps }
    "stop" { docker compose stop }
    "start" { docker compose start }

    # --- Dev ---
    "dev" {
        Write-Green "Starting in development mode..."
        docker compose -f docker-compose.yml -f docker-compose.dev.yml up
    }
    "dev-build" {
        Write-Green "Building and starting in development mode..."
        docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
    }

    # --- Logs ---
    "logs" { docker compose logs -f }
    "logs-api" { docker compose logs -f api_gateway }
    "logs-retrieval" { docker compose logs -f retrieval }
    "logs-frontend" { docker compose logs -f frontend }

    # --- Health ---
    "health" {
        Write-Blue "Service Health Status:"
        docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
    }

    # --- Database ---
    "db-shell" {
        docker compose exec postgres psql -U ir_user -d ir_db
    }
    "db-backup" {
        New-Item -ItemType Directory -Force -Path "backups" | Out-Null
        Write-Blue "Backing up database..."
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        docker compose exec -T postgres pg_dump -U ir_user ir_db | Out-File -Encoding utf8 "backups/backup_$stamp.sql"
        Write-Green "Backup completed!"
    }
    "db-restore" {
        Write-Yellow "Restoring from latest backup..."
        $latest = Get-ChildItem "backups/*.sql" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($null -eq $latest) {
            Write-Red "No backups found in the backups folder"
        } else {
            Get-Content $latest.FullName | docker compose exec -T postgres psql -U ir_user ir_db
            Write-Green "Restore completed!"
        }
    }

    # --- Shells ---
    "shell-api" { docker compose exec api_gateway /bin/sh }
    "shell-retrieval" { docker compose exec retrieval /bin/sh }
    "shell-preprocessing" { docker compose exec preprocessing /bin/sh }

    # --- Tests ---
    "test" {
        Write-Blue "Running tests..."
        docker compose exec api_gateway pytest tests/ -v
    }
    "test-integration" {
        Write-Blue "Running integration tests..."
        docker compose -f docker-compose.yml -f docker-compose.test.yml up --abort-on-container-exit
    }

    # --- Indices ---
    "build-indices" {
        Write-Blue "Building indices..."
        docker compose -f docker-compose.yml -f docker-compose.tools.yml run --rm build_indices
    }
    "build-tfidf" {
        Write-Blue "Building TF-IDF index..."
        docker compose -f docker-compose.yml -f docker-compose.tools.yml run --rm build_tfidf
    }
    "build-bm25" {
        Write-Blue "Building BM25 index..."
        docker compose -f docker-compose.yml -f docker-compose.tools.yml run --rm build_bm25
    }
    "build-embeddings" {
        Write-Blue "Building embedding index..."
        docker compose -f docker-compose.yml -f docker-compose.tools.yml run --rm build_embeddings
    }

    # --- DB views ---
    "refresh-views" {
        Write-Blue "Refreshing materialized views..."
        docker compose -f docker-compose.yml -f docker-compose.tools.yml run --rm refresh_views
    }

    # --- Cache ---
    "clear-cache" {
        Write-Red "Clearing all cache..."
        docker compose -f docker-compose.yml -f docker-compose.tools.yml run --rm clear_cache
    }
    "cache-stats" {
        Write-Blue "Cache statistics:"
        docker compose -f docker-compose.yml -f docker-compose.tools.yml run --rm cache_stats
    }

    # --- Data ---
    "init-db" {
        Write-Blue "Initializing database schema..."
        docker compose -f docker-compose.yml -f docker-compose.tools.yml run --rm init_db
    }
    "check-data" {
        Write-Blue "Checking data status..."
        docker compose -f docker-compose.yml -f docker-compose.tools.yml run --rm check_data
    }
    "download-dataset" {
        Write-Blue "Downloading dataset..."
        docker compose -f docker-compose.yml -f docker-compose.tools.yml run --rm download_dataset
    }
    "preprocess-data" {
        Write-Blue "Preprocessing dataset..."
        docker compose -f docker-compose.yml -f docker-compose.tools.yml run --rm preprocess_dataset
    }
    "load-to-db" {
        Write-Blue "Loading data to database..."
        docker compose -f docker-compose.yml -f docker-compose.tools.yml run --rm load_to_db
    }

    # --- Evaluation ---
    "evaluate" {
        Write-Blue "Running evaluation..."
        docker compose -f docker-compose.yml -f docker-compose.tools.yml run --rm run_evaluation
    }

    # --- Cleanup ---
    "clean" {
        Write-Red "Removing stopped containers..."
        docker compose rm -f
    }
    "clean-all" {
        Write-Red "Removing all containers, images, and volumes..."
        docker compose down -v --rmi all
    }
    "clean-cache" {
        Write-Red "Cleaning Python cache..."
        Get-ChildItem -Path . -Include "__pycache__" -Recurse -Directory | Remove-Item -Recurse -Force
        Get-ChildItem -Path . -Include "*.pyc","*.pyo" -Recurse -File | Remove-Item -Force
    }
    "clean-logs" {
        Write-Red "Cleaning logs..."
        Remove-Item -Path "logs/*.log*" -Force -ErrorAction SilentlyContinue
    }

    # --- System info ---
    "stats" { docker stats }
    "inspect" {
        Write-Blue "Docker Images:"
        docker images | Select-String "ir"
        Write-Host ""
        Write-Blue "Running Containers:"
        docker compose ps
        Write-Host ""
        Write-Blue "Volumes:"
        docker volume ls | Select-String "ir"
        Write-Host ""
        Write-Blue "Networks:"
        docker network ls | Select-String "ir"
    }

    # --- Quickstart ---
    "quickstart" {
        Write-Green "Quick start: building and starting services..."
        & $PSCommandPath build
        & $PSCommandPath up
        Write-Green "Services started! Opening logs..."
        Start-Sleep -Seconds 2
        & $PSCommandPath logs
    }

    # --- Production ---
    "prod-up" {
        Write-Green "Starting in production mode..."
        docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
    }
    "prod-build" {
        Write-Blue "Building for production..."
        docker compose -f docker-compose.yml -f docker-compose.prod.yml build
    }

    # --- Init ---
    "init" {
        Write-Blue "Initializing project..."
        if (-not (Test-Path ".env")) {
            Write-Yellow "Creating .env from .env.example..."
            Copy-Item ".env.example" ".env"
            Write-Green ".env file created! Please update with your values."
        } else {
            Write-Green ".env already exists."
        }
        Write-Blue "Creating required directories..."
        New-Item -ItemType Directory -Force -Path "data/datasets","data/indices","data/processed","data/evaluation","logs","backups" | Out-Null
        Write-Blue "Building Python base image..."
        & $PSCommandPath build-python-base
        Write-Green "Project initialized!"
        Write-Host ""
        Write-Yellow "Next steps:"
        Write-Host "  1. Review and update the .env file"
        Write-Host "  2. Run '.\run.ps1 build' to build Docker images"
        Write-Host "  3. Run '.\run.ps1 up' to start services"
        Write-Host "  4. Run '.\run.ps1 download-dataset' to download data"
        Write-Host "  5. Run '.\run.ps1 build-indices' to build search indices"
    }

    default {
        Write-Red "Unknown target: $Target"
        Show-Help
    }
}
