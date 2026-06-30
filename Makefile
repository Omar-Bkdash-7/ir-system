# =============================================================================
# Makefile - Simplified Docker and Project Management Commands
# =============================================================================
# Usage: make <target>
# Example: make build, make up, make logs
# =============================================================================

.PHONY: help build build-no-cache up down restart logs clean test shell

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
RED := \033[0;31m
YELLOW := \033[0;33m
NC := \033[0m # No Color

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
help: ## Show this help message
	@echo "$(BLUE)IR System - Docker Management$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# -----------------------------------------------------------------------------
# Docker Build Commands
# -----------------------------------------------------------------------------
build: build-python-base ## Build all Docker images
	@echo "$(BLUE)Building all Docker images...$(NC)"
	docker compose build

build-no-cache: build-python-base-no-cache ## Build all images without using cache
	@echo "$(BLUE)Building all images (no cache)...$(NC)"
	docker compose build --no-cache

build-python-base: ## Build the base Python image (required first)
	@echo "$(BLUE)Building Python base image...$(NC)"
	docker build -f docker/python-base.Dockerfile -t ir-python-base:latest .

build-python-base-no-cache: ## Build base image without cache
	@echo "$(BLUE)Building Python base image (no cache)...$(NC)"
	docker build --no-cache -f docker/python-base.Dockerfile -t ir-python-base:latest .

build-base: ## Build only base image
	@echo "$(BLUE)Building base image...$(NC)"
	docker build -f docker/base.Dockerfile -t ir-base:latest .

build-ml: ## Build ML image
	@echo "$(BLUE)Building ML image...$(NC)"
	docker build -f docker/ml.Dockerfile -t ir-ml:latest .

build-frontend: ## Build frontend image
	@echo "$(BLUE)Building frontend image...$(NC)"
	docker build -f docker/frontend.Dockerfile -t ir-frontend:latest .

# -----------------------------------------------------------------------------
# Docker Compose Commands
# -----------------------------------------------------------------------------
up: ## Start all services
	@echo "$(GREEN)Starting all services...$(NC)"
	docker compose up -d

up-build: ## Build and start all services
	@echo "$(GREEN)Building and starting all services...$(NC)"
	docker compose up -d --build

down: ## Stop all services
	@echo "$(RED)Stopping all services...$(NC)"
	docker compose down

down-volumes: ## Stop all services and remove volumes
	@echo "$(RED)Stopping services and removing volumes...$(NC)"
	docker compose down -v

restart: ## Restart all services
	@echo "$(YELLOW)Restarting all services...$(NC)"
	docker compose restart

# -----------------------------------------------------------------------------
# Development Commands
# -----------------------------------------------------------------------------
dev: ## Start in development mode with hot reload
	@echo "$(GREEN)Starting in development mode...$(NC)"
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up

dev-build: ## Build and start in development mode
	@echo "$(GREEN)Building and starting in development mode...$(NC)"
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build

# -----------------------------------------------------------------------------
# Logs Commands
# -----------------------------------------------------------------------------
logs: ## Show logs from all services
	docker compose logs -f

logs-api: ## Show logs from API gateway
	docker compose logs -f api_gateway

logs-retrieval: ## Show logs from retrieval service
	docker compose logs -f retrieval

logs-frontend: ## Show logs from frontend
	docker compose logs -f frontend

# -----------------------------------------------------------------------------
# Service Management
# -----------------------------------------------------------------------------
ps: ## Show running services
	docker compose ps

stop: ## Stop all services (without removing)
	docker compose stop

start: ## Start stopped services
	docker compose start

# -----------------------------------------------------------------------------
# Database Commands
# -----------------------------------------------------------------------------
db-shell: ## Open PostgreSQL shell
	docker compose exec postgres psql -U ir_user -d ir_db

db-backup: ## Backup database to ./backups/
	@mkdir -p backups
	@echo "$(BLUE)Backing up database...$(NC)"
	docker compose exec -T postgres pg_dump -U ir_user ir_db > backups/backup_$$(date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)Backup completed!$(NC)"

db-restore: ## Restore database from latest backup
	@echo "$(YELLOW)Restoring from latest backup...$(NC)"
	docker compose exec -T postgres psql -U ir_user ir_db < $$(ls -t backups/*.sql | head -1)
	@echo "$(GREEN)Restore completed!$(NC)"

# -----------------------------------------------------------------------------
# Service Shell Access
# -----------------------------------------------------------------------------
shell-api: ## Open shell in API gateway container
	docker compose exec api_gateway /bin/bash

shell-retrieval: ## Open shell in retrieval container
	docker compose exec retrieval /bin/bash

shell-preprocessing: ## Open shell in preprocessing container
	docker compose exec preprocessing /bin/bash

# -----------------------------------------------------------------------------
# Testing Commands
# -----------------------------------------------------------------------------
test: ## Run tests in containers
	@echo "$(BLUE)Running tests...$(NC)"
	docker compose exec api_gateway pytest tests/ -v

test-integration: ## Run integration tests
	@echo "$(BLUE)Running integration tests...$(NC)"
	docker compose -f docker-compose.yml -f docker-compose.test.yml up --abort-on-container-exit

# -----------------------------------------------------------------------------
# Index Building Commands
# -----------------------------------------------------------------------------
build-indices: ## Build all indices
	@echo "$(BLUE)Building indices...$(NC)"
	docker compose run --rm build_indices

build-tfidf: ## Build TF-IDF index only
	@echo "$(BLUE)Building TF-IDF index...$(NC)"
	docker compose run --rm build_indices python scripts/build_tfidf_index.py

build-bm25: ## Build BM25 index only
	@echo "$(BLUE)Building BM25 index...$(NC)"
	docker compose run --rm build_indices python scripts/build_bm25_index.py

build-embeddings: ## Build embedding index only
	@echo "$(BLUE)Building embedding index...$(NC)"
	docker compose run --rm build_indices python scripts/build_embedding_index.py

# -----------------------------------------------------------------------------
# Data Management
# -----------------------------------------------------------------------------
download-dataset: ## Download dataset using ir-datasets
	@echo "$(BLUE)Downloading dataset...$(NC)"
	docker compose run --rm build_indices python scripts/download_dataset.py

preprocess-data: ## Preprocess downloaded dataset
	@echo "$(BLUE)Preprocessing dataset...$(NC)"
	docker compose run --rm build_indices python scripts/preprocess_dataset.py

load-to-db: ## Load processed data to database
	@echo "$(BLUE)Loading data to database...$(NC)"
	docker compose run --rm build_indices python scripts/load_documents_to_db.py

# -----------------------------------------------------------------------------
# Evaluation Commands
# -----------------------------------------------------------------------------
evaluate: ## Run full evaluation
	@echo "$(BLUE)Running evaluation...$(NC)"
	docker compose run --rm build_indices python scripts/evaluate_retrieval.py

# -----------------------------------------------------------------------------
# Cleanup Commands
# -----------------------------------------------------------------------------
clean: ## Remove stopped containers
	@echo "$(RED)Removing stopped containers...$(NC)"
	docker compose rm -f

clean-all: ## Remove containers, images, and volumes
	@echo "$(RED)Removing all containers, images, and volumes...$(NC)"
	docker compose down -v --rmi all

clean-cache: ## Clean Python cache files
	@echo "$(RED)Cleaning Python cache...$(NC)"
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-logs: ## Clean log files
	@echo "$(RED)Cleaning logs...$(NC)"
	rm -rf logs/*.log*

# -----------------------------------------------------------------------------
# System Information
# -----------------------------------------------------------------------------
stats: ## Show container resource usage
	docker stats

inspect: ## Show detailed system information
	@echo "$(BLUE)Docker Images:$(NC)"
	@docker images | grep ir
	@echo ""
	@echo "$(BLUE)Running Containers:$(NC)"
	@docker compose ps
	@echo ""
	@echo "$(BLUE)Volumes:$(NC)"
	@docker volume ls | grep ir
	@echo ""
	@echo "$(BLUE)Networks:$(NC)"
	@docker network ls | grep ir

health: ## Check health status of all services
	@echo "$(BLUE)Service Health Status:$(NC)"
	@docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

# -----------------------------------------------------------------------------
# Quick Start
# -----------------------------------------------------------------------------
quickstart: ## Quick start: build, start, and show logs
	@echo "$(GREEN)Quick start: building and starting services...$(NC)"
	@make build-python-base
	@make build
	@make up
	@echo "$(GREEN)Services started! Opening logs...$(NC)"
	@sleep 2
	@make logs

# -----------------------------------------------------------------------------
# Production Commands
# -----------------------------------------------------------------------------
prod-up: ## Start in production mode
	@echo "$(GREEN)Starting in production mode...$(NC)"
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

prod-build: ## Build for production
	@echo "$(BLUE)Building for production...$(NC)"
	docker compose -f docker-compose.yml -f docker-compose.prod.yml build

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
init: ## Initialize project (first time setup)
	@echo "$(BLUE)Initializing project...$(NC)"
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)Creating .env from .env.example...$(NC)"; \
		cp .env.example .env; \
		echo "$(GREEN).env file created! Please update with your values.$(NC)"; \
	else \
		echo "$(GREEN).env already exists.$(NC)"; \
	fi
	@echo "$(BLUE)Creating required directories...$(NC)"
	@mkdir -p data/datasets data/indices data/processed data/evaluation logs backups
	@echo "$(BLUE)Building Python base image...$(NC)"
	@make build-python-base
	@echo "$(GREEN)Project initialized!$(NC)"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Review and update .env file"
	@echo "  2. Run 'make build' to build Docker images"
	@echo "  3. Run 'make up' to start services"
	@echo "  4. Run 'make download-dataset' to download data"
	@echo "  5. Run 'make build-indices' to build search indices"
