# Makefile for managing the Mini-SOAR application lifecycle.
# It automatically detects whether to use 'docker-compose' or 'docker compose'.

# --- Automatic Command Detection ---
# Check if the classic 'docker-compose' command is available.
# The '2>/dev/null' suppresses any "command not found" errors.
COMPOSE_CMD := $(shell command -v docker-compose 2>/dev/null)

# If the COMPOSE_CMD variable is empty, it means 'docker-compose' was not found.
# In that case, we fall back to the modern 'docker compose' syntax.
ifeq ($(COMPOSE_CMD),)
  COMPOSE_CMD := docker compose
endif
# --- End of Command Detection ---


# Phony targets are not actual files. This prevents conflicts.
.PHONY: all build up down logs clean

# The default command when running 'make' is 'make all', which runs 'make up'.
all: up

# Build or rebuild the service images defined in the compose file.
build:
	@echo "Building Docker image(s) using '$(COMPOSE_CMD)'..."
	@$(COMPOSE_CMD) build

# Create and start containers. Includes '--build' to ensure images are up-to-date.
up:
	@echo "Starting Mini-SOAR application using '$(COMPOSE_CMD)'..."
	@$(COMPOSE_CMD) up --build -d
	@echo ""
	@echo "Application is starting up in detached mode."
	@echo "View logs with: make logs"
	@echo "Access the app at: http://localhost:8501"


# Stop and remove containers, networks, and volumes created by 'up'.
down:
	@echo "Stopping Mini-SOAR application using '$(COMPOSE_CMD)'..."
	@$(COMPOSE_CMD) down

# Follow the real-time logs from the application service.
logs:
	@echo "Following application logs... (Press Ctrl+C to exit)"
	@$(COMPOSE_CMD) logs -f

# A full cleanup: stops containers and removes generated files for a fresh start.
clean: down
	@echo "Cleaning up generated files and directories..."
	@rm -rf ./models
	@rm -rf ./data
	@echo "Cleanup complete."
