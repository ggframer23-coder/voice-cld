.PHONY: help install install-dev clean clean-all test format lint run watch venv

# Default target
help:
	@echo "Voice-to-Text Transcription Tool - Makefile Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install       - Create venv and install dependencies"
	@echo "  make install-dev   - Install with development dependencies"
	@echo ""
	@echo "Running:"
	@echo "  make run FILE=path/to/audio.mp3  - Transcribe single file"
	@echo "  make batch DIR=path/to/audio/    - Batch process directory"
	@echo "  make watch DIR=path/to/audio/    - Watch directory for new files"
	@echo ""
	@echo "Development:"
	@echo "  make test          - Run tests with pytest"
	@echo "  make format        - Format code with black"
	@echo "  make lint          - Lint code with ruff"
	@echo "  make check         - Run format + lint + test"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         - Remove Python cache and build artifacts"
	@echo "  make clean-all     - Remove venv, cache, models, and outputs"
	@echo ""
	@echo "Search (when implemented):"
	@echo "  make search QUERY='search term'  - Search transcripts"
	@echo "  make rebuild-index - Rebuild FAISS vector index"

# Python and venv settings
PYTHON := python3
VENV := .venv
BIN := $(VENV)/bin
PYTHON_VENV := $(BIN)/python
PIP := $(BIN)/pip

# Model and output settings
MODEL ?= small
OUTPUT_DIR ?= ./transcripts
LANGUAGE ?= auto

# Create virtual environment
venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created at $(VENV)"

# Install production dependencies
install: venv
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -e .
	@echo ""
	@echo "✓ Installation complete!"
	@echo "  Activate venv: source $(VENV)/bin/activate"
	@echo "  Run transcription: make run FILE=audio.mp3"

# Install development dependencies
install-dev: venv
	@echo "Installing development dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@echo ""
	@echo "✓ Development installation complete!"
	@echo "  Run tests: make test"
	@echo "  Format code: make format"

# Run single file transcription
run:
	@if [ -z "$(FILE)" ]; then \
		echo "Error: FILE not specified"; \
		echo "Usage: make run FILE=path/to/audio.mp3 [MODEL=small]"; \
		exit 1; \
	fi
	@echo "Transcribing: $(FILE)"
	@echo "Model: $(MODEL)"
	$(PYTHON_VENV) -m src.cli "$(FILE)" --model $(MODEL) \
		$(if $(OUTPUT),--output "$(OUTPUT)",) \
		$(if $(filter-out auto,$(LANGUAGE)),--language $(LANGUAGE),)

# Batch process directory
batch:
	@if [ -z "$(DIR)" ]; then \
		echo "Error: DIR not specified"; \
		echo "Usage: make batch DIR=path/to/audio/ [MODEL=small]"; \
		exit 1; \
	fi
	@echo "Batch processing: $(DIR)"
	@echo "Model: $(MODEL)"
	@echo "Output: $(OUTPUT_DIR)"
	$(PYTHON_VENV) -m src.batch "$(DIR)" \
		--output-dir "$(OUTPUT_DIR)" \
		--model $(MODEL) \
		$(if $(filter-out auto,$(LANGUAGE)),--language $(LANGUAGE),) \
		$(if $(ONLY_NEW),--only-new,)

# Watch directory for new files
watch:
	@if [ -z "$(DIR)" ]; then \
		echo "Error: DIR not specified"; \
		echo "Usage: make watch DIR=path/to/audio/ [MODEL=small]"; \
		exit 1; \
	fi
	@echo "Watching directory: $(DIR)"
	@echo "Model: $(MODEL)"
	@echo "Output: $(OUTPUT_DIR)"
	$(PYTHON_VENV) -m src.watcher "$(DIR)" \
		--output-dir "$(OUTPUT_DIR)" \
		--model $(MODEL) \
		$(if $(filter-out auto,$(LANGUAGE)),--language $(LANGUAGE),)

# Search transcripts (semantic search)
search:
	@if [ -z "$(QUERY)" ]; then \
		echo "Error: QUERY not specified"; \
		echo "Usage: make search QUERY='search term'"; \
		exit 1; \
	fi
	@echo "Searching for: $(QUERY)"
	$(PYTHON_VENV) -m src.search "$(QUERY)" \
		$(if $(DATE_FROM),--date-from $(DATE_FROM),) \
		$(if $(DATE_TO),--date-to $(DATE_TO),)

# Rebuild FAISS vector index
rebuild-index:
	@echo "Rebuilding FAISS index from transcripts..."
	$(PYTHON_VENV) -m src.vectorstore rebuild "$(OUTPUT_DIR)"

# Show index statistics
index-stats:
	@echo "FAISS index statistics:"
	$(PYTHON_VENV) -m src.vectorstore stats

# Run tests
test:
	@echo "Running tests..."
	$(BIN)/pytest tests/ -v --cov=src --cov-report=term-missing

# Format code with black
format:
	@echo "Formatting code with black..."
	$(BIN)/black src/ tests/

# Lint code with ruff
lint:
	@echo "Linting code with ruff..."
	$(BIN)/ruff check src/ tests/

# Lint and auto-fix
lint-fix:
	@echo "Linting and fixing code..."
	$(BIN)/ruff check --fix src/ tests/

# Run all checks (format + lint + test)
check: format lint test
	@echo ""
	@echo "✓ All checks passed!"

# Clean Python cache and build artifacts
clean:
	@echo "Cleaning Python cache and build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.py,cover" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/
	@echo "✓ Cleaned Python cache and build artifacts"

# Clean everything (including venv, models, and outputs)
clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf $(VENV)
	@echo "Removing downloaded models..."
	rm -rf models/
	@echo "Removing FAISS index..."
	rm -f transcripts/metadata/faiss_index.bin
	rm -f transcripts/metadata/id_mapping.json
	@echo ""
	@echo "✓ Full cleanup complete!"
	@echo "  Note: Transcript JSON files preserved"
	@echo "  To remove transcripts: rm -rf transcripts/json/"

# Development workflow shortcuts
dev-setup: install-dev
	@echo ""
	@echo "Development environment ready!"
	@echo "Try: make test"

# Quick transcription with tiny model (for testing)
quick-test:
	@if [ -z "$(FILE)" ]; then \
		echo "Error: FILE not specified"; \
		echo "Usage: make quick-test FILE=audio.mp3"; \
		exit 1; \
	fi
	$(PYTHON_VENV) -m src.cli "$(FILE)" --model tiny

# Show project info
info:
	@echo "Project: Voice-to-Text Memory Aide"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Venv: $(VENV)"
	@if [ -d "$(VENV)" ]; then \
		echo "Status: ✓ Virtual environment exists"; \
		if [ -f "$(PYTHON_VENV)" ]; then \
			echo "Python (venv): $$($(PYTHON_VENV) --version)"; \
		fi \
	else \
		echo "Status: ✗ Virtual environment not found"; \
		echo "Run: make install"; \
	fi
