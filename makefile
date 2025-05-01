# Makefile for Stock Analysis Project (Cross-Platform Focus - CORRECTED MKDIR/CLEAN)

# --- Variables ---
# Use 'python' - assumes it's correct command after venv activation
PYTHON = python
# Use Python's pip module directly for installs
PIP = $(PYTHON) -m pip
SRC_DIR = src
DATA_DIR = data
RAW_DATA_DIR = $(DATA_DIR)/raw
PROCESSED_DATA_DIR = $(DATA_DIR)/processed
VIS_DIR = $(DATA_DIR)/visualizations
RESULTS_DIR = results
MODELS_DIR = models
CORRELATION_RESULTS = correlation_results.csv
# Adjust if your model.py saves results to the results directory
MODEL_RESULTS_CSV = $(RESULTS_DIR)/model_performance_results.csv
REQUIREMENTS = requirements.txt
TEST_DIR = tests
VENV_DIR = venv

# Determine Python executable within venv based on OS guess
# This is imperfect but covers common cases for the install target
ifeq ($(OS),Windows_NT)
	VENV_PYTHON = $(VENV_DIR)/Scripts/python.exe
	# --- THIS IS THE FIX --- Use POSIX standard mkdir -p
	MKDIR = mkdir -p $@
	# --- THIS IS THE FIX --- Use Python for cleaning on Windows too
	CLEAN_CMD = $(PYTHON) -c "import shutil; shutil.rmtree('{}', ignore_errors=True)"
else
	VENV_PYTHON = $(VENV_DIR)/bin/python
	MKDIR = mkdir -p $@
	CLEAN_CMD = rm -rf "{}"
endif


# --- Phony Targets ---
.PHONY: all install data process correlate visualize train test run clean help setup check_venv

# --- Default Target ---
all: run

# --- Installation ---
# Creates venv and installs requirements using the venv's python/pip
# Does NOT require manual activation beforehand.
install: $(VENV_DIR)/.installed

$(VENV_DIR)/.installed: $(REQUIREMENTS)
	@echo "Setting up virtual environment and installing dependencies..."
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment '$(VENV_DIR)' using system '$(PYTHON)'..."; \
		$(PYTHON) -m venv $(VENV_DIR); \
		echo "Virtual environment created."; \
	else \
		echo "Virtual environment '$(VENV_DIR)' already exists."; \
	fi
	@echo "Installing/Updating packages using venv pip..."
	@$(VENV_PYTHON) -m pip install --upgrade pip setuptools wheel certifi
	@$(VENV_PYTHON) -m pip install -r $(REQUIREMENTS)
	@echo "Installation complete. Activate environment manually before running other targets."
	@echo "Windows CMD: $(VENV_DIR)\\Scripts\\activate"
	@echo "Windows PowerShell: .\\$(VENV_DIR)\\Scripts\\Activate.ps1"
	@echo "macOS/Linux: source $(VENV_DIR)/bin/activate"
	@touch $(VENV_DIR)/.installed # Mark as complete

# --- Venv Check ---
# Simple check target. Does not halt execution but prints a warning.
# Users should activate manually.
check_venv:
	@if [ -z "$(VENV)" ]; then \
		echo "[Warning] Virtual environment does not appear to be active."; \
		echo "[Warning] Please activate it manually before running targets like 'data', 'process', 'train', 'test', etc."; \
		echo "[Warning] Win CMD: venv\\Scripts\\activate | Win PS: .\\venv\\Scripts\\Activate.ps1 | macOS/Linux: source venv/bin/activate"; \
	fi

# --- Data Pipeline ---
# These targets now assume the user has activated the venv manually first.
data: check_venv $(RAW_DATA_DIR)
	@echo "--- Fetching Raw Data ---"
	@$(PYTHON) $(SRC_DIR)/data_loader.py
	@echo "Data fetching complete."

process: check_venv data $(PROCESSED_DATA_DIR)
	@echo "--- Processing Raw Data ---"
	@$(PYTHON) $(SRC_DIR)/data_processor.py
	@echo "Data processing complete."

correlate: check_venv process
	@echo "--- Calculating Correlations ---"
	@$(PYTHON) $(SRC_DIR)/correlation_analysis.py
	@echo "Correlation analysis complete."

# Ensure viz directory exists before running
visualize: check_venv process $(VIS_DIR)
	@echo "--- Generating Visualizations ---"
	@$(PYTHON) $(SRC_DIR)/visualization.py
	@echo "Visualization generation complete."

# Ensure results/models directories exist before running train
train: check_venv correlate $(MODELS_DIR) $(RESULTS_DIR)
	@echo "--- Training Model ---"
	@$(PYTHON) $(SRC_DIR)/model.py
	@echo "Model training complete."

# --- Testing ---
test: check_venv install # Ensure packages are installed, then check activation
	@echo "--- Running Tests ---"
	@$(PYTHON) -m pytest $(TEST_DIR)
	@echo "Testing complete."


# --- Helper Targets for Directory Creation ---
# These use the MKDIR variable defined above based on OS
$(RAW_DATA_DIR):
	@echo "Ensuring directory exists: $@"
	@$(MKDIR)

$(PROCESSED_DATA_DIR):
	@echo "Ensuring directory exists: $@"
	@$(MKDIR)

$(VIS_DIR):
	@echo "Ensuring directory exists: $@"
	@$(MKDIR)

$(RESULTS_DIR):
	@echo "Ensuring directory exists: $@"
	@$(MKDIR)

$(MODELS_DIR):
	@echo "Ensuring directory exists: $@"
	@$(MKDIR)


# --- Full Workflow ---
# Assumes manual activation before running 'make run'
# Ensure all directories needed by subsequent steps are listed as dependencies
run: check_venv install data process correlate $(RESULTS_DIR) $(MODELS_DIR) $(VIS_DIR) train visualize
	@echo "--- Project Workflow Executed Successfully ---"
	@echo "[Reminder] Ensure venv was activated before running 'make run'."

# --- Cleaning ---
# Uses Python for removing directories/files where possible for portability
# Updated path for MODEL_RESULTS_CSV based on revised model.py saving location
clean:
	@echo "Cleaning generated files..."
	$(call CLEAN_CMD,$(RAW_DATA_DIR))
	$(call CLEAN_CMD,$(PROCESSED_DATA_DIR))
	$(call CLEAN_CMD,$(VIS_DIR))
	$(call CLEAN_CMD,$(RESULTS_DIR))
	$(call CLEAN_CMD,$(MODELS_DIR))
	$(PYTHON) -c "import os; os.path.exists('$(CORRELATION_RESULTS)') and os.remove('$(CORRELATION_RESULTS)')"
	$(PYTHON) -c "import os; os.path.exists('$(MODEL_RESULTS_CSV)') and os.remove('$(MODEL_RESULTS_CSV)')"
	# Clean python cache files
	$(PYTHON) -c "import shutil, glob; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('$(SRC_DIR)/__pycache__')]"
	$(PYTHON) -c "import shutil, glob; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('$(TEST_DIR)/__pycache__')]"
	$(call CLEAN_CMD,.pytest_cache)
	$(PYTHON) -c "import os; os.path.exists('$(VENV_DIR)/.installed') and os.remove('$(VENV_DIR)/.installed')"
	@echo "Clean complete. Virtual environment $(VENV_DIR) itself was not removed."

# --- Help ---
help:
	@echo "Available commands:"
	@echo "  make install     : Set up virtual environment $(VENV_DIR) and install dependencies."
	@echo "                   : (Does not require manual activation)"
	@echo "  --- Requires manual venv activation BEFORE running: ---"
	@echo "  make data        : Fetch raw stock data into $(RAW_DATA_DIR)."
	@echo "  make process     : Clean and process raw data into $(PROCESSED_DATA_DIR)."
	@echo "  make correlate   : Calculate and save stock correlations."
	@echo "  make visualize   : Generate visualizations into $(VIS_DIR)."
	@echo "  make train       : Train model, save results/model into $(RESULTS_DIR)/, $(MODELS_DIR)/."
	@echo "  make test        : Run project tests from $(TEST_DIR)/ (requires pytest)."
	@echo "  make run         : Run the full workflow (install, data, process, correlate, train, visualize)."
	@echo "  make clean       : Remove generated data, results, models, and caches."
	@echo "  make help        : Show this help message."
	@echo "---------------------------------------------------------------------"
	@echo "Manual Activation Commands:"
	@echo "  Windows CMD:         $(VENV_DIR)\\Scripts\\activate"
	@echo "  Windows PowerShell:  .\\$(VENV_DIR)\\Scripts\\Activate.ps1"
	@echo "  macOS/Linux (bash):  source $(VENV_DIR)/bin/activate"
	@echo "---------------------------------------------------------------------"


# --- Legacy Setup Target ---
setup: install

# --- Debug Target ---
# Add this if needed again for debugging which python make uses by default
# print_python_path:
#	@echo "System Python potentially used by make to create venv:"
#	@$(PYTHON) --version
#	@$(PYTHON) -c "import sys; print(sys.executable)"