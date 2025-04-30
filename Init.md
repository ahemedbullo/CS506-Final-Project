Okay, let's update your `Init.md` guide to reflect the new workflow using the cross-platform `Makefile`. This will make it much clearer for everyone on the team how to set up and run the project.

Here's a revised version of your `Init.md`:

```markdown
# 🚀 CS506-Final-Project: Setup and Usage Guide

This guide explains how to set up the project environment and run the analysis using the provided `Makefile`.

---

## **📌 1. Clone the Repository (First Time Only)**

If you haven't already cloned the repository, run the following in your terminal:

```bash
git clone [https://github.com/ahemedbullo/CS506-Final-Project.git](https://github.com/ahemedbullo/CS506-Final-Project.git)
cd CS506-Final-Project
```

✔ **This downloads the project to your local machine.**

If you've already cloned the repo, `cd CS506-Final-Project` and proceed to **Step 2**.

---

## **📌 2. Pull the Latest Changes (Optional, Recommended)**

Before starting work, ensure you have the latest code:

```bash
git pull origin main
```

✔ **This syncs your local copy with the main repository.**

---

## **📌 3. Set Up Environment & Install Dependencies (Using Make)**

This step uses the `Makefile` to create a Python virtual environment (`venv` folder) if it doesn't exist and install all required packages from `requirements.txt`.

Run the following command in your terminal (from the project root directory):

```bash
make install
```

* You **do not** need to activate the virtual environment before running this command.
* This command handles both `venv` creation and package installation.

✔ **Your environment is now set up with the correct dependencies.**

---

## **📌 4. Activate Virtual Environment (Manual Step)**

**IMPORTANT:** Before running any analysis or testing steps using `make`, you **must manually activate** the virtual environment in your current terminal session.

Choose the command appropriate for your operating system and shell:

* **Windows (Command Prompt):**
    ```bash
    venv\Scripts\activate
    ```
* **Windows (PowerShell):**
    ```powershell
    .\venv\Scripts\Activate.ps1
    ```
    *(Note: If you get an execution policy error on PowerShell, you might need to run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once)*
* **Mac/Linux (bash, zsh):**
    ```bash
    source venv/bin/activate
    ```

✔ **You should see `(venv)` at the beginning of your terminal prompt.** Keep this terminal open and active while working.

---

## **📌 5. Run the Analysis Workflow (Using Make)**

While the `venv` is **active**, use `make` commands to run the project workflow:

* **To run the entire pipeline (Recommended):**
    *(Fetches data, processes, correlates, trains model, visualizes)*
    ```bash
    make run
    ```
* **To run individual steps:**
    * Fetch data: `make data`
    * Process data: `make process`
    * Calculate correlations: `make correlate`
    * Train the model: `make train`
    * Generate visualizations: `make visualize`
    * Run tests: `make test` (Requires `pytest` installed via `make install`)

* **To see all available commands:**
    ```bash
    make help
    ```

✔ **The `Makefile` ensures steps run in the correct order with the right dependencies.** Outputs are saved in `data/`, `results/`, `models/`, and `correlation_results.csv`, `model_results.csv`.

---

## **📌 6. Making Changes & Committing Code (Git Workflow)**

Follow standard Git practices:

1.  Ensure you are on the `main` branch and have the latest changes (`git pull origin main`).
2.  Create a new branch for your feature or fix: `git checkout -b your-feature-branch-name`
3.  Make your code changes.
4.  Add and commit your changes:
    ```bash
    git add .
    git commit -m "Brief description of your changes"
    ```
5.  Push your branch to GitHub: `git push origin your-feature-branch-name`
6.  Go to the GitHub repository page and open a **Pull Request (PR)** to merge your branch into `main`. Ensure your code works and tests pass (GitHub Actions will run `make test`).

---

## **📌 7. Ending Your Work Session**

When you finish working, deactivate the virtual environment:

```bash
deactivate
```

✔ **This exits the `venv` and cleans up your terminal session.**

---

## **🎯 Quick Summary (Typical Workflow)**

1.  `cd CS506-Final-Project`
2.  `git pull origin main` *(Optional, recommended)*
3.  `make install` *(Only if first time or requirements changed)*
4.  Activate venv (e.g., `.\venv\Scripts\Activate.ps1` or `source venv/bin/activate`)
5.  Run analysis (e.g., `make run` or other `make` targets)
6.  *(If making changes)* Use Git workflow (new branch, commit, push, PR)
7.  `deactivate` *(When done)*

