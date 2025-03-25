# 🚀 Project Initialization Guide

This guide will help you set up and start working on the project. Follow these steps **every time** you start working.

---

## **📌 1. Clone the Repository (First Time Only)**
If you haven't already cloned the repository, run:
```bash
git clone https://github.com/ahemedbullo/CS506-Final-Project.git
cd CS506-Final-Project

✔ **This downloads the project to your local machine.**

If you've already cloned the repo, skip this step and move to **Step 2**.

---

## **📌 2. Pull the Latest Changes**
Before you start working, make sure you have the latest code and data:
```bash
git pull origin main
```
✔ **This ensures you’re working with the most up-to-date version.**

---

## **📌 3. Create and Activate Virtual Environment**
To manage dependencies, create a virtual environment:
```bash
python3 -m venv venv
```
Activate it:
- **Mac/Linux:**
  ```bash
  source venv/bin/activate
  ```
- **Windows (Command Prompt):**
  ```bash
  venv\Scripts\activate
  ```
- **Windows (PowerShell):**
  ```powershell
  venv\Scripts\Activate.ps1
  ```
✔ **You should see `(venv)` before your terminal prompt.**

---

## **📌 4. Install Dependencies**
Run the following command to install all required packages:
```bash
pip install -r requirements.txt
```
✔ **This ensures everyone is using the same dependencies.**

---

## **📌 5. Download or Generate Data**
If the `data/raw/` and `data/processed/` folders are empty, run:
```bash
python src/data_loader.py
```
✔ **This fetches the stock data and stores it locally.**

---

## **📌 6. Run the Project**
Now you can run scripts for analysis:
- **Data Cleaning:**
  ```bash
  python src/feature_engineering.py
  ```
- **Visualizations:**
  ```bash
  python src/visualization.py
  ```
- **Correlation Analysis:**
  ```bash
  python src/correlation_analysis.py
  ```
- **Model Training:**
  ```bash
  python src/model.py
  ```
✔ **Each script performs a key step in the project workflow.**

---

## **📌 7. Making Changes & Committing Code**
If you make changes, **create a new branch** before committing:
```bash
git checkout -b feature-branch-name
```
Once done, add and commit your changes:
```bash
git add .
git commit -m "Your commit message"
git push origin feature-branch-name
```
Then, open a **Pull Request (PR) on GitHub** to merge changes into `main`.

---

## **📌 8. Ending Your Work Session**
When you're done:
```bash
deactivate  # Exit the virtual environment
```
✔ **This keeps your environment clean.**

---

## **🎯 Summary**
✅ **Clone repo (`git clone`), or pull updates (`git pull`)**  
✅ **Create & activate virtual environment (`venv`)**  
✅ **Install dependencies (`pip install -r requirements.txt`)**  
✅ **Run data scripts (`python src/data_loader.py`, etc.)**  
✅ **Commit & push changes (`git commit & git push`)**  
✅ **Exit virtual environment (`deactivate`)**  

🚀 **Now you're ready to contribute to the project!**

