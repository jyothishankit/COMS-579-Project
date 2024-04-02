# COMS-579-Project
RAG

Python version used: 3.9.0

Steps to run:
1. Run Docker Desktop/Docker daemon process.
2. Open 2 terminals.
3. Run "docker compose up -d"
4. Second terminal, first run:
4(i)   python version of 3.9.0 to be used:
4(ii)  python --version
Python 3.9.0
4(iii) Run "python -m venv ." in the root level of project(containing requirements_local.txt)
4(iv) Activate virtual env, For Windows use: "Scripts\activate", For MacOS/Linux use "source .venv/bin/activate" 
4(v) "pip install -r requirements_local.txt".
5. Run "python main.py genemutation.pdf". (genemutation.pdf is the file name)


Video link for RAG demo for indexing, spliting, fetching nearby vector:
https://iowastate-my.sharepoint.com/:v:/g/personal/ankitj99_iastate_edu/EUq64OGM_hBDp7dMt2a3cKIBYyaCtLqWBXxUOPpYhfvHlw


