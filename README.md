# COMS-579-Project
RAG

Python version used: 3.9.0

Steps to run:
1. Run Docker Desktop/Docker daemon process.
2. Open 2 terminals.
3. First terminal: Run "docker compose up -d"
4. Second terminal, follow the below steps.
5. python version of 3.9.0 to be used:
6.  python --version
Python 3.9.0
7. Run "python -m venv ." in the root level of project(containing requirements_local.txt)
8. Activate virtual env, For Windows use: "Scripts\activate", For MacOS/Linux use "source bin/activate" 
9. "pip install -r requirements_local.txt".
10. Run "python main.py --pdf_file=genemutation.pdf --question="What is the relation between hypermutable brains and age?"". (genemutation.pdf is the file name)
11. !!Caution: Code takes more than 7mins to show output.

Video link for RAG demo for indexing, spliting, fetching nearby vector:
https://iowastate-my.sharepoint.com/:v:/g/personal/ankitj99_iastate_edu/EUq64OGM_hBDp7dMt2a3cKIBYyaCtLqWBXxUOPpYhfvHlw


