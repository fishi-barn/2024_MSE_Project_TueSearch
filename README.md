# Tuesearch
**Authors**: Stephan Amann, Tanja Huber, Markus Potthast, Tina Truong

**Date**: 22.07.2024

GitHub repository for the final project of the 'Modern Search Engines' class during the summer term at the Eberhard-Karls Universität Tübingen.

## Objectives
- Web Crawling & Indexing for English content related to Tübingen only.
- Query Processing.
- Search Result Presentation.

## Repository Structure

The repository is organized into several directories:

- [`.streamlit/`](.streamlit): Contains the theme for the web page.
- [`certs/`](certs): Contains certificates for secure connections.
- [`data/`](data): Contains the frontier, corpus, index, and other intermediate saves.
- [`static/`](static): Contains the web page CSS.

## Running the Code

### 1. Clone the Repository

- Need to download index which is a large file (~120mb) => need to install git-lfs
  - Download and install git-lfs
    - using PackageCloud.io
      - `!curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash`
      - `sudo apt-get install git-lfs` (for debian-based linux systems such as ubuntu)
    - or download tar.gz and install manually from their official [website](https://git-lfs.com/).
  - Configure git for lfs (will only download a pointer to the lfs file)
    - `git lfs install --skip-smudge`
  - Clone the repository
    - `git clone https://github.com/fishi-barn/2024_MSE_Project_TueSearch.git`
  - In the cloned repository, pull the index with lfs
    - `git lfs pull --include=index.json`

### 2. Set Up a Virtual Environment (Optional but Recommended)

- `bash`
- `python -m venv venv`
- `source venv/bin/activate  # On Windows use venv\Scripts\activate`

### 3. Install Required Packages

- Built in `python 3.10`
- Requirements file is supplied for python packages
    - `pip install -r requirements.txt`

### 4. Run The Notebooks

- Single queries and query files are possible
    - `tse.search("query")`
    - `tse.search_from_file("queries.txt")`
- Using jupyter
    - `jupyter notebook MSE_24_Group_Projects.ipynb`
- Run in IDE with notebook support

### 5. Run Web Page with Streamlit

It is possible to run streamlit over the notebook or by running streamlit in terminal

- `streamlit run app.py`
