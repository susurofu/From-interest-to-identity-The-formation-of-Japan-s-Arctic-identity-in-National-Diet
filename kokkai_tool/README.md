# National Diet Speech Analysis Toolkit

This is a tool for analysis of Japan's National Diet Speeches. With it, you can download speeches and save them as a csv table. Current analysis features include: 1) top keyword frequency analysis and 2) topic modeling (supports Non-Negative Matrix Factorization as topic modeling method). The app supports the visualization of the results. Csvs and figures are saved in speeches/outputs.


## Requirements

- Python 
- Internet access to download required Python packages and Diet Speeches


## Installation

Clone or download this folder, then open a terminal in it:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download ja_core_news_lg
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

You can use any Python distribution and create your custom environments.


## Run the application

Naviagte the folder where you installed the files and run the application.
```bash
cd folder_with_app
python app.py
```

## How to Use

1. On **Data source**, enter one or more space-separated keywords and a date range, or browse for an existing CSV.
2. On **Keyword analysis**, choose how many top keywords to show and run the analysis.
3. On **Topic modeling**, optionally evaluate a range of topic counts, then extract topics.
4. After extraction, choose **Share** or **Count** and create the yearly topic-distribution figure.

Generated CSV and PNG files are placed in an `outputs` directory next to the selected dataset. Downloaded speech CSV files are placed in `speeches`.


## Notes

- I refactored my code for Diet research paper with LLMs, reviewed it and tested app for the proper performance. If you have courgae to develop this project, I appologize in advance. (You know about the readability of AI-code and its maintability.)
- Large datasets can take several minutes to tokenize. Anyway, the interface stays responsive and reports tokenization progress.
- Topic labels in the yearly chart use each topic's first three keywords.
- Keyword counts are based on spaCy lemmas after punctuation, spaCy stopwords, and the bundled custom Japanese stopword list are removed.
- To get the potential optimal number of topics, first, evaluate their count.
- The analysis is sensetive to the stopwords. If you meet many unrelated stopwords in topics, feel free to add new stopwords to stopwords_japanese.txt, located in services folder.

## Credentials

The code which downloads speeches from Diet API was developed by Nishio Hirokazu and can be accessed via this repository: 国会議事録API取得スクリプト.

## Citations

If you use this application, please, cite our paper:

Suslov, Artem and Ohnishi, Fujio, From Interest to Identity: The Formation of Japan's Arctic Identity in National Diet Discourse, 1949-2025 (May 10, 2026). Available at SSRN: https://ssrn.com/abstract=6751059 or http://dx.doi.org/10.2139/ssrn.6751059

