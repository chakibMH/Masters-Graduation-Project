# Academic Expert Search - Graduation Project

This repository contains the source code, datasets, and documentation for our **Master's graduation project**, which focuses on **scientific expert search using deep learning**. Our project was presented at **TAMARICS 2022**, an international competition held at the University of Tamanghasset, Algeria.

## 📌 Project Overview
Scientific expert search is a crucial task in the academic world, used for identifying researchers for supervision, evaluation, or collaboration. Our work aims to enhance expert search systems by leveraging **deep learning** and **natural language processing (NLP)** techniques, specifically transformer-based models like **BERT, SciBERT, and RoBERTa**. 

### 🔍 Key Contributions
- **New indexing approach**: Sentence-based indexing instead of full-document indexing.
- **Query expansion**: Augmenting search queries using definitions to improve accuracy.
- **Score distribution adjustment**: Mitigating the dominance of highly prolific authors.
- **Dataset creation**: A new test corpus extracted from ACM for better benchmarking.

## 📂 Repository Structure
```
/Project-Name
│── /code                # Source code for expert search system
│── /data                # Datasets and processed files
│── /docs                # Documentation and articles
│   │── full-article.pdf   # Detailed article about the project
│   │── TAMARICS_2022.pdf  # Summary article presented at TAMARICS 2022
│   │── slides.pdf         # Presentation slides
│── README.md             # Overview of the project
│── requirements.txt      # Dependencies (if applicable)
│── LICENSE              # License file (if open-source)
```

## 📖 Documentation
- **[TAMARICS 2022 Article](./docs/TAMARICS_2022.pdf)**: A summary of our project presented at an international competition.
- **[Full Project Report](./docs/full-article.pdf)**: A comprehensive document detailing our methods and findings.
- **[Presentation Slides](./docs/slides.pdf)**: Key insights from our work in a slide format.

## ⚙️ Installation & Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the expert search model:
   ```sh
   python main.py
   ```

## 🏆 Acknowledgments
This project was developed as part of our **Master’s thesis** at **University of Science and Technology Houari Boumediene** and was supervised by [Professor's Name]. Special thanks to **TAMARICS 2022** for providing a platform to present our research.

---
📌 *For more details, check the articles in the `/docs` folder.*
