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
│── /Evaluation            # Scripts used to evaluate our methods, and compare it with previous works
│── /Results                # An exaustive list of all our results described in the article (in the form of binary files and csv). Contains also plottings
│── /Scraping                #
│── /Test_&_Tutos                #
│── /Web Site                #
│── /Weighted_Sampling   # Utility package used in the code
│── /Masters-Graduation-Project                # Script responsible of building the FAISS Index, for efficient similarity search [FAISS - AI Meta](https://ai.meta.com/tools/faiss/)
│── /main_&_execution               # Scripts responsible for the Back End connexion with the app, in addition to running the indexation, query augmentation and similarity search
│── /docs                # Documentation and articles
│   │── full-article.pdf   # Detailed article about the project (in French)
│   │── TAMARICS_2022.pdf  # Summary article presented at TAMARICS 2022 (English)
│   │── slides.pdf         # Presentation slides (English)
│── README.md             # Overview of the project

```

## 📖 Documentation
- **[TAMARICS 2022 Article](./docs/TAMARICS_2022.pdf)**: A summary of our project presented at an international competition.
- **[Full Project Report](./docs/full-article.pdf)**: A comprehensive document detailing our methods and findings.
- **[Presentation Slides](./docs/slides.pdf)**: Key insights from our work in a slide format.

## ⚙️ Installation & Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/Masters-Graduation-Project.git
   cd Masters-Graduation-Project
   ```

   ```
2. Run the web app:
   ```sh
   python main.py
   ```

## 🏆 Acknowledgments
This project was developed as part of our **Master’s thesis** at **University of Science and Technology Houari Boumediene** and was supervised by [Professor's Name]. Special thanks to **TAMARICS 2022** for providing a platform to present our research.

---
📌 *For more details, check the articles in the `/docs` folder.*
