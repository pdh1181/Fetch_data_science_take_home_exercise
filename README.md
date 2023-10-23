# Fetch Data Science take home exercise
## Overview
The Offer Search Tool is a web application built with Flask that allows users to search for offers based on brands, product categories, and retailers. It utilizes a BERT-based model for text similarity to provide relevant search results. Users can enter a query, and the tool will return a list of offers along with their similarity scores.

## Prerequisites
- Python 3.x
- Flask
- pandas
- scikit-learn
- transformers (for BERT model)

## Installation

1. Clone the repository:
  ```bash
  https://github.com/pdh1181/Fetch_data_science_take_home_exercise.git
  ```
2. Navigate to the project directory
  ```bash
  cd Fetch_data_science_take_home_exercise
  ```
3. Install the required packages using pip:
  You can install the required packages using the following command:
  ```bash
  pip3 install pandas Flask scikit-learn transformers
  ```
## Usage
1. Run the Flask application
  ```bash
  python3 app.py
  ```
2. Open your web browser and go to [http://localhost:5000](http://127.0.0.1:5000/) to access the application.
3. Enter your search query and click the search button to view the relevant offers and their similarity scores.

- The exercise involved generating a dataset (**data.csv**) using information from **brand_category.csv**, **categories.csv**, and **offer_retailer.csv** files. The detailed process of data generation can be found in the **data_processing.ipynb** notebook.
- **data_processing.ipynb**: This Jupyter Notebook outlines the step-by-step process of generating the **data.csv** file. It provides a clear and comprehensive explanation of how the data was processed and prepared for analysis.
- **take_home.ipynb**: This Jupyter Notebook gives a high-level overview of the exercise, including the methods used for text analysis. It serves as a summary of the key steps taken to filter the offer according to user query.

## How it works

1. **User Input:**
- **Query:** Users enter their search query into the application. Queries can include brand names, product categories, or retailer names.

2. **Data Processing:**
- **Data Preparation:** The tool processes a dataset containing information about various offers, including details like brands, product categories, retailers, and specific offers.
- **Text Preprocessing:** Text data from the dataset and user queries are preprocessed. This step involves tokenization, removing stop words, and converting text to lowercase to ensure consistent and accurate matching.

3. **BERT Model for Text Similarity:**
- **Feature Extraction:** The tool employs a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model. BERT is a transformer-based deep learning model designed to understand context and semantics in text data.
- **Embedding Generation:** BERT generates high-dimensional word embeddings for both user queries and offer descriptions. These embeddings capture semantic meaning, allowing for accurate comparisons between the query and offers.

4. **Cosine Similarity Calculation:**
- **Similarity Score:** The tool calculates the cosine similarity between the user query embedding and the embeddings of all offers. Cosine similarity measures the cosine of the angle between two non-zero vectors and determines how similar these vectors are.
- **Ranking Offers:** Offers are ranked based on their cosine similarity scores. Higher scores indicate a stronger semantic match between the user query and the offer description.

5. **Result Presentation:**
- **Display:** The top-ranking offers, along with their similarity scores, are presented to the user. Offers are displayed in the application interface, showing the offer details and the percentage-based similarity score.
- **User Feedback:** Users can quickly assess the relevance of each offer based on the similarity score, allowing for efficient decision-making.

6. **Scalability and Performance:**
- **Efficiency:** The use of pre-trained models like BERT enhances the efficiency of the search process, allowing the tool to handle large datasets and respond quickly to user queries.
- **Scalability:** The tool is designed to scale efficiently, making it suitable for handling diverse and extensive datasets as the application's user base grows.

By integrating state-of-the-art natural language processing techniques and machine learning models, the Offer Search Tool ensures that users receive accurate, contextually relevant, and personalized search results tailored to their specific queries. The combination of BERT-based embeddings and cosine similarity calculation empowers the application to deliver a sophisticated search experience, making it a powerful tool for users seeking targeted offers from a vast and varied dataset.

