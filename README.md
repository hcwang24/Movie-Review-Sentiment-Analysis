# Movie Review Sentiment Analysis Dashboard


Welcome to the **Movie Review Sentiment Analysis Dashboard**, a tool designed to analyze the sentiment of movie reviews using advanced natural language processing (NLP) techniques and machine learning models. This app provides sentiment predictions and visual insights into the factors influencing them using SHAP explanations.

## Author
**HanChen Wang**  
*November 2024*

---

## Features

- **Interactive Sentiment Analysis**: Input your own movie review text or analyze a random review.
- **SHAP Explanation**: Gain insights into the most influential words driving sentiment predictions.
- **Preprocessing Transparency**: Track changes to words during preprocessing, including slang translation and stemming.
- **Random Review Generator**: Explore the app with preloaded demo reviews.

---

## Technologies Used

- **Frontend**: Built using [Dash](https://dash.plotly.com) and styled with [Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/).
- **Backend**: 
  - **Machine Learning**: XGBoost for predictions and SHAP for model interpretability.
  - **NLP Preprocessing**: NLTK and SpaCy for text cleaning and tokenization.
- **Data**:
  - Slang dictionary for translation.
  - Demo dataset of movie reviews from IMDb.

---

## Getting Started

## Testing on the Web
You can visit the webpage deployed on render.com with this [link](https://dash.plotly.com).

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- Required Python libraries: `dash`, `dash-bootstrap-components`, `nltk`, `spacy`, `shap`, `pandas`, `joblib`, `plotly`, `matplotlib`.

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/hcwang24/sentiment_analysis.git
   cd sentiment_analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK and SpaCy data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

4. Run the app:
   ```bash
   python app.py
   ```

5. Open the app in your browser at [http://127.0.0.1:8050/](http://127.0.0.1:8050/).

---

## File Structure

```
movie-sentiment-dashboard/
├── app.py                  # Main application script
├── models/
│   ├── Vectorizer.pkl      # Pretrained vectorizer
│   ├── XGBoost_model.pkl   # Sentiment prediction model
├── data/
│   └── slang-dict.csv      # Dictionary of slang terms
├── demo/
│   └── imdb_1000.csv       # Demo movie review dataset
├── assets/                 # Additional resources (CSS, images, etc.)
├── requirements.txt        # List of dependencies
└── README.md               # Project documentation
```

---

## Usage

1. **Input a Movie Review**: Paste your text into the input box or click **Generate Random Review**.
2. **Analyze Sentiment**: Click the **Analyze Sentiment** button to get predictions.
3. **Explore Results**:
   - View the sentiment prediction (positive/negative).
   - Examine the top contributing words using SHAP visualizations.

---

## Example Output

### Sentiment Analysis Prediction
- **Review**: *"This movie was absolutely amazing! I loved every second of it."*
- **Prediction**: Positive (81% Confidence).

### SHAP Explanation
The visualization highlights the most influential words, such as:
- **Positive Contributors**: "amazing", "loved".
- **Negative Contributors**: (None in this case).

---

## Contributions

Contributions are welcome! Feel free to:
- Submit issues for bugs or feature requests.
- Fork the repository and create a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [IMDb Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [Dash Framework](https://dash.plotly.com/)
- [SHAP Library](https://github.com/slundberg/shap) 

Enjoy exploring the sentiment of movie reviews! 🚀