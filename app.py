import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import joblib
import zipfile
import os
import re
import pandas as pd
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import plotly.graph_objs as go
import shap  # Import SHAP
from IPython.display import display, HTML
from dash_dangerously_set_inner_html import DangerouslySetInnerHTML
from matplotlib import colors
import matplotlib as plt

from spacy import displacy

# Define preprocessing function with word tracking
def preprocess_with_tracking(text):
    word_mapping = []
    text = re.sub(r'http\S+|www\S+|\S+\.\S+/\S*', '', text) # Remove URLs (http, https, www)
    text = re.sub(r'<.*?>', '', text)                       # Remove HTML-like tags (anything between < and >)
    text = re.sub(r'@\w+|#\w+|@ \w+', '', text)             # Remove mentions and hashtags
    text = re.sub(r'(\w)\1{5,}', '', text)                  # Remove any single character repeated more than 5 times
    text = re.sub(r'[^A-Za-z\s]', ' ', text)                # Remove special characters and numbers
    text = re.sub(r'\s+', ' ', text).strip()                # Remove extra whitespace after cleaning
    
    tokens = word_tokenize(text)
    # Translate slangs and Generation Z slangs
    modified_tokens = []
    for word in tokens:
        new_word = slang_dict.get(word, word)
        if word != new_word:
            word_mapping.append((word, new_word))
        modified_tokens.append(new_word)
    
    # Convert tokens to lowercase and remove stopwords
    tokens_lower = [word.lower() for word in modified_tokens if word.lower() not in stopwords.words('english')]
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = []
    for word in tokens_lower:
        stemmed_word = stemmer.stem(word)
        if word != stemmed_word:
            word_mapping.append((word, stemmed_word))
        stemmed_tokens.append(stemmed_word)
        
    word_mapping = list(set(word_mapping))
    
    return ' '.join(stemmed_tokens), word_mapping

# Function to unzip and load models
def load_zipped_model(zip_file_path, model_filename):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall('models')  # Extract files to the 'models' folder
    model = joblib.load(os.path.join('models', model_filename))
    return model

# Load the vectorizer and voting classifier from the zipped files
vectorizer = load_zipped_model('models/Vectorizer.zip', 'vectorizer.pkl')
voting_classifier = load_zipped_model('models/voting_classifier_compressed.zip', 'voting_classifier_model.pkl')

# Load XGBoost model for SHAP explanations
xgboost = load_zipped_model('models/XGBoost.zip', 'XGBoost_model.pkl')

# Load the slang dictionary
slang_dict = pd.read_csv("data/slang-dict.csv", index_col = 'keyword').to_dict()

# Load demo text samples
def load_demo_text(file_path='demo/imdb_1000.csv'):
    with open(file_path, 'r') as file:
        return file.readlines()

# Randomly select a review
def get_random_review(demo_texts):
    return random.choice(demo_texts).strip()

# Load demo text samples
demo_texts = load_demo_text()

# Dash app layout
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = dbc.Container([
    html.H1("Movie Review Sentiment Analysis", style={'textAlign': 'center', 'padding': '10px'}),
    
    # Input Section
    dbc.Row([
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    # Textarea Input Section
                    dbc.Col([
                        dcc.Textarea(
                            id='input-text',
                            placeholder='Enter movie review text or generate a random review...',
                            value=get_random_review(demo_texts),  # Load a random sample text initially
                            style={'width': '100%', 'height': '100px'}
                        ),
                    ]),
                    dbc.Col([
                        html.Div([
                            html.Button('Analyze Sentiment', id='submit-val', n_clicks=0, style={
                                'background-color': '#008CBA', 'color': 'white', 'padding': '10px 24px', 'font-size': '16px',
                                'margin-right': '10px'
                            }),
                            html.Button('Generate Random Review', id='generate-review', n_clicks=0, style={
                                'background-color': '#f44336', 'color': 'white', 'padding': '10px 24px', 'font-size': '16px'
                            })
                        ], style={'margin-top': '5px'})
                    ]),
                ])
            ])
        ], style={'margin-top': '10px'})
    ]),

    dbc.Row([
        dbc.Card([
            dbc.CardBody([
                dbc.Col([
                    html.Label('Select the number of top SHAP words to display:'),
                    dcc.Input(id='n-shap-input', type='number', value=20, min=1, style={'margin-top': '40px', 'margin-left': '10px', 'margin-right': '10px'})
                ], width=6),
                dbc.Row([
                    dcc.Graph(id='output-sentiment-graph', style={"display": "inline-block", "width": "30%"}),
                    dcc.Graph(id='shap-plot', style={"display": "inline-block", "width": "60%"}),
                ])
            ])
        ])
    ]),
        
    # Output Section: SpaCy Visualization
    dbc.Row([
        dbc.Card([
            dbc.CardBody([
                    # Visualization Section
                    dbc.Col([
                        html.H4("Text Analysis Visualization", style={'textAlign': 'center', 'margin-bottom': '10px'}),
                        dcc.Markdown(
                            id='spacy-output', 
                            dangerously_allow_html=True,
                            style={'width': '100%', 'height': '200px', 'border': '1px solid #ccc', 'padding': '10px'}
                        ),
                    ], width=5),
            ])
        ], style={'width': '80%', 'margin': '10px auto'})
    ], justify='center'),
    
    # Footer
    html.Footer("Developed by HanChen Wang, October 2024", style={'textAlign': 'center', 'padding': '10px'})
], fluid=True, style={
    'background-image': 'url("/assets/work.jpg")',  # Adjusted to reference assets folder
    'background-size': 'cover',
    'background-repeat': 'no-repeat',
    'background-attachment': 'fixed',
    'background-position': 'center',
    'min-height': '100vh',
})

# Callback to generate a random review
@app.callback(
    Output('input-text', 'value'),
    [Input('generate-review', 'n_clicks')]
)
def update_review(n_clicks):
    if n_clicks > 0:
        return get_random_review(demo_texts)
    return ''

# Function to generate SHAP values and plot using Plotly
def generate_shap_plotly(text_vector, model, n=20):
    # Create a SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(text_vector)
    shap_values_dict = {}
    shap_values_dict['XGBoost'] = shap_values[0]
    shap_values_df = pd.DataFrame(shap_values_dict, index=vectorizer.get_feature_names_out())
    
    top_n_shap_values_df = shap_values_df.loc[shap_values_df['XGBoost'].abs().nlargest(n).index]
    top_n_shap_values_df = top_n_shap_values_df.sort_values(by="XGBoost")

    fig = go.Figure()

    # Add horizontal bars for the top 20 words based on XGBoost SHAP values
    fig.add_trace(go.Bar(
        x=top_n_shap_values_df['XGBoost'],  # SHAP values on x-axis
        y=top_n_shap_values_df.index,       # Words on y-axis
        orientation='h',                     # Horizontal bar
        marker=dict(color='orange'),
        name='XGBoost SHAP Values',
        hovertemplate=(
            '<b>%{y}</b><br>' +  # Hover label for word
            'XGBoost SHAP: %{x:.4f}<br>'
        )
    ))

    # Update layout with title and axis labels
    fig.update_layout(
        title=f"Top {n} Words with Highest SHAP Values (XGBoost)",
        xaxis_title="XGBoost SHAP Value",  # SHAP values on the x-axis
        yaxis_title="Word",                # Words on the y-axis
        template="plotly_white",
        hoverlabel=dict(bgcolor=None),
        showlegend=False
    )

    return fig, shap_values_df['XGBoost']

def tag_original_text_with_shap(original_text, shap_values, word_mapping, top_n=20):
    """
    Tags the original text based on SHAP values using SpaCy visualizer.
    
    Arguments:
    original_text -- the original text string
    shap_values -- SHAP values corresponding to the words in the processed text
    word_mapping -- a list of tuples (original_word, processed_word) that maps original words to processed words
    top_n -- the number of top words based on SHAP values to tag
    
    Returns:
    HTML visualization with SpaCy displaCy
    """
    # Tokenize the original text
    original_words = word_tokenize(original_text)

    # Create a dictionary for processed words to SHAP values
    processed_to_shap = {processed: shap_values.get(processed, 0.0) for _, processed in word_mapping}

    # Reverse mapping to match original to processed words
    reverse_mapping = {original: processed for original, processed in word_mapping}

    # Get the top N processed words by absolute SHAP value
    top_processed_words = sorted(processed_to_shap.keys(), key=lambda word: abs(processed_to_shap[word]), reverse=True)[:top_n]

    # Separate positive and negative SHAP values
    positive_shap_values = [processed_to_shap[processed_word] for processed_word in top_processed_words if processed_to_shap[processed_word] > 0.01]
    negative_shap_values = [processed_to_shap[processed_word] for processed_word in top_processed_words if processed_to_shap[processed_word] < -0.01]

    # Normalize SHAP values for color scaling
    pos_norm = colors.Normalize(vmin=0, vmax=1)
    neg_norm = colors.Normalize(vmin=-1, vmax=0)

    # Generate colors for positive and negative values
    positive_colors = {f"POS ({shap:.2f})": colors.to_hex(plt.cm.Reds(pos_norm(shap))) for shap in positive_shap_values}
    negative_colors = {f"NEG ({shap:.2f})": colors.to_hex(plt.cm.Blues(1-abs(neg_norm(shap)))) for shap in negative_shap_values}

    # Combine colors into options
    color_options = {**positive_colors, **negative_colors}

    # Prepare entities for SpaCy visualization
    entities = []
    for word in original_words:
        processed_word = reverse_mapping.get(word, None)
        if processed_word in top_processed_words:
            shap_value = processed_to_shap[processed_word]
            if abs(shap_value) >= 0.01:
                label = f"{'POS' if shap_value > 0 else 'NEG'} ({shap_value:.2f})"
                start = original_text.find(word)
                end = start + len(word)
                entities.append({
                    "start": start,
                    "end": end,
                    "label": label,
                })

    # Create a SpaCy-compatible dictionary for rendering
    spacy_data = {
        "text": original_text,
        "ents": entities,
        "title": "SHAP Highlighting"
    }
    
    options = {"colors": color_options}

    # Use SpaCy's displacy.render to generate the HTML
    return displacy.render(spacy_data, style="ent", manual=True, options=options)

# Callback function to process text, predict sentiment, and display SHAP explanations
@app.callback(
    [Output('output-sentiment-graph', 'figure'), Output('shap-plot', 'figure'), Output('spacy-output', 'children')],
    [Input('submit-val', 'n_clicks'), Input('n-shap-input', 'value')],
    [Input('input-text', 'value')]
)
def predict_sentiment(n_clicks, top_n, input_text):
    if n_clicks > 0 and input_text:
        n_clicks = 0
        # Preprocess input text
        processed_text, word_mapping = preprocess_with_tracking(input_text)
        
        # Convert text to vector using TF-IDF vectorizer
        text_vector = vectorizer.transform([processed_text])
        
        # Predict sentiment using the VotingClassifier
        proba = voting_classifier.predict_proba(text_vector)
        
        positive_confidence = proba[0][1]
        negative_confidence = proba[0][0]
        
        # Create a Plotly bar plot for sentiment analysis result
        sentiment_fig = go.Figure(data=[
            go.Bar(name='Sentiment', x=['Negative', 'Positive'], y=[negative_confidence, positive_confidence],
                   marker_color=['#008CBA', '#f44336'])
        ])
        
        sentiment_fig.update_layout(
            title='Sentiment Analysis Result',
            xaxis_title='Sentiment',
            yaxis_title='Confidence Level',
            yaxis_range=[0, 1],
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'size': 16},
        )

        # Generate SHAP values plot using Plotly
        shap_fig, shap_values = generate_shap_plotly(text_vector, xgboost, n=top_n)
        
        # Tag words indicating the SHAP values
        tagged_html = tag_original_text_with_shap(input_text, shap_values, word_mapping, top_n)

        # Return both sentiment plot and SHAP plot
        return sentiment_fig, shap_fig, tagged_html

    return {}, {}, "No analysis available."

if __name__ == '__main__':
    app.run_server(debug=True)
