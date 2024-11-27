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

def compute_shap_values(model, text_vector, vectorizer, word_mapping, top_n=20):
    """
    Computes SHAP values for the given model and text vector, filters and identifies the top N words.

    Parameters:
        model: The trained model for which SHAP values are computed.
        text_vector: The vectorized representation of the input text.
        vectorizer: The vectorizer used to create the text vector.
        word_mapping: A list of tuples mapping original to processed words.
        top_n: The number of top SHAP values to retain.

    Returns:
        pd.DataFrame: A DataFrame of the top N words with their SHAP values.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(text_vector)
    shap_values_dict = {"SHAP": shap_values[0]}
    shap_values_df = pd.DataFrame(shap_values_dict, index=vectorizer.get_feature_names_out())

    processed_words = {processed for _, processed in word_mapping}
    filtered_shap_values_df = shap_values_df.loc[shap_values_df.index.isin(processed_words)]
    filtered_shap_values_df = filtered_shap_values_df.loc[filtered_shap_values_df['SHAP'].abs() >= 0.01]
    top_n_shap_values_df = filtered_shap_values_df.loc[filtered_shap_values_df['SHAP'].abs().nlargest(top_n).index]
    top_n = top_n_shap_values_df.shape[0]
    return shap_values_df, top_n_shap_values_df.sort_values(by="SHAP"), top_n

def generate_spacy_visualization(input_text, shap_values_df, top_n_shap_values_df, word_mapping):
    """
    Generates a SpaCy visualization for SHAP values highlighting.

    Parameters:
        input_text: The original input text.
        shap_values_df: The DataFrame containing SHAP values.
        top_n_shap_values_df: The DataFrame of top N SHAP values.
        word_mapping: A list of tuples mapping original to processed words.

    Returns:
        None: Renders the SpaCy visualization in the notebook.
    """
    reverse_mapping = {original: processed for original, processed in word_mapping}

    # Tokenize and calculate word positions
    original_words = word_tokenize(input_text)
    word_positions = []
    current_position = 0
    for word in original_words:
        start_idx = input_text.find(word, current_position)
        end_idx = start_idx + len(word)
        word_positions.append((word, start_idx, end_idx))
        current_position = end_idx

    # Separate positive and negative SHAP values
    positive_shap_values = [shap_values_df['SHAP'][word] for word in top_n_shap_values_df.index if shap_values_df['SHAP'][word] > 0.01]
    negative_shap_values = [shap_values_df['SHAP'][word] for word in top_n_shap_values_df.index if shap_values_df['SHAP'][word] < -0.01]

    pos_norm = colors.Normalize(vmin=0, vmax=1)
    neg_norm = colors.Normalize(vmin=-1, vmax=0)
    positive_colors = {f"POS ({shap:.2f})": colors.to_hex(plt.cm.Reds(pos_norm(shap))) for shap in positive_shap_values}
    negative_colors = {f"NEG ({shap:.2f})": colors.to_hex(plt.cm.Blues(1 - abs(neg_norm(shap)))) for shap in negative_shap_values}
    color_options = {**positive_colors, **negative_colors}

    # Prepare entities for visualization
    entities = []
    for word, start, end in word_positions:
        processed_word = reverse_mapping.get(word, None)
        if processed_word in shap_values_df.index:
            shap_value = shap_values_df['SHAP'][processed_word]
            if abs(shap_value) >= 0.01:
                label = f"{'POS' if shap_value > 0 else 'NEG'} ({shap_value:.2f})"
                entities.append({"start": start, "end": end, "label": label})
    
    # Modify the input text to add underscores between adjacent entities
    modified_text = input_text
    offset = 0  # Tracks the shift in indices due to added underscores
    for i in range(len(entities) - 1):
        current_end = entities[i]["end"] + offset
        next_start = entities[i + 1]["start"] + offset
        # If entities are adjacent, insert an underscore
        if current_end + 1 == next_start:
            modified_text = modified_text[:current_end] + "_" + modified_text[current_end:]
            offset += 1  # Increment offset due to the added character
    
    # Update entity indices to match the modified text
    adjusted_entities = []
    for entity in entities:
        start = entity["start"] + modified_text[:entity["start"]].count("_")
        end = entity["end"] + modified_text[:entity["end"]].count("_")
        adjusted_entities.append({"start": start, "end": end, "label": entity["label"]})
    
    # Create SpaCy-compatible data structure
    spacy_data = {"text": modified_text, "ents": adjusted_entities, "title": "SHAP Highlighting"}
    options = {"colors": color_options}
    return displacy.render(spacy_data, style="ent", manual=True, options=options), color_options

def plot_shap_bar_chart(top_n_shap_values_df, top_n, color_options):
    """
    Plots a horizontal bar chart of the top N words based on SHAP values, using specified colors for the bars.

    Parameters:
        top_n_shap_values_df: The DataFrame of top N SHAP values with words as the index.
        top_n: The number of top SHAP values being visualized.
        color_options: A dictionary mapping SHAP value labels to their respective colors.

    Returns:
        None: Displays the bar chart.
    """
    # Assign colors to bars based on SHAP value labels
    bar_colors = [
        color_options[f"{'POS' if shap > 0 else 'NEG'} ({shap:.2f})"]
        for shap in top_n_shap_values_df['SHAP']
    ]

    # Create the horizontal bar chart
    shap_fig = go.Figure()
    shap_fig.add_trace(go.Bar(
        x=top_n_shap_values_df['SHAP'],  # SHAP values on x-axis
        y=top_n_shap_values_df.index,       # Words on y-axis
        orientation='h',                    # Horizontal bars
        marker=dict(color=bar_colors),      # Bar colors
        name='XGBoost SHAP Values',
        hovertemplate=(
            '<b>%{y}</b><br>' +  # Hover label for word
            'XGBoost SHAP: %{x:.2f}<br>'
        )
    ))

    # Update layout with title and axis labels
    shap_fig.update_layout(
        title=f"Top {top_n} Words with Highest SHAP Values (XGBoost)",
        xaxis_title="XGBoost SHAP Value",  # SHAP values on the x-axis
        yaxis_title="Word",                # Words on the y-axis
        template="plotly_white",
        hoverlabel=dict(bgcolor=None),
        showlegend=False
    )
    return shap_fig

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
        
        shap_values_df, top_n_shap_values_df, top_n = compute_shap_values(xgboost, text_vector, vectorizer, word_mapping, top_n=top_n)
        
        # Tag words indicating the SHAP values
        tagged_html, color_options = generate_spacy_visualization(input_text, shap_values_df, top_n_shap_values_df, word_mapping)
        
        shap_fig = plot_shap_bar_chart(top_n_shap_values_df, top_n, color_options)

        # Return both sentiment plot and SHAP plot
        return sentiment_fig, shap_fig, tagged_html

    return {}, {}, "No analysis available."

if __name__ == '__main__':
    app.run_server(debug=True)
