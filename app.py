from flask import Flask, request, render_template
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# load the data
df = pd.read_csv('data.csv')

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
def get_similarity_score(query, offer):
    # Tokenize and encode the query and offer
    tokens = tokenizer([query, offer], padding=True, truncation=True, return_tensors='pt')
    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
    
    # Get the embeddings for each sentence
    query_embedding = embeddings[0].mean(dim=1).squeeze().detach().numpy()
    offer_embedding = embeddings[1].mean(dim=1).squeeze().detach().numpy()
    
    # Reshape the embeddings to 2D arrays
    query_embedding = query_embedding.reshape(1, -1)
    offer_embedding = offer_embedding.reshape(1, -1)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(query_embedding, offer_embedding)[0][0]
    return similarity

def search_offers(query, df):
    # Filter the DataFrame based on multiple columns and query
    mask = df.apply(lambda row: any(query.lower() in str(row[col]).lower() for col in ['BRAND', 'PRODUCT_CATEGORY', 'IS_CHILD_CATEGORY_TO', 'RETAILER']), axis=1)
    filtered_df = df[mask]
    
    # Calculate similarity score for each offer and add it to the DataFrame
    filtered_df['SIMILARITY_SCORE'] = filtered_df['OFFER'].apply(lambda x: get_similarity_score(query, x))
    
    # Sort by similarity score in descending order
    filtered_df = filtered_df.sort_values(by='SIMILARITY_SCORE', ascending=False)
    
    # Return the filtered DataFrame
    return filtered_df


@app.route('/', methods=['GET', 'POST'])
def index():
    results = df.copy()
    results['SIMILARITY_SCORE']=0
    if request.method == 'POST':
        query = request.form['query']
        results = search_offers(query, df)
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)