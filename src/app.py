from flask import Flask, render_template, request, jsonify
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

app = Flask(__name__)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Simple response database
responses = {
    "hello": ["Hi there!", "Hello!", "Hey!", "Greetings!"],
    "how are you": ["I'm doing well, thanks!", "I'm great!", "All good!"],
    "bye": ["Goodbye!", "See you later!", "Bye!", "Take care!"],
    "default": ["I'm not sure about that.", "Could you rephrase that?", "Interesting..."]
}

def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text.lower())
    # Remove punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(token) for token in tokens 
             if token not in string.punctuation and token not in stop_words]
    return " ".join(tokens)

def get_response(message):
    processed_message = preprocess_text(message)
    
    # Check for matching patterns
    for key in responses:
        if key in processed_message:
            return random.choice(responses[key])
    
    return random.choice(responses["default"])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    ai_response = get_response(user_message)
    return jsonify({"response": ai_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

