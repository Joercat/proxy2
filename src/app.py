from flask import Flask, render_template, request, jsonify
import random
import nltk
import json
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from datetime import datetime

app = Flask(__name__)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load responses from JSON file
def load_responses():
    try:
        with open('responses.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "greetings": {
                "patterns": ["hello", "hi", "hey", "good morning", "good evening"],
                "responses": ["Hi there!", "Hello!", "Hey!", "Greetings!", "Welcome!"],
                "context": "greeting"
            },
            "farewell": {
                "patterns": ["bye", "goodbye", "see you", "later"],
                "responses": ["Goodbye!", "See you later!", "Bye!", "Take care!", "Until next time!"],
                "context": "farewell"
            },
            "gratitude": {
                "patterns": ["thank you", "thanks", "appreciate"],
                "responses": ["You're welcome!", "Glad I could help!", "My pleasure!", "Anytime!"],
                "context": "gratitude"
            },
            "mood": {
                "patterns": ["how are you", "how do you feel", "what's up"],
                "responses": ["I'm doing great!", "I'm learning new things!", "Excited to chat!", "Ready to help!"],
                "context": "mood"
            },
            "capabilities": {
                "patterns": ["what can you do", "help me", "your abilities"],
                "responses": ["I can chat, learn, and help with various topics!", "I'm an AI assistant ready to help!", "Let's explore what we can do together!"],
                "context": "capabilities"
            }
        }

responses_data = load_responses()

# Learning system
class LearningSystem:
    def __init__(self):
        self.conversation_history = []
        self.learned_patterns = {}
        self.load_learned_patterns()

    def load_learned_patterns(self):
        try:
            with open('learned_patterns.json', 'r') as f:
                self.learned_patterns = json.load(f)
        except FileNotFoundError:
            self.learned_patterns = {}

    def save_learned_patterns(self):
        with open('learned_patterns.json', 'w') as f:
            json.dump(self.learned_patterns, f)

    def learn_from_interaction(self, user_input, response, feedback=None):
        timestamp = datetime.now().isoformat()
        interaction = {
            'user_input': user_input,
            'response': response,
            'timestamp': timestamp,
            'feedback': feedback
        }
        self.conversation_history.append(interaction)
        
        # Learn new patterns
        processed_input = preprocess_text(user_input)
        if processed_input not in self.learned_patterns:
            self.learned_patterns[processed_input] = {
                'responses': [response],
                'usage_count': 1,
                'success_rate': 1.0
            }
        else:
            self.learned_patterns[processed_input]['usage_count'] += 1
            if response not in self.learned_patterns[processed_input]['responses']:
                self.learned_patterns[processed_input]['responses'].append(response)
        
        self.save_learned_patterns()

learning_system = LearningSystem()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(token) for token in tokens 
             if token not in string.punctuation and token not in stop_words]
    return " ".join(tokens)

def get_response(message):
    processed_message = preprocess_text(message)
    
    # Check learned patterns first
    if processed_message in learning_system.learned_patterns:
        learned_data = learning_system.learned_patterns[processed_message]
        response = random.choice(learned_data['responses'])
        return response
    
    # Check predefined patterns
    for category, data in responses_data.items():
        for pattern in data['patterns']:
            if pattern in processed_message:
                response = random.choice(data['responses'])
                learning_system.learn_from_interaction(message, response)
                return response
    
    # Generate a response based on context analysis
    context_response = generate_context_response(processed_message)
    learning_system.learn_from_interaction(message, context_response)
    return context_response

def generate_context_response(processed_message):
    # Simple context-based response generation
    words = processed_message.split()
    
    if any(word in ['what', 'who', 'where', 'when', 'why', 'how'] for word in words):
        return "That's an interesting question. Let me learn more about it."
    
    if any(word in ['can', 'could', 'would', 'will'] for word in words):
        return "I'm analyzing your request to provide better assistance."
    
    if any(word in ['feel', 'think', 'believe', 'opinion'] for word in words):
        return "I'm processing different perspectives on this topic."
    
    return "I'm learning about this topic to provide better responses in the future."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    ai_response = get_response(user_message)
    return jsonify({"response": ai_response})

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    learning_system.learn_from_interaction(
        data['user_message'],
        data['ai_response'],
        data['feedback']
    )
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
