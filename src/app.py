from flask import Flask, render_template, request, jsonify
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import random

app = Flask(__name__)

# Load BERT model for semantic understanding
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

class DynamicAI:
    def __init__(self):
        self.conversation_memory = []
        self.knowledge_base = {}
        self.context_window = 5

    def get_embedding(self, text):
        tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**tokens)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def generate_response(self, user_input):
        # Add to conversation memory
        self.conversation_memory.append({"role": "user", "content": user_input})
        
        # Get context from recent conversations
        recent_context = self.conversation_memory[-self.context_window:]
        
        # Generate embedding for user input
        input_embedding = self.get_embedding(user_input)
        
        # Generate response based on context and semantic understanding
        response = self.create_dynamic_response(user_input, input_embedding, recent_context)
        
        self.conversation_memory.append({"role": "assistant", "content": response})
        return response

    def create_dynamic_response(self, user_input, input_embedding, context):
        # Analyze sentiment and intent
        sentiment_score = self.analyze_sentiment(user_input)
        intent = self.detect_intent(user_input)
        
        # Generate response components
        context_info = self.extract_context_info(context)
        relevant_knowledge = self.retrieve_relevant_knowledge(input_embedding)
        
        # Construct response
        response_components = []
        
        if intent == "question":
            response_components.append(self.generate_answer(user_input, relevant_knowledge))
        elif intent == "statement":
            response_components.append(self.generate_insight(user_input, context_info))
        elif intent == "greeting":
            response_components.append(self.generate_greeting(sentiment_score))
        
        # Add follow-up or elaboration
        if random.random() < 0.7:  # 70% chance to add follow-up
            response_components.append(self.generate_follow_up(context_info))
        
        return " ".join(response_components)

    def analyze_sentiment(self, text):
        # Simple sentiment analysis
        positive_words = set(['good', 'great', 'awesome', 'excellent', 'happy', 'love'])
        negative_words = set(['bad', 'terrible', 'awful', 'sad', 'hate', 'wrong'])
        
        words = text.lower().split()
        sentiment_score = sum(1 for word in words if word in positive_words)
        sentiment_score -= sum(1 for word in words if word in negative_words)
        
        return sentiment_score

    def detect_intent(self, text):
        question_words = set(['what', 'why', 'how', 'when', 'where', 'who'])
        text_lower = text.lower()
        
        if any(word in text_lower.split() for word in question_words):
            return "question"
        elif any(greeting in text_lower for greeting in ['hello', 'hi', 'hey']):
            return "greeting"
        else:
            return "statement"

    def extract_context_info(self, context):
        # Extract key information from recent conversation context
        key_points = []
        for message in context:
            # Extract entities, topics, and themes
            content = message['content'].lower()
            words = content.split()
            key_points.extend([word for word in words if len(word) > 4])  # Simple keyword extraction
        
        return list(set(key_points))  # Remove duplicates

    def retrieve_relevant_knowledge(self, query_embedding):
        # Simulate knowledge retrieval
        # In a real implementation, this would search through a knowledge base
        return ["I understand this topic", "Let me share my perspective", "Based on my analysis"]

    def generate_answer(self, question, knowledge):
        # Generate a specific answer based on the question and knowledge
        return f"Based on my analysis, {random.choice(knowledge)}. Would you like to know more?"

    def generate_insight(self, statement, context):
        # Generate an insight or observation
        return f"I notice that {random.choice(context)} is important here. Let me elaborate..."

    def generate_greeting(self, sentiment):
        time_greetings = ["Hope you're having a great day!", "It's wonderful to interact with you!"]
        return random.choice(time_greetings)

    def generate_follow_up(self, context):
        follow_ups = [
            "What are your thoughts on this?",
            "Would you like to explore this further?",
            "How does this relate to your experience?",
            f"I'm curious about your perspective on {random.choice(context) if context else 'this topic'}."
        ]
        return random.choice(follow_ups)

ai = DynamicAI()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    ai_response = ai.generate_response(user_message)
    return jsonify({"response": ai_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
