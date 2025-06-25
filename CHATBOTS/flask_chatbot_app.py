from flask import Flask, request, jsonify
from transformers import pipeline, set_seed

app = Flask(__name__)

# Load the chatbot model
chatbot_model = pipeline("text-generation", model="microsoft/DialoGPT-medium")
set_seed(42)

def generate_response(user_input):
    response = chatbot_model(user_input, max_length=100, num_return_sequences=1)
    return response[0]['generated_text']

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'error': 'Please provide a message'}), 400
    
    response = generate_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
