from flask import Flask, request, render_template, jsonify
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

app = Flask(__name__)

# Load the model and tokenizer
model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

@app.route('/')
def login():
    return render_template("login.html")

@app.route("/login/", methods=["POST"])
def hello_user():
    name = request.form["username"]
    return render_template("chat.html", name=name)

@app.route("/getresponse/", methods=["POST"])
def get_response():
    data = extract_data()
    question = request.form["user-input"]

    # Tokenizing and encoding
    encoding = tokenizer.encode_plus(question, data["contexts"], padding=True, add_special_tokens=True, max_length=512, truncation="longest_first", return_tensors="pt")

    # Model inference
    outputs = model(input_ids=encoding['input_ids'],attention_mask=encoding['attention_mask'])
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Convert start and end scores to probabilities if needed
    start_probs = torch.softmax(start_scores, dim=1)
    end_probs = torch.softmax(end_scores, dim=1)

    # Find the index of the maximum probability in both start and end logits
    start_index = torch.argmax(start_probs, dim=1).item()
    end_index = torch.argmax(end_probs, dim=1).item()

    # Get the answer tokens
    answer_tokens = encoding['input_ids'][0][start_index:end_index + 1]

    # Convert the answer tokens to a string
    answer = tokenizer.decode(answer_tokens)

    # Format the output
    formatted_output = f"{question}?\n{answer}\n"

    return formatted_output

def extract_data():
    # Read the data from the text file
    with open('datafiles/year_6_ES_context.txt', 'r') as file:
        data = file.read()
    # Split the data into contexts, questions, and answers
    contexts = []
    questions = []
    answers = []    
    for line in data.split('\n'):
        if line.startswith('context'):
            contexts.append(line.split(':')[1].strip())
        elif line.startswith('question'):
            questions.append(line.split(':')[1].strip())
        elif line.startswith('answer'):
            answers.append(line.split(':')[1].strip())
             
    # Return the extracted data as a dictionary
    return {
        'contexts': contexts,
        'questions': questions,
        'answers': answers
    }

if __name__ == "__main__":
    app.run(debug=True)
