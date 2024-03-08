import tensorflow as tf
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

# Read the data from the text file
with open('/content/drive/MyDrive/bert_docs/year_6_ES_context.txt', 'r') as file:
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

# Print the extracted data
print("Contexts:")
for context in contexts:
    print(context)

print("\nQuestions:")
for question in questions:
    print(question)

print("\nAnswers:")
for answer in answers:
    print(answer)

# Model and tokenizer initialization
model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
question1= input("enter question: ")
# Tokenizing and encoding
encoding = tokenizer.encode_plus(question1, context, padding=True, add_special_tokens=True, max_length=512, truncation="longest_first", return_tensors="pt")

# Model inference
outputs = model(**encoding)
start_scores = outputs.start_logits
end_scores = outputs.end_logits

# Convert start and end scores to probabilities if needed
start_probs = torch.softmax(start_scores, dim=1)
end_probs = torch.softmax(end_scores, dim=1)

# Find the index of the maximum probability in both start and end logits
start_index = torch.argmax(start_probs, dim=1).item()
end_index = torch.argmax(end_probs, dim=1).item()

# Get the answer tokens
answer_tokens = tokenizer.convert_ids_to_tokens(encoding.input_ids[0][start_index:end_index + 1])

# Join the answer tokens into a single string
answer = tokenizer.convert_tokens_to_string(answer_tokens)

# Format the output
formatted_output = f"{question1}?\n{answer}\n"

print(formatted_output)
