from flask import Flask, render_template
from transformers import pipeline
from collections import OrderedDict

app = Flask(__name__)

# Initialize summarizer and question generator with fine-tuned model
#summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summarizer = pipeline("summarization", model="t5-small")


# Initialize the question generation pipeline with T5 and the SentencePiece tokenizer
question_generator = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl", tokenizer="valhalla/t5-base-qg-hl")


@app.route('/')
def home():
    verse = "For God so loved the world, that he gave his only Son, that whoever believes in him should not perish but have eternal life. - John 3:16"
    summary = summarize_verse(verse)
    questions = generate_questions(verse)
    return render_template('index.html', verse=verse, summary=summary, context=context, questions=questions)

def summarize_verse(verse):
    summary = summarizer(verse, max_length=50, min_length=10, do_sample=False)[0]['summary_text']
    return summary.strip()

def generate_questions(text):
    num_questions=5
    # Add a prompt to instruct the model to generate questions
    input_text = f"generate questions: {text}"
    
    # Generate more than 3 questions to account for possible duplicates
    generated_questions = question_generator(input_text, num_return_sequences=num_questions, num_beams=10)
    
    # Extract the questions from the model output
    questions = [question['generated_text'] for question in generated_questions]
    
    # Remove duplicate questions (exact matches)
    unique_questions = list(OrderedDict.fromkeys(questions))  # Preserve order while removing duplicates
    
    # If there are fewer than 3 unique questions, generate more
    while len(unique_questions) < 2:
        additional_questions = generate_questions(text, num_questions=3)  # Generate a few more
        unique_questions.extend([q for q in additional_questions if q not in unique_questions])
        
        # Remove duplicates again just in case
        unique_questions = list(OrderedDict.fromkeys(unique_questions))

    # Return exactly 3 unique questions
    return unique_questions[:2]

if __name__ == "__main__":
    app.run(debug=True)
