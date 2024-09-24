from flask import Flask, render_template, request, redirect, url_for, jsonify
from transformers import pipeline
from nltk import sent_tokenize
import os
import fitz  # PyMuPDF library for PDF handling
from langchain_community.llms import DeepInfra
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from flask_caching import Cache

app = Flask(__name__)

# Cache configuration
app.secret_key = 'your_secret_key'
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# DeepInfra API setup
os.environ["DEEPINFRA_API_TOKEN"] = 'fCUq30zmzPgZJMKx2Z8kUB7HB2cgC374'
llm = DeepInfra(model_id="meta-llama/Meta-Llama-3-70B-Instruct")
llm.model_kwargs = {
    "temperature": 0.7,
    "repetition_penalty": 1.2,
    "max_new_tokens": 100,
    "top_p": 0.9,
}

template = """Conversation:
{conversation}

Current Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["conversation", "question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Summarization pipeline
summarizer = pipeline("summarization")

# Function to summarize text
def summarize_text(text):
    try:
        summarized = summarizer(text, max_length=150, min_length=30, do_sample=False)
        return summarized[0]['summary_text']
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "Error summarizing text."

# Function to generate Minutes of Meeting
def generate_mom(text):
    sentences = sent_tokenize(text)
    mom = "Minutes of Meeting:\n\n"
    for i, sentence in enumerate(sentences):
        mom += f"{i+1}. {sentence}\n"
    return mom

# Function to generate tasks
def generate_tasks(text):
    sentences = sent_tokenize(text)
    tasks = "Actionable Tasks:\n\n"
    for sentence in sentences:
        if "action" in sentence.lower() or "task" in sentence.lower():
            tasks += f"- {sentence}\n"
    return tasks

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_text = ""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype='pdf')
        for page in doc:
            pdf_text += page.get_text()
        return pdf_text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

# Endpoint to summarize document
@app.route('/summarize_document', methods=['POST'])
def summarize_document():
    file = request.files.get('document')
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    file_extension = os.path.splitext(file.filename)[1].lower()
    
    # Read and decode file content based on its type
    if file_extension in ['.txt', '.md']:
        try:
            text = file.read().decode('utf-8')
        except Exception as e:
            print(f"Error reading text file: {e}")
            return jsonify({'error': 'Error reading text file'}), 400
    elif file_extension == '.pdf':
        text = extract_text_from_pdf(file)
        if not text:
            return jsonify({'error': 'Error extracting text from PDF'}), 400
    else:
        return jsonify({'error': 'Unsupported file type'}), 400
    
    # Summarize the document text
    summary = summarize_text(text)
    if summary == "Error summarizing text.":
        return jsonify({'error': 'Error summarizing text'}), 500
    
    return jsonify({'summary': summary})

# Endpoint to generate MoM, Summary, and Tasks
@app.route('/generate_mom_summary_tasks', methods=['POST'])
def generate_mom_summary_tasks():
    text = request.form.get('meeting_text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    summary = summarize_text(text)
    mom = generate_mom(text)
    tasks = generate_tasks(text)
    
    return jsonify({'summary': summary, 'mom': mom, 'tasks': tasks})

# Endpoint to generate SQL query
@app.route('/generate_sql_query', methods=['POST'])
def generate_sql_query():
    data = request.json
    query_text = data.get('text', '')
    if not query_text:
        return jsonify({'error': 'No query text provided'}), 400
    
    # Generate SQL query using DeepInfra
    sql_prompt = f"Generate an SQL query for the following text: {query_text}"
    sql_query = llm_chain.run(conversation='', question=sql_prompt)
    
    return jsonify({'query': sql_query})

# Endpoint to generate email summary
@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    email_text = request.form.get('email_text')
    
    if not email_text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Summarize the email text
    summary = summarize_text(email_text)
    
    return jsonify({'summary': summary})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)
