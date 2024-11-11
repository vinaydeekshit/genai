from flask import Flask, render_template, request, redirect, url_for, jsonify
from transformers import pipeline
from nltk import sent_tokenize
import os
from docx import Document
import pdfplumber
from langchain_community.llms import DeepInfra
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from flask_caching import Cache
import logging 

app = Flask(__name__)


os.environ["DEEPINFRA_API_TOKEN"] = 'fCUq30zmzPgZJMKx2Z8kUB7HB2cgC374'

# Initialize the DeepInfra model
llm = DeepInfra(model_id="meta-llama/Meta-Llama-3-70B-Instruct")
llm.model_kwargs = {
    "temperature": 0.7,
    "repetition_penalty": 1.2,
    "max_new_tokens": 1000, 
    "top_p": 0.9,
}


template = """Conversation:
{conversation}

Current Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["conversation", "question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text if text.strip() else "No text found in the DOCX file."

# Function to extract text from .pdf
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text if text.strip() else "No text found in the PDF file."

# Function to extract text from .txt
def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text if text.strip() else "No text found in the TXT file."

# General function to extract text based on file extension
def extract_text(file_path):
    _, ext = os.path.splitext(file_path)
    if ext.lower() == '.docx':
        return extract_text_from_docx(file_path)
    elif ext.lower() == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext.lower() == '.txt':
        return extract_text_from_txt(file_path)
    else:
        return f"Unsupported file format: {ext}"

# Function to send extracted text to DeepInfra for summarization

def summarize_text_with_deepinfra(text):
    if not text.strip():
        return "No text to summarize."
    
    os.environ["DEEPINFRA_API_TOKEN"] = 'fCUq30zmzPgZJMKx2Z8kUB7HB2cgC374'
    
    # Initialize DeepInfra model
    llm = DeepInfra(model_id="meta-llama/Meta-Llama-3-70B-Instruct")
    llm.model_kwargs = {
        "temperature": 0.7,
        "repetition_penalty": 1.2,
        "max_new_tokens": 1000,
        "top_p": 0.9,
    }

    # Prompt template for summarization
    prompt_template = PromptTemplate(input_variables=["text"], template="Summarize the following document: {text}")
    chain = LLMChain(llm=llm, prompt=prompt_template)
    summary = chain.run({"text": text})
    
    return summary


# Route for handling file upload and summarization
# Route for handling file upload and summarization
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file
    file_path = os.path.join(os.getcwd(), file.filename)
    file.save(file_path)

    # Extract text from the uploaded file
    extracted_text = extract_text(file_path)

    # Summarize the extracted text
    summary = summarize_text_with_deepinfra(extracted_text)

    # Optionally, you can remove the saved file after processing
    os.remove(file_path)

    return jsonify({'success': 'File uploaded successfully', 'summary': summary})



@app.route('/generate_sql_query', methods=['POST'])
def generate_sql_query():
    data = request.json
    query_text = data.get('text', '')
    if not query_text:
        return jsonify({'error': 'No query text provided'}), 400
    
    # Generate SQL query using DeepInfra
    sql_prompt = f"Generate an MSSQL or MYSQL or Postgresee or oracle or mongodb  query for the following text: {query_text}"
    sql_query = llm_chain.run(conversation='', question=sql_prompt)
    
    return jsonify({'query': sql_query})

@app.route('/extract_meeting_info', methods=['POST'])
def extract_meeting_info():
    transcript = request.form.get('transcript')
    logging.info(f"Received transcript: {transcript}")  # Log received transcript
    if not transcript:
        return jsonify({'error': 'No transcript provided'}), 400

    prompt = f"""
    Extract the summary, tasks, and minutes of meeting from the following transcript. give seperate paragraphs for summary,tasks and minutes of meeting:

    {transcript}
    """
    logging.info(f"Generated prompt for LLM: {prompt}")  # Log the generated prompt
    
    # Generate response using DeepInfra
    try:
        response = llm.generate([prompt])
        logging.info(f"LLM Response: {response}")  # Log the response for debugging

        if response.generations and response.generations[0]:
            return jsonify({'response': response.generations[0][0].text.strip()})
    except Exception as e:
        logging.error(f"Error during LLM generation: {str(e)}")  # Log any exceptions that occur

    return jsonify({'error': 'No response generated.'}), 500


logging.basicConfig(level=logging.DEBUG)

@app.route('/email_summarize', methods=['POST'])
def generate_summary():
    data = request.json
    text = data.get('text', '')  # Extract the text from the JSON payload

    # Log the incoming request data for debugging
    print(f"Received text: {text}")

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        # Generate the summary using the model or service
        summary_prompt = f"Summarize the following text: {text}"
        summary = llm_chain.run(conversation='', question=summary_prompt)
        
        # Check if summary is valid or fallback
        if not summary:
            summary = "Unable to generate a meaningful summary. Please check the content and try again."

        return jsonify({'summary': summary})

    except Exception as e:
        return jsonify({'error': f'Error generating summary: {str(e)}'}), 500





@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/dashboard')  # Ensure the route is correct
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)
