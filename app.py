from flask import Flask, request, render_template, send_file, jsonify
from dotenv import load_dotenv
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.faiss import FAISS
from fpdf import FPDF
import os
import openai
from flask_cors import CORS

app = Flask(__name__, static_folder='assets')
CORS(app)

load_dotenv()

# Load environment variables for OpenAI API Key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/upload', methods=['POST'])
def upload_content():
    if 'file' not in request.files or 'content_type' not in request.form:
        return jsonify({'success': False, 'message': 'No file or content type selected!'}), 400

    file = request.files['file']
    content_type = request.form['content_type']
    custom_prompt = request.form.get('custom_prompt')
    custom_prompt_result = request.form.get('custom_prompt_result')

    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file!'}), 400

    # Save the uploaded PDF file
    pdf_path = os.path.join('uploads', file.filename)
    file.save(pdf_path)

    # Process and store content from the PDF
    try:
        process_pdf_content(pdf_path)
        return jsonify({'success': True, 'message': 'PDF uploaded successfully!'}), 200
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return jsonify({'success': False, 'message': 'Failed to process PDF.'}), 500

@app.route('/chatbot', methods=['POST'])
def chatbot_query():
    data = request.get_json()
    user_message = data.get('message')

    if not user_message:
        return jsonify({'response': 'Please enter a message.'}), 400

    try:
        # Generate a response using the language model
        response = generate_chatbot_response(user_message)
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error generating response: {e}")
        return jsonify({
            'response': (
                "It seems there's a network or connection issue. Please check your internet connection "
                "and try again. In the meantime, here's a tip: If you're having trouble connecting, "
                "try restarting your router or checking your network settings."
            )
        })

def process_pdf_content(pdf_path):
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=50, separator="\n"
    )
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("vector_db")

def generate_content(content_type, custom_prompt, custom_prompt_result):
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    llm = ChatOpenAI()

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    retriever = FAISS.load_local("vector_db", OpenAIEmbeddings()).as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    if custom_prompt_result:
        input_text = custom_prompt_result
    elif custom_prompt:
        input_text = custom_prompt
    else:
       input_texts = {
           'questions': (
               "Based on the content, generate 10 multiple choice questions with 4 answer options each, "
               "10 true or false questions, 5 definition questions, and 5 short notes questions."
           ),
           'answers': "Based on the content, generate detailed answers to key questions.",
           'lesson_plan': "Based on the content, generate a lesson plan and weekly timetable for a student to follow.",
           'summary': "Based on the content, generate a summary of the document."
       }
       input_text = input_texts.get(content_type, "Based on the content, generate a summary.")

    response = retrieval_chain.invoke({"input": input_text})
    return response["answer"]

def generate_chatbot_response(user_message):
    try:
        # Attempt to retrieve information from the processed PDF content
        retriever = FAISS.load_local("vector_db", OpenAIEmbeddings()).as_retriever()
        retrieval_chain = create_retrieval_chain(
            retriever,
            create_stuff_documents_chain(ChatOpenAI(), hub.pull("langchain-ai/retrieval-qa-chat"))
        )
        response = retrieval_chain.invoke({"input": user_message})["answer"]

        # If a response is generated, return it
        if response:
            return response
    except Exception as e:
        print("Error with vector-based retrieval:", e)

    # Fallback to OpenAI API for general knowledge responses if no PDF content or error occurs
    try:
        fallback_response = openai.Completion.create(
            engine="text-davinci-003",  # or any other GPT-3.5 or GPT-4 model
            prompt=user_message,
            max_tokens=150
        )
        return fallback_response.choices[0].text.strip()
    except Exception as e:
        print("Error with OpenAI API:", e)
        return (
            "I'm sorry, but I'm currently unable to fetch information. Please check your internet "
            "connection or try again later."
        )

def create_pdf(content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, content)
    pdf_filename = "generated_content.pdf"
    pdf.output(pdf_filename)
    return pdf_filename

if __name__ == "__main__":
    app.run(debug=True, port=7000)
