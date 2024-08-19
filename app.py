from flask import Flask, request, render_template, send_file
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
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

load_dotenv()

# Load environment variables for OpenAI API Key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_content():
    if 'file' not in request.files or 'content_type' not in request.form:
        return 'No file or content type selected!', 400

    file = request.files['file']
    content_type = request.form['content_type']

    if file.filename == '':
        return 'No selected file!', 400

    # Save the uploaded PDF file
    pdf_path = os.path.join('uploads', file.filename)
    file.save(pdf_path)
    print(pdf_path)

    # Generate content based on the selected type
    content = generate_content(pdf_path, content_type)

    # Generate a PDF with the content
    pdf_filename = create_pdf(content)

    return send_file(pdf_filename, as_attachment=True)

def generate_content(pdf_path, content_type):
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=50, separator="\n"
    )
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("vector_db")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    llm = ChatOpenAI()

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    retriever = FAISS.load_local("vector_db", embeddings).as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # Define the input based on the selected content type
    input_texts = {
        'questions': "Based on the content, generate quiz questions with multiple choice answers.",
        'answers': "Based on the content, generate detailed answers to key questions.",
        'lesson_plan': "Based on the content, generate a lesson plan and weekly timetable for a student to follow.",
        'summary': "Based on the content, generate a summary of the document."
    }

    input_text = input_texts.get(content_type, "Based on the content, generate a summary.")

    response = retrieval_chain.invoke({"input": input_text})

    return response["answer"]

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
