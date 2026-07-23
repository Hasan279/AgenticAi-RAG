import warnings
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA

def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings('ignore')

embedding_model = HuggingFaceEmbeddings()

def get_llm(temp=0.5, tokens=256):
    return OllamaLLM(
        model="llama3.2",
        temperature=temp if temp else 0.5,
        num_predict=tokens if tokens else 256
    )

def document_loader(file):
    loader = PyPDFLoader(file.name)
    loaded_document = loader.load()
    return loaded_document

def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len
    )
    chunks = splitter.split_documents(data)
    return chunks

def vector_database(chunks):
    vector_db = Chroma.from_documents(chunks, embedding_model)
    return vector_db

def retriever(file):
    if file is None:
        return None
    doc = document_loader(file)
    chunks = text_splitter(doc)
    vectordb = vector_database(chunks)
    return vectordb.as_retriever()

def retriever_qa(file, query, retriever_obj):
    if retriever_obj is None:
        return "Please upload a PDF document first."
    
    llm = get_llm()
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever_obj, 
        return_source_documents=False
    )
    response = qa.invoke(query)
    return response['result']

with gr.Blocks(title="RAG Chatbot") as rag_application:
    gr.Markdown("# RAG Chatbot")
    gr.Markdown("Upload a PDF document and ask any question. The chatbot will try to answer using the provided document.")
    
    retriever_state = gr.State()
    
    file_input = gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath")
    query_input = gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    output_text = gr.Textbox(label="Output")
    
    submit_btn = gr.Button("Submit")
    
    file_input.change(
        fn=retriever,
        inputs=[file_input],
        outputs=[retriever_state]
    )
    
    submit_btn.click(
        fn=retriever_qa,
        inputs=[file_input, query_input, retriever_state],
        outputs=[output_text]
    )

rag_application.launch()