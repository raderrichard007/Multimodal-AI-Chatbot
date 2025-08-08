import os
import whisper
from PIL import Image
import fitz  
import streamlit as st
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import chromadb
import openai
import tempfile
from chromadb.config import Settings


load_dotenv("key.env")
openai.api_key = os.getenv("OPENAI_API_KEY")
print("API KEY:", openai.api_key)

# Load Whisper model
def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result['text']

# Analyze image with BLIP model
def analyze_image(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    return description

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    pdf_text = ""
    for page in pdf_document:
        pdf_text += page.get_text()
    return pdf_text

# Initialize ChromaDB client
client = chromadb.Client(Settings())

# Define collection name
collection_name = "pdf_collection"

# Check if the collection exists
existing_collections = [col.name for col in client.list_collections()]
if collection_name in existing_collections:
    collection = client.get_collection(name=collection_name)
else:
    collection = client.create_collection(name=collection_name)

# Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

# Upload PDF text to the vector database
def upload_pdf_to_chroma(pdf_text):
    if isinstance(pdf_text, str):
        pdf_text = [pdf_text]
    doc_ids = [str(i) for i in range(len(pdf_text))]
    collection.add(
    documents=pdf_text,
    metadatas=[{"source": "pdf"} for _ in range(len(pdf_text))],
    ids=doc_ids
    )

def transcribe_audio(audio_file):
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    # Load and transcribe the audio file
    model = whisper.load_model("base")
    result = model.transcribe(tmp_path)
    return result['text']

# Search vector DB
def search_pdf(query):
    results = collection.query(query_texts=[query], n_results=3)
    return results["documents"]

# Generate response from OpenAI
def generate_response(query):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=query,
        max_tokens=150
    )
    return response['choices'][0]['text'].strip()

# Streamlit Interface
def chatbot_interface():
    st.title("🧠 Multimodal AI Chatbot")

    # Audio upload
    audio_file = st.file_uploader("Upload Audio", type=["mp3", "wav"])
    if audio_file:
        st.info("Transcribing...")
        text_from_audio = transcribe_audio(audio_file)
        st.success("Transcribed Text:")
        st.write(text_from_audio)

    # Image upload
    image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if image_file:
        st.info("Analyzing Image...")
        image_description = analyze_image(image_file)
        st.success("Image Description:")
        st.write(image_description)

    # PDF upload
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf_file:
        st.info("Extracting text from PDF...")
        pdf_text = extract_text_from_pdf(pdf_file)
        upload_pdf_to_chroma(pdf_text)
        st.success("Extracted PDF Text (preview):")
        st.write(pdf_text[:1000] + "...")

    # Text input for chat
    user_input = st.text_input("Ask something...")
    if user_input:
        st.info("Generating response...")
        response = generate_response(user_input)
        st.success("Chatbot Response:")
        st.write(response)

if __name__ == "__main__":
    chatbot_interface()
