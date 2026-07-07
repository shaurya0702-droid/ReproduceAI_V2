# This file contains only infrastructure

import os
import uuid
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

GROQ_API_KEY= os.getenv("GROQ_API_KEY")
EMBEDDING_MODEL= "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL= "llama-3.1-8b-instant"
CHUNK_SIZE= 500
CHUNK_OVERLAP= 50
TOP_K= 4


def load_pdf(path: str): # load a pdf
    loader= PyMuPDFLoader(path)
    docs= loader.load()
    return docs

def split_documents(docs): # Split documents into overlapping chunks for retrieval.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    return chunks

def get_embedding_model(): #Create and return the embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    return embeddings

def build_vectorstore(chunks, embeddings): # Create the Chroma vector database
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"paper_{uuid.uuid4().hex}",
    )
    return vectorstore

def get_retriever(vectorstore, k=TOP_K): # Return a retriever object.
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )
    return retriever

def get_llm(temperature=0): # Create the Groq LLM.
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=temperature
    )
    return llm

def format_docs(docs):# Convert retrieved documents into plain text
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

def get_sources(docs): # Return page numbers used for retrieval
    pages = [
        doc.metadata["page"] + 1
        for doc in docs
    ]
    return sorted(set(pages))