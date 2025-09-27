#!/usr/bin/env python3
"""
RAG Telegram Bot (Irish Law) — FAISS→Chroma auto-fallback

Quick start:
  # set your bot token
  export TELEGRAM_BOT_TOKEN="YOUR_TOKEN_HERE"

  # run
  python rag_chatbot.py

If FAISS isn't installed, the script will fall back to Chroma (requires: pip install chromadb).
"""

import os
import logging
from typing import List, Optional

# Optional: load .env if present
try:
    from dotenv import load_dotenv  # pip install python-dotenv (optional)
    load_dotenv()
except Exception:
    pass

from bs4 import BeautifulSoup

# LangChain core & community
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Embeddings & LLM backends
from langchain_ollama import OllamaEmbeddings, OllamaLLM
# optional fallback (unused by default, but kept wired)
from langchain_huggingface import HuggingFaceEmbeddings

# RAG chain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Telegram bot (v20+ / v21)
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

from httpx import RemoteProtocolError, ReadError
from asyncio import CancelledError, TimeoutError as AsyncTimeoutError

# ------------------------------
# Vector store backend (FAISS preferred, Chroma fallback)
# ------------------------------
# You may force a backend via env: export VECTOR_BACKEND=faiss|chroma
VECTOR_BACKEND = os.environ.get("VECTOR_BACKEND", "faiss").lower()
VectorStoreClass = None
Chroma = None  # set if available
FAISS = None   # set if available

try:
    from langchain_community.vectorstores import FAISS as _FAISS
    FAISS = _FAISS
except Exception:
    if VECTOR_BACKEND == "faiss":
        # If the user explicitly asked for FAISS but it's missing, downgrade preference
        VECTOR_BACKEND = "chroma"

try:
    from langchain_community.vectorstores import Chroma as _Chroma
    Chroma = _Chroma
except Exception:
    # Will error later only if needed
    pass

# ------------------------------
# Config
# ------------------------------
DOCS_DIR = os.environ.get("DOCS_DIR", "docs")
FAISS_INDEX_DIR = os.environ.get("FAISS_INDEX_DIR", "faiss_index")
CHROMA_DIR = os.environ.get("CHROMA_DIR", "chroma_index")

# Embeddings + LLM
USE_OLLAMA_EMBEDDINGS = True  # keep one embedding backend consistently
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_EMBED_MODEL = os.environ.get(
    "OLLAMA_EMBED_MODEL", "nomic-embed-text")  # or "mxbai-embed-large"
LLM_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "llama3")

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("irish-law-rag-bot")

# ------------------------------
# Document loading
# ------------------------------


def load_documents(directory: str = DOCS_DIR) -> List[Document]:
    """Recursively load PDFs and HTML files from a directory."""
    documents: List[Document] = []

    if not os.path.isdir(directory):
        logger.warning("Docs directory %s does not exist.", directory)
        return documents

    for root, _, files in os.walk(directory):
        for filename in files:
            path = os.path.join(root, filename)
            try:
                fl = filename.lower()
                if fl.endswith(".pdf"):
                    loader = PyPDFLoader(path)
                    docs = loader.load()
                    for d in docs:
                        # ensure metadata presence + source path
                        meta = dict(d.metadata or {})
                        meta["source"] = path
                        d.metadata = meta
                    documents.extend(docs)

                elif fl.endswith(".html") or fl.endswith(".htm"):
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        soup = BeautifulSoup(f, "lxml")
                        for el in soup(["script", "style", "nav", "header", "footer"]):
                            el.decompose()
                        text = soup.get_text(separator=" ", strip=True)
                        if text:
                            documents.append(
                                Document(page_content=text,
                                         metadata={"source": path})
                            )
            except Exception as e:
                logger.error("Error loading %s: %s", path, e)

    logger.info("Loaded %d documents from %s", len(documents), directory)
    return documents

# ------------------------------
# Splitter
# ------------------------------


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    logger.info("Split into %d chunks.", len(chunks))
    return chunks

# ------------------------------
# Embeddings
# ------------------------------


def get_embeddings():
    if USE_OLLAMA_EMBEDDINGS:
        logger.info("Using OllamaEmbeddings model=%s", OLLAMA_EMBED_MODEL)
        return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
    else:
        logger.info("Using HuggingFaceEmbeddings model=%s", HF_EMBEDDING_MODEL)
        return HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)

# ------------------------------
# Vector store
# ------------------------------


def build_or_load_vector_store(texts: List[Document]):
    """
    Build or load a vector store (FAISS preferred, else Chroma).
    """
    embeddings = get_embeddings()

    # Prefer FAISS if available
    if VECTOR_BACKEND == "faiss" and FAISS is not None:
        logger.info("Vector backend: FAISS")
        # allow_dangerous_deserialization is required by LangChain for FAISS load
        if os.path.isdir(FAISS_INDEX_DIR):
            try:
                vs = FAISS.load_local(
                    FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True
                )
                logger.info("Loaded FAISS index from %s", FAISS_INDEX_DIR)
                return vs
            except Exception as e:
                logger.warning(
                    "Failed to load FAISS index, will rebuild: %s", e)

        logger.info("Creating FAISS index at %s ...", FAISS_INDEX_DIR)
        vs = FAISS.from_documents(texts, embeddings)
        vs.save_local(FAISS_INDEX_DIR)
        logger.info("Saved FAISS index.")
        return vs

    # Fallback to Chroma
    if Chroma is None:
        raise SystemExit(
            "No vector store available. Install one of:\n"
            "  conda install -c conda-forge faiss-cpu=1.8.0  (recommended)\n"
            "  or: pip install faiss-cpu==1.8.0.post3\n"
            "  or: pip install chromadb  (for Chroma fallback)\n"
        )

    logger.info("Vector backend: Chroma")
    if os.path.isdir(CHROMA_DIR):
        vs = Chroma(persist_directory=CHROMA_DIR,
                    embedding_function=embeddings)
        logger.info("Opened Chroma index at %s", CHROMA_DIR)
        return vs

    logger.info("Creating Chroma index at %s ...", CHROMA_DIR)
    vs = Chroma.from_documents(texts, embeddings, persist_directory=CHROMA_DIR)
    vs.persist()
    logger.info("Saved Chroma index.")
    return vs

# ------------------------------
# RAG chain
# ------------------------------


def setup_rag_chain(vector_store) -> RetrievalQA:
    """
    Build a simple RetrievalQA chain with a custom prompt.
    """
    llm = OllamaLLM(model=LLM_MODEL)

    # RetrievalQA(stuff) expects {context} + {question} in the prompt
    prompt_template = (
        "You are a legal assistant specializing in Irish law.\n"
        "Use the following context to answer the user's question accurately and concisely.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )
    logger.info("RAG chain ready (LLM=%s).", LLM_MODEL)
    return qa_chain

# ------------------------------
# Telegram handlers
# ------------------------------


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hello! I'm a RAG chatbot for Irish legal documents.\n"
        "Ask about the Irish Constitution, statutes, or other legal texts.\n"
        "Use /help for more info."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Ask questions about Irish legal documents (e.g., 'What does Article 40 cover?').\n"
        "I use Retrieval-Augmented Generation to provide answers from your PDF and HTML sources."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = (update.message.text or "").strip()
    if not query:
        await update.message.reply_text("Please send a text question.")
        return

    try:
        logger.info("User asked: %r", query)
        result = context.bot_data["qa_chain"].invoke({"query": query})
        answer = result.get("result") if isinstance(
            result, dict) else str(result)
        if not answer:
            await update.message.reply_text("Sorry, I couldn't produce an answer.")
            return
        await update.message.reply_text(answer)

    except (RemoteProtocolError, ReadError, AsyncTimeoutError, CancelledError):
        # Typical when stopping mid-stream or Ollama drops the HTTP stream.
        logger.warning(
            "Generation interrupted or stream closed; sending friendly notice.")
        try:
            await update.message.reply_text("I was interrupted while generating the answer. Please try again.")
        except Exception:
            pass  # don't loop on errors sending error text

    except Exception as e:
        logger.exception("Error handling message:")
        try:
            await update.message.reply_text(f"Error: {str(e)}")
        except Exception:
            pass
# ------------------------------
# Main
# ------------------------------


def main():
    logger.info("Loading documents from %s ...", DOCS_DIR)
    docs = load_documents(DOCS_DIR)
    if not docs:
        logger.error(
            "No documents found in '%s' (or all failed to load). Exiting.", DOCS_DIR)
        return

    logger.info("Splitting documents ...")
    texts = split_documents(docs)

    logger.info("Building/Loading vector store ...")
    vector_store = build_or_load_vector_store(texts)

    logger.info("Setting up RAG chain ...")
    qa_chain = setup_rag_chain(vector_store)

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        logger.error("TELEGRAM_BOT_TOKEN is not set. Export it and re-run.")
        return

    app = Application.builder().token(bot_token).build()
    app.bot_data["qa_chain"] = qa_chain

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Starting Telegram bot polling ... (backend=%s)",
                VECTOR_BACKEND.upper())
    app.run_polling()


if __name__ == "__main__":
    main()
