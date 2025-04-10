# Insurance Customer Support Chatbot

This Colab notebook demonstrates a complete workflow for building a customer support chatbot for Niva Bupa Insurance using Retrieval Augmented Generation (RAG) techniques and fine-tuning an OpenAI model. The chatbot is designed to provide accurate and contextually relevant responses to customer inquiries by leveraging both conversational data and policy document information.

---

## Table of Contents

- [Objective](#objective)
- [Data Preparation](#data-preparation)
- [Fine-Tuning the OpenAI Model](#fine-tuning-the-openai-model)
- [Scraping and Processing Policy Documents](#scraping-and-processing-policy-documents)
- [Document Embedding and Storage](#document-embedding-and-storage)
- [RAG Chain Setup](#rag-chain-setup)
- [Gradio Chatbot UI](#gradio-chatbot-ui)
- [Usage](#usage)
- [Conclusion](#conclusion)

---

## Objective

The primary objective of this project is to build a Niva Bupa Insurance customer support chatbot that:
- Utilizes RAG (Retrieval Augmented Generation) to combine generative models with retrieval of relevant information.
- Fine-tunes an OpenAI model using real insurance customer support conversation data.
- Scrapes and processes policy documents from the Niva Bupa website.
- Integrates processed document data into a vector store for efficient retrieval.
- Implements a conversational interface using Gradio to interact with end users.

---

## Data Preparation

- **Dataset:**  
  The notebook loads customer support conversation data from the `aibabyshark/insurance_customer_support_conversation` dataset.

- **Preprocessing:**  
  - Extracts and structures both customer and agent messages into a JSONL file.
  - Splits the dataset into training and testing sets.
  - Validates the data format and estimates token counts to ensure it is ready for fine-tuning.

---

## Fine-Tuning the OpenAI Model

- **Authentication:**  
  Uses an API key (`OPENAI_API_KEY`) to authenticate with the OpenAI platform.
  
- **Uploading Data:**  
  - Uploads training and validation datasets to OpenAI.
  
- **Fine-Tuning Process:**  
  - Initiates a fine-tuning job on the `gpt-4o-2024-08-06` model.
  - Monitors the progress of the fine-tuning job.
  - Retrieves the fine-tuned model ID upon completion.
  - Downloads the resulting fine-tuning file for future use.

---

## Scraping and Processing Policy Documents

- **Website Scraping:**  
  Uses `requests_html` to scrape the Niva Bupa website for PDF links containing policy documents.
  
- **PDF Processing:**  
  - Downloads the policy document PDFs.
  - Converts the PDFs into images using PyMuPDF.
  
- **Description Generation:**  
  - Integrates the Groq API with the `llama-3.2-90b-vision-preview` model to generate detailed descriptions of the images.
  - Combines these descriptions with the original text extracted from the PDFs to create comprehensive document representations.

---

## Document Embedding and Storage

- **Document Chunking:**  
  Utilizes the `RecursiveCharacterTextSplitter` to break down documents into manageable chunks.
  
- **Embedding Creation:**  
  Generates embeddings using `OpenAIEmbeddings` for each document chunk.
  
- **Storage:**  
  Stores the document embeddings along with the chunks in a Chroma vector store for efficient retrieval during query time.

---

## RAG Chain Setup

- **Model Initialization:**  
  - Initializes a ChatOpenAI model using the fine-tuned model ID.
  
- **Retriever Configuration:**  
  - Creates a history-aware retriever that factors in past chat interactions for context-aware retrieval.
  
- **QA Chain:**  
  - Builds a question-answering chain that merges retrieved context with user queries.
  
- **RAG Integration:**  
  Combines the retriever and QA chain into a single RAG pipeline that manages multi-turn conversations by maintaining chat history.

---

## Gradio Chatbot UI

- **Interface Creation:**  
  - Develops a Gradio-based chatbot interface to interact with the RAG chain.
  
- **Session Management:**  
  Integrates session management to maintain conversational context over multiple interactions.
  
- **User Interaction:**  
  The final user interface allows customers to ask questions and receive answers drawn from both fine-tuned conversational data and policy document knowledge.

---

## Usage

1. **Prepare the Data:**
   - Download and preprocess the insurance customer support dataset.
   - Validate and format the data for fine-tuning.

2. **Fine-Tune the Model:**
   - Authenticate with OpenAI and upload your dataset.
   - Start the fine-tuning process using the specified model.
   - Monitor the fine-tuning job and retrieve the model ID upon completion.

3. **Process Policy Documents:**
   - Run the web scraping code to gather policy document PDFs.
   - Convert PDFs to images and generate descriptive text.
   - Chunk the documents and create embeddings.
   - Store the processed data in a Chroma vector store.

4. **Set Up the RAG Chain:**
   - Initialize the ChatOpenAI model with the fine-tuned model ID.
   - Configure the retriever and QA chain.
   - Combine the elements into a complete RAG pipeline.

5. **Interact Using Gradio:**
   - Launch the Gradio UI.
   - Start a conversation with the chatbot, which integrates historical context and policy document information to generate responses.

---

## Conclusion

This notebook offers a comprehensive end-to-end solution for creating an intelligent customer support chatbot for Niva Bupa Insurance. By combining fine-tuning of a conversational model with advanced document processing and retrieval techniques, the project delivers a system capable of providing detailed and contextually relevant support to customers.

Feel free to explore the code, modify parameters as needed, and extend the functionality to further tailor the chatbot to specific requirements.

---
