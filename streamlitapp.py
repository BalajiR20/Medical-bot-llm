import streamlit as st
import pickle
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load the LLM chain

def load_llm_chain():
    with open("llm_chain.pkl", "rb") as f:
        llm_chain = pickle.load(f)
    return llm_chain
def load_llm_summarizer():
    with open("llm_summarizer.pkl", "rb") as f:
        llm_summarizer = pickle.load(f)
    return llm_summarizer

# Load the vector store if applicable

def load_vectorstore():
    return FAISS.load_local("faiss_final","huggingface_embeddings",allow_dangerous_deserialization=True)

# Streamlit App
def main():
    st.title("LLM Chain in Streamlit")

    # Load resources
    llm_chain = load_llm_chain()
    llm_summarizer = load_llm_summarizer()

    # Optional: Load vector store and use it if the chain requires a retriever
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Input box for user query
    query = st.text_input("Enter your query:")

    if st.button("Run Query"):
        if query:
            # Run the chain
            response = llm_chain.invoke(query)
            output = []
            for i, doc in enumerate(response['source_documents']):
                output.append(f"**Source {i + 1}:**\n{doc.page_content}\n")

            # Join the output into a single string
            Instructions: '''Summarize the text below by:
            1. Highlighting the most important points.
            2. Using bullet points for clarity.
            3. Keeping the summary under 50 words.'''
            output_str = "\n".join(output)
            summary = llm_summarizer(output_str, max_length=400,min_length=200, do_sample=False)
            st.write(summary[0]['summary_text'])
            
            
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
