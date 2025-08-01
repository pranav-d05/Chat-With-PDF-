from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI



def main():
    load_dotenv()    
    st.set_page_config(page_title='Ask Your PDF')
    st.header("Ask Your PDF")
    
    pdf = st.file_uploader("Upload Your PDF ",type="pdf")    

    if pdf is not None :
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        #Split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        chunks = text_splitter.split_text(text)
        
        embd = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        knowledge_base = FAISS.from_texts(chunks,embd)
        
        
        user_ques = st.text_input("Ask a question about your PDF:")
        if user_ques:
            docs = knowledge_base.similarity_search(user_ques)
            llm = ChatGoogleGenerativeAI(
                     model="gemini-2.5-flash",
                     google_api_key="YOUR_API_KEY"
                    )
            chain = load_qa_chain(llm,chain_type ="stuff")
            response = chain.run(input_documents=docs,question=user_ques)

            st.write(response)
                     
if __name__ == '__main__':
    main()
