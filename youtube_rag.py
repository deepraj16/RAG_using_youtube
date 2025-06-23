import os 
from langchain_openai import ChatOpenAI 
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_together import TogetherEmbeddings
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def setup_llm():
    """Setup and configure the language model"""
    os.environ["OPENAI_API_KEY"] = "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"
    os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
    
    llm = ChatOpenAI(
        model="llama3-70b-8192",   
        temperature=0.7,
    )
    return llm

def get_video_transcript(video_id):
    """Get transcript from YouTube video"""
    if not video_id:
        raise ValueError("Video ID is not provided.")
    
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        return transcript
    
    except TranscriptsDisabled:
        return None
    except NoTranscriptFound:
        return None
    except Exception as e:
        return None

def create_document_chunks(transcript):
    """Split transcript into smaller chunks"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])  
    return chunks

def setup_embeddings_and_vectorstore(chunks):
    """Setup embeddings and create vector store"""
    os.environ["OPENAI_API_KEY"] = "2f2c9e25075a8768b707adcff56865283eb487cb962e920fbcc6f432cbb001f7"
    os.environ["OPENAI_API_BASE"] = "https://api.together.xyz/v1"
    
    embeddings = TogetherEmbeddings(
        model="BAAI/bge-base-en-v1.5"
    )
    
    vector_store = FAISS.from_documents(chunks, embeddings) 
    return vector_store

def create_prompt_template():
    """Create the prompt template for the RAG system"""
    prompt = PromptTemplate(
        template=""" 
            You are a helpful assistant. 
            Answer ONLY from the provided transcript context. 
            If the context is insufficient, just say you don't know. 
            {context} 
            Question: {question}
        """, 
        input_variables=['context', 'question']
    )
    return prompt

def format_docs(retrieved_docs):
    """Format retrieved documents into context text"""
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

def setup_rag_chain(vector_store, prompt, llm):
    """Setup the complete RAG chain"""
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 4})
    
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })
    
    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser
    
    return main_chain

def initialize_rag_system(video_id):
    """Initialize the complete RAG system for a video"""
    try:
        # Setup LLM
        llm = setup_llm()
        
        # Get transcript
        transcript = get_video_transcript(video_id)
        if not transcript:
            return None, "Failed to get transcript. Video may not have captions or captions may be disabled."
        
        # Create chunks
        chunks = create_document_chunks(transcript)
      # print(chunks[0])
        # Setup vector store
        vector_store = setup_embeddings_and_vectorstore(chunks)
        
        # Create prompt
        prompt = create_prompt_template()
        
        # Setup RAG chain
        main_chain = setup_rag_chain(vector_store, prompt, llm)
        
        return main_chain, "Success"
    
    except Exception as e:
        return None, f"Error initializing RAG system: {str(e)}"

def query_video(question, main_chain):
    """Query the video using the RAG chain"""
    try:
        response = main_chain.invoke(question)
        return response
    except Exception as e:
        return None

def process_video_query(video_id, question):
    """Process a single query for a video - complete pipeline"""
    main_chain, status = initialize_rag_system(video_id)
    
    if main_chain is None:
        return {"error": status}
    
    response = query_video(question, main_chain)
    
    if response:
        return {"response": response}
    else:
        return {"error": "Failed to process query"}


