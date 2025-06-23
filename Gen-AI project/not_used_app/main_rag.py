import os 
from langchain_openai import ChatOpenAI 
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain_together import TogetherEmbeddings
from langchain_core.runnables import RunnableParallel,RunnableLambda,RunnablePassthrough
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from not_used_app.app import get_id

# four steps for RAG
# 1. Get the transcript of the video 
# indexing 
# 

os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="llama3-70b-8192",   # "llama3-70b-8192"
    temperature=0.7,
    max_tokens=2000,  # Adjust as needed
)
# model develpment 


video_id = get_id()  # Assuming get_id() returns the video ID from the Flask app
if not video_id:
    raise ValueError("Video ID is not provided. Please set the video ID before running the script.")



try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    print(transcript)

except TranscriptsDisabled:
    print("Captions are disabled for this video.")
except NoTranscriptFound:
    print("No transcript found in the requested language.")
except Exception as e:
    print(f"Error: {e}")


#spliting and creation of documents

spliter = RecursiveCharacterTextSplitter(chunk_size=1000 ,chunk_overlap=200)
chunks = spliter.create_documents([transcript])  

os.environ["OPENAI_API_KEY"] = "2f2c9e25075a8768b707adcff56865283eb487cb962e920fbcc6f432cbb001f7"
os.environ["OPENAI_API_BASE"] = "https://api.together.xyz/v1"

embeddings = TogetherEmbeddings(
    model="BAAI/bge-base-en-v1.5"
)

# vector store creation 
vector_store = FAISS.from_documents(chunks,embeddings) 


prompt = PromptTemplate(
    template= """ 
        you are helpful assistant. 
        Answer ONLY from the provided transcript context. 
        if the context is insufficient just say you dont't know . 
        {context} 
        Question:{question}
    """, 
    input_variables=['context','question']
)
retriver= vector_store.as_retriever(search_type="similarity" , search_kwargs={'k':4})  # k is number of doument 

# question
question          = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs    = retriver.invoke(question)
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({"context": context_text, "question": question})


def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text


parallel_chain = RunnableParallel({
    'context': retriver | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})


parser = StrOutputParser()


main_chain = parallel_chain| prompt | llm | parser

# final output
# main_chain.invoke({"question": question, "context": context_text})
responce=main_chain.invoke("can you summarizer the video")
