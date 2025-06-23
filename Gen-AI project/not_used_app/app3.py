
from youtube_rag import initialize_rag_system, query_video, process_video_query
id="ZXiruGOCn9s"  # Example YouTube video ID
question = "What is the main topic of the video?"
result = process_video_query(id, question)
print(result)