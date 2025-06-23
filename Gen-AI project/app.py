from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from youtube_rag import initialize_rag_system, query_video, process_video_query

app = Flask(__name__)
CORS(app)

# Global variables
current_video_id = ""
rag_chains = {}  # Store RAG chains for different videos

@app.route("/")
def index():
    return render_template("index2.html")

@app.route("/get_youtube_video_info", methods=["POST"])
def get_youtube_video_info():
    global current_video_id
    
    data = request.get_json()
    video_url = data.get("video_id")

    if not video_url:
        return jsonify({"error": "video_id is required"}), 400
    
    # Extract video ID from URL
    if "=" in video_url:
        video_id = video_url.split("=")[1]
    else:
        video_id = video_url
    print(id)
    current_video_id = video_id
    
    video_info = {
        "video_id": video_id,
        "title": "YouTube Video",
        "description": "Video loaded successfully. You can now ask questions about this video.",
        "status": "ready_for_questions"
    }
    
    return jsonify(video_info)

@app.route("/initialize_video", methods=["POST"])
def initialize_video():
    global current_video_id, rag_chains
    
    if not current_video_id:
        return jsonify({"error": "No video ID found. Please load a video first."}), 400
    
    # Check if RAG chain already exists for this video
    if current_video_id in rag_chains:
        return jsonify({"status": "success", "message": "Video already initialized and ready for questions."})
    
    # Initialize RAG system
    main_chain, status = initialize_rag_system(current_video_id)
    
    if main_chain:
        rag_chains[current_video_id] = main_chain
        return jsonify({"status": "success", "message": "Video initialized successfully. You can now ask questions."})
    else:
        return jsonify({"error": status}), 400

@app.route("/ask_question", methods=["POST"])
def ask_question():
    global current_video_id, rag_chains
    
    data = request.get_json()
    question = data.get("question")
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    if not current_video_id:
        return jsonify({"error": "No video loaded. Please load a video first."}), 400
    
    # Initialize RAG system if not already done
    if current_video_id not in rag_chains:
        main_chain, status = initialize_rag_system(current_video_id)
        if not main_chain:
            return jsonify({"error": f"Failed to initialize video: {status}"}), 400
        rag_chains[current_video_id] = main_chain
    
    # Get the RAG chain for current video
    main_chain = rag_chains[current_video_id]
    
    # Query the video
    response = query_video(question, main_chain)
    
    if response:
        return jsonify({
            "question": question,
            "answer": response,
            "video_id": current_video_id
        })
    else:
        return jsonify({"error": "Failed to process your question. Please try again."}), 400

@app.route("/quick_query", methods=["POST"])
def quick_query():
    """Single endpoint for quick question without separate initialization"""
    global current_video_id
    
    data = request.get_json()
    question = data.get("question")
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    if not current_video_id:
        return jsonify({"error": "No video loaded. Please load a video first."}), 400
    
    # Use the complete pipeline function
    result = process_video_query(current_video_id, question)
    
    if "error" in result:
        return jsonify(result), 400
    else:
        return jsonify({
            "question": question,
            "answer": result["response"],
            "video_id": current_video_id
        })

@app.route("/get_current_video", methods=["GET"])
def get_current_video():
    """Get current loaded video information"""
    global current_video_id
    
    if current_video_id:
        is_initialized = current_video_id in rag_chains
        return jsonify({
            "video_id": current_video_id,
            "initialized": is_initialized,
            "status": "ready" if is_initialized else "loaded"
        })
    else:
        return jsonify({"error": "No video loaded"}), 404

def get_id(): 
    """Helper function to get current video ID (for compatibility)"""
    return current_video_id

if __name__ == "__main__":
    app.run(debug=True)