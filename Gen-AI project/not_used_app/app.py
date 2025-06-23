from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")



#youtube video info endpoint 
video_id=""

@app.route("/get_youtube_video_info", methods=["POST"])
def get_youtube_video_info():
    data = request.get_json()
    video_id = data.get("video_id")

    if not video_id:
        return jsonify({"error": "video_id is required"}), 400
    
    
    video_id = video_id.split("=")[1]
    
    video_info = {
        "video_id": video_id,
        "title": "Sample Video Title",
        "description": "This is a sample description for the video.",
        "views": 123456,
        "likes": 7890,
        "dislikes": 123
    }
    print(f"Received video_id: {video_id}")
    return jsonify(video_info)

def get_id(): 
    return video_id

if __name__ == "__main__":
    app.run(debug=True)
