import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, session, jsonify, stream_with_context, Response
from groq import Groq

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecret")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def get_chat_history():
    return session.get('chat_history', [])

def save_message(role, content):
    history = get_chat_history()
    history.append({"role": role, "content": content})
    session['chat_history'] = history

@app.route("/", methods=["GET"])
def home():
    return render_template("chat.html", chat_history=get_chat_history())

@app.route("/send", methods=["POST"])
def send():
    user_msg = request.json.get("message", "")
    save_message("user", user_msg)
    return jsonify({"status": "ok"})

@app.route("/stream", methods=["POST"])
def stream():
    # Add user message to history
    user_msg = request.json.get("message", "")
    save_message("user", user_msg)
    history = get_chat_history()

    # Groq expects message list, e.g. [{"role": "user", "content": ...}, {"role": "assistant", ...}]
    def generate():
        completion = client.chat.completions.create(
            model=MODEL,
            messages=history,
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        for chunk in completion:
            if hasattr(chunk, "choices"):
                for choice in chunk.choices:
                    if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                        content = choice.delta.content
                        if content is not None:
                            yield content
    # Stream as text/event-stream for SSE (Server Sent Events)
    return Response(stream_with_context(generate()), mimetype="text/event-stream")

@app.route("/reset", methods=["POST"])
def reset():
    session["chat_history"] = []
    return jsonify({"status": "reset"})

if __name__ == "__main__":
    app.run(debug=True)
