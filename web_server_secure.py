#!/usr/bin/env python3
"""Web deployment server for OpenClaw chatbot (security-hardened)."""
import json
import os
import logging
import threading
from datetime import date
from functools import wraps
from flask import Flask, request, jsonify
from collections import defaultdict
from time import time

# Security constants
MAX_REQUESTS_PER_MINUTE = 30  # Per IP
MAX_MEMORY_SIZE = 1000
MAX_PROMPT_LENGTH = 2000
MAX_MEMORY_ITEM_LENGTH = 500
DANGEROUS_PATTERNS = [
    "```", "`", "$(", "${", " import ", " exec", " eval",
    "os.", "subprocess.", "system(", " bash", " sh ", "python",
    "open(", "file(", "input(", "compile(", "__import__("
]

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

WORKSPACE = os.path.dirname(os.path.realpath(__file__))
MEMORY_DIR = os.path.join(WORKSPACE, "memory")
MEMORY_FILE = os.path.join(MEMORY_DIR, f"{date.today().isoformat()}.json")

# Get port from environment (for deployment platforms like Render)
port = int(os.environ.get('PORT', 5000))

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Global state with thread safety
state = {"memory": [], "llm_mode": "transformers", "llm_status": "not_loaded"}
state_lock = threading.RLock()
llm_pipeline = None

# Rate limiting: IP -> [timestamps]
request_times = defaultdict(list)
rate_limit_lock = threading.Lock()


def load_llm():
    global llm_pipeline
    try:
        from transformers import pipeline
        llm_pipeline = pipeline("text-generation", model="gpt2", max_new_tokens=150)
        state["llm_mode"] = "transformers"
        state["llm_status"] = "ready"
        logger.info("LLM loaded: transformers")
    except Exception:
        state["llm_status"] = "failed: check logs"
        logger.error("LLM init failed", exc_info=False)
        llm_pipeline = None


def rate_limit(f):
    """Rate limit by IP address."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        ip = request.remote_addr
        now = time()
        
        with rate_limit_lock:
            # Remove old timestamps (>1 min)
            request_times[ip] = [t for t in request_times[ip] if now - t < 60]
            
            if len(request_times[ip]) >= MAX_REQUESTS_PER_MINUTE:
                logger.warning(f"Rate limit exceeded for IP {ip}")
                return jsonify({"error": "rate limit exceeded"}), 429
            
            request_times[ip].append(now)
        
        return f(*args, **kwargs)
    return decorated_function


def is_safe_prompt(prompt: str) -> bool:
    """Check for dangerous patterns."""
    prompt_lower = " " + prompt.lower() + " "
    for danger in DANGEROUS_PATTERNS:
        if danger in prompt_lower:
            logger.warning(f"Unsafe prompt detected: '{danger}'")
            return False
    return True


def llm_respond(prompt: str) -> str:
    """Generate response with safety checks."""
    if llm_pipeline is None:
        return "LLM not loaded."
    
    if len(prompt) > MAX_PROMPT_LENGTH:
        logger.warning(f"Prompt too long: {len(prompt)} chars")
        return f"Prompt too long (max {MAX_PROMPT_LENGTH})."
    
    if not is_safe_prompt(prompt):
        logger.warning("Unsafe prompt rejected")
        return "Prompt contains unsafe patterns."
    
    try:
        out = llm_pipeline(prompt, do_sample=True, temperature=0.7)
        return out[0]["generated_text"]
    except Exception:
        logger.error("LLM inference failed")
        return "LLM error (internal)"


@app.route("/status", methods=["GET"])
@rate_limit
def status():
    """Get agent status."""
    with state_lock:
        return jsonify({
            "mode": "OpenClaw web API (hardened)",
            "memory_items": len(state["memory"]),
            "llm_mode": state.get("llm_mode"),
            "llm_status": state.get("llm_status"),
            "max_memory": MAX_MEMORY_SIZE,
        })


@app.route("/chat", methods=["POST"])
@rate_limit
def chat():
    """Chat endpoint with security validation."""
    data = request.get_json() or {}
    
    # Validate input
    if not isinstance(data, dict):
        logger.warning(f"Invalid request body type: {type(data)}")
        return jsonify({"error": "invalid request"}), 400
    
    user_message = data.get("message", "").strip()
    
    if not user_message:
        return jsonify({"error": "empty message"}), 400
    
    if len(user_message) > MAX_PROMPT_LENGTH:
        logger.warning(f"Message too long: {len(user_message)} chars")
        return jsonify({"error": "message too long"}), 400
    
    # Get response
    response = llm_respond(user_message)
    
    # Store in memory (with size limits)
    with state_lock:
        if len(state["memory"]) >= MAX_MEMORY_SIZE:
            logger.info("Memory full, removing oldest")
            state["memory"].pop(0)
        
        state["memory"].append({"user": user_message, "bot": response, "timestamp": date.today().isoformat()})
    
    logger.info(f"Chat request: {len(user_message)} chars")
    return jsonify({"response": response})


@app.route("/memory", methods=["GET"])
@rate_limit
def get_memory():
    """Get all memories with size limit."""
    with state_lock:
        # Don't expose full history to avoid DoS via large responses
        total = len(state["memory"])
        recent = state["memory"][-10:] if state["memory"] else []  # Last 10 only
    
    logger.info(f"Memory retrieval: {total} total, returning {len(recent)} recent")
    return jsonify({"total": total, "recent": recent})


@app.route("/memory", methods=["POST"])
@rate_limit
def add_memory():
    """Add a memory item with validation."""
    data = request.get_json() or {}
    item = data.get("item", "").strip()
    
    if not item:
        return jsonify({"error": "empty item"}), 400
    
    if len(item) > MAX_MEMORY_ITEM_LENGTH:
        return jsonify({"error": f"item too long (max {MAX_MEMORY_ITEM_LENGTH})"}), 400
    
    with state_lock:
        if len(state["memory"]) >= MAX_MEMORY_SIZE:
            logger.info("Memory full, removing oldest")
            state["memory"].pop(0)
        state["memory"].append(item)
    
    logger.info(f"Memory added: {len(item)} chars")
    return jsonify({"ok": True})


@app.route("/memory/<int:idx>", methods=["DELETE"])
@rate_limit
def delete_memory(idx):
    """Delete memory by index."""
    with state_lock:
        if 0 <= idx < len(state["memory"]):
            removed = state["memory"].pop(idx)
            logger.info(f"Memory deleted at index {idx}")
            return jsonify({"ok": True, "deleted": str(removed)[:50]})
    
    return jsonify({"error": "index out of range"}), 400


@app.errorhandler(404)
def not_found(e):
    logger.warning(f"404: {request.path}")
    return jsonify({"error": "not found"}), 404


@app.errorhandler(500)
def server_error(e):
    logger.error(f"500: {request.path}")
    return jsonify({"error": "internal server error"}), 500


@app.before_request
def before_request():
    """Log all requests."""
    logger.info(f"{request.method} {request.path} from {request.remote_addr}")


if __name__ == "__main__":
    load_llm()
    logger.warning("Starting development server. For production use gunicorn/uWSGI + reverse proxy (nginx)")
    # IMPORTANT: debug=False, hosts should be configured for deployment
    app.run(debug=False, host="0.0.0.0", port=port)
