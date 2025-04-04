from flask import Flask, Response, request, jsonify
import json
import threading
import queue
import time
from dotenv import load_dotenv
from road_attribution_correction import RoadAttributionCorrection
from road_segment_corrector import RoadSegmentCorrector
from config import Config

# Initialize Flask app
app = Flask(__name__)

# Dictionary to store progress queues for each task
progress_queues = {}

def process_tile(task_id, tile):
    """Function that processes a tile and reports progress."""
    q = progress_queues[task_id]
    
    # Load environment variables
    load_dotenv()
    
    # Report initial status
    q.put(json.dumps({
        "status": "started",
        "tile": tile,
        "message": f"Processing tile: {tile}"
    }))
    
    time.sleep(1)  # Simulate initial loading
    
    # Report that we're starting the road attribution correction
    q.put(json.dumps({
        "status": "processing",
        "tile": tile,
        "current_step": "RoadAttributionCorrection",
        "message": f"Starting RoadAttributionCorrection for tile: {tile}"
    }))
    
    # Process the road attribution correction
    road_attr_result = RoadAttributionCorrection(tile).process()
    
    # Report completion of road attribution correction with results
    q.put(json.dumps({
        "status": "processing",
        "tile": tile,
        "current_step": "RoadAttributionCorrection completed",
        "message": f"Completed RoadAttributionCorrection for tile: {tile}",
        "result": road_attr_result
    }))
    
    # Report that we're starting the road segment corrector
    q.put(json.dumps({
        "status": "processing",
        "tile": tile,
        "current_step": "RoadSegmentCorrector",
        "message": f"Starting RoadSegmentCorrector for tile: {tile}"
    }))
    
    # Process the road segment corrector
    road_segment_result = RoadSegmentCorrector(tile).process()
    
    # Report completion of road segment corrector with results
    q.put(json.dumps({
        "status": "completed",
        "tile": tile,
        "current_step": "RoadSegmentCorrector completed",
        "message": f"Completed RoadSegmentCorrector for tile: {tile}",
        "road_attr_result": road_attr_result,
        "road_segment_result": road_segment_result
    }))
    
    # Signal completion
    q.put(None)

@app.route('/start-task', methods=['POST'])
def start_task():
    """Start a new tile processing task and return its ID."""
    data = request.get_json() or {}
    tile = data.get('tile', '23599610')  # Default tile if none provided
    
    # Create a unique task ID
    task_id = f"task_{int(time.time())}"
    
    # Create a queue for this task
    progress_queues[task_id] = queue.Queue()
    
    # Start the processing in a separate thread
    thread = threading.Thread(target=process_tile, args=(task_id, tile))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "task_id": task_id,
        "status": "started",
        "tile": tile
    })

@app.route('/task-progress/<task_id>', methods=['GET'])
def task_progress(task_id):
    """Stream progress updates for a specific task."""
    if task_id not in progress_queues:
        return jsonify({"error": "Task not found"}), 404
    
    def generate_progress_updates():
        q = progress_queues[task_id]
        
        while True:
            progress_data = q.get()
            
            # None signals end of updates
            if progress_data is None:
                # Clean up the queue when done
                del progress_queues[task_id]
                break
            
            yield f"data: {progress_data}\n\n"
    
    return Response(generate_progress_updates(), 
                   mimetype='text/event-stream',
                   headers={'Cache-Control': 'no-cache',
                           'Connection': 'keep-alive'})

@app.route('/')
def index():
    """Simple HTML page to test the API."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Road Attribution Task</title>
        <style>
            body { font-family: monospace; margin: 20px; }
            #output { 
                white-space: pre; 
                background-color: #f0f0f0; 
                padding: 10px; 
                border: 1px solid #ccc; 
                height: 500px; 
                overflow-y: auto; 
            }
        </style>
    </head>
    <body>
        <h1>Road Attribution Task</h1>
        <div>
            <label for="tile">Tile ID:</label>
            <input type="text" id="tile" value="23599610">
            <button id="start-btn">Start Processing</button>
        </div>
        <h2>Progress:</h2>
        <div id="output"></div>
        
        <script>
            document.getElementById('start-btn').addEventListener('click', async () => {
                const tile = document.getElementById('tile').value;
                const outputDiv = document.getElementById('output');
                
                outputDiv.textContent = `Starting task for tile: ${tile}...\n`;
                
                try {
                    // Start the task
                    const response = await fetch('/start-task', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ tile: tile })
                    });
                    
                    const data = await response.json();
                    const taskId = data.task_id;
                    
                    outputDiv.textContent += `Task started with ID: ${taskId}\n`;
                    
                    // Connect to the event stream
                    const eventSource = new EventSource(`/task-progress/${taskId}`);
                    
                    eventSource.onmessage = (event) => {
                        const progressData = JSON.parse(event.data);
                        const formatted = JSON.stringify(progressData, null, 2);
                        outputDiv.textContent += `${formatted}\n\n`;
                        outputDiv.scrollTop = outputDiv.scrollHeight; // Auto-scroll to bottom
                    };
                    
                    eventSource.onerror = () => {
                        outputDiv.textContent += `Task completed.\n`;
                        eventSource.close();
                    };
                    
                } catch (error) {
                    outputDiv.textContent += `Error: ${error.message}\n`;
                }
            });
        </script>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True, port=Config.PORT)
