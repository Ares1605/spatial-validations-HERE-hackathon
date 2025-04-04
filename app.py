from flask import Flask, Response, request, jsonify, send_from_directory
import json
import threading
import queue
import time
from dotenv import load_dotenv
import helper
from road_attribution_correction import RoadAttributionCorrection
from road_segment_corrector import RoadSegmentCorrector
from typing import Dict, List, Any, Tuple
import os

# Initialize Flask app - now without a static folder to handle static files manually
app = Flask(__name__)

# Dictionary to store progress queues for each task
progress_queues = {}

def process_tiles(task_id):
    """Function that processes all available tiles and reports progress."""
    q = progress_queues[task_id]
    
    # Load environment variables
    load_dotenv()
    
    # Get all available tiles
    tiles = helper.get_tiles()
    total_tiles = len(tiles)
    
    # Report initial status
    q.put(json.dumps({
        "status": "started",
        "message": f"Starting processing of {total_tiles} tiles",
        "total_tiles": total_tiles,
        "current_tile_index": 0,
        "tiles": tiles
    }, default=str))
    
    # Process each tile
    for tile_index, tile in enumerate(tiles):
        # Report starting this tile
        q.put(json.dumps({
            "status": "processing",
            "current_tile": tile,
            "current_tile_index": tile_index + 1,
            "total_tiles": total_tiles,
            "progress_percent": round((tile_index / total_tiles) * 100, 1),
            "message": f"Processing tile {tile_index + 1}/{total_tiles}: {tile}"
        }, default=str))
        
        try:
            # Get violations data for this tile
            violations_data = helper.get_violations_data(tile)
            
            # Report starting the road attribution correction
            q.put(json.dumps({
                "status": "processing",
                "current_tile": tile,
                "current_tile_index": tile_index + 1,
                "total_tiles": total_tiles,
                "current_step": "RoadAttributionCorrection",
                "message": f"Starting RoadAttributionCorrection for tile: {tile}"
            }, default=str))
            
            # Process the road attribution correction
            road_attr_corrector = RoadAttributionCorrection(tile, violations_data)
            road_attr_result, remaining_violations = road_attr_corrector.process()
            
            # Package the results in the expected format
            road_attr_formatted = {
                "type": "road-attribution-correction",
                "data": road_attr_result
            }
            
            # Report completion of road attribution correction
            q.put(json.dumps({
                "status": "processing",
                "current_tile": tile,
                "current_tile_index": tile_index + 1,
                "total_tiles": total_tiles,
                "current_step": "RoadAttributionCorrection completed",
                "message": f"Completed RoadAttributionCorrection for tile: {tile}",
                "road_attr_fixed_count": len(road_attr_result),
                "remaining_violations_count": len(remaining_violations),
                "road_attr_result": road_attr_formatted
            }, default=str))
            
            # Report starting the road segment corrector
            q.put(json.dumps({
                "status": "processing",
                "current_tile": tile,
                "current_tile_index": tile_index + 1,
                "total_tiles": total_tiles,
                "current_step": "RoadSegmentCorrector",
                "message": f"Starting RoadSegmentCorrector for tile: {tile}"
            }, default=str))
            
            # Process the road segment corrector
            road_segment_corrector = RoadSegmentCorrector(tile, violations_data)
            road_segment_result, segment_remaining_violations = road_segment_corrector.process()
            
            # Package the results in the expected format
            road_segment_formatted = {
                "type": "road-segment-corrector",
                "data": road_segment_result
            }
            
            # Calculate cumulative violations numbers
            total_violations = len(violations_data)
            total_fixed = len(road_attr_result) + len(road_segment_result)
            total_remaining = len(remaining_violations) + len(segment_remaining_violations)
            
            # Report completion of road segment corrector
            q.put(json.dumps({
                "status": "processing",
                "current_tile": tile,
                "current_tile_index": tile_index + 1,
                "total_tiles": total_tiles,
                "current_step": "RoadSegmentCorrector completed",
                "message": f"Completed RoadSegmentCorrector for tile: {tile}",
                "total_violations": total_violations,
                "road_attr_fixed_count": len(road_attr_result),
                "road_segment_fixed_count": len(road_segment_result),
                "total_fixed_count": total_fixed,
                "total_remaining_count": total_remaining,
                "road_attr_result": road_attr_formatted,
                "road_segment_result": road_segment_formatted
            }, default=str))
            
        except Exception as e:
            # Report any errors for this tile but continue with the next
            q.put(json.dumps({
                "status": "error",
                "current_tile": tile,
                "current_tile_index": tile_index + 1,
                "total_tiles": total_tiles,
                "message": f"Error processing tile {tile}: {str(e)}",
                "error": str(e)
            }, default=str))
    
    # Calculate and track overall statistics
    total_attr_fixed = 0
    total_segment_fixed = 0
    total_violations_processed = 0
    all_attr_results = []
    all_segment_results = []
    
    # Review message history to calculate totals
    message_history = list(q.queue)
    for message in message_history:
        if isinstance(message, str):
            try:
                data = json.loads(message)
                if data.get("road_attr_fixed_count"):
                    total_attr_fixed += data.get("road_attr_fixed_count", 0)
                if data.get("road_segment_fixed_count"):
                    total_segment_fixed += data.get("road_segment_fixed_count", 0)
                if data.get("total_violations"):
                    total_violations_processed += data.get("total_violations", 0)
                
                # Collect all results
                if data.get("road_attr_result"):
                    all_attr_results.append(data.get("road_attr_result"))
                if data.get("road_segment_result"):
                    all_segment_results.append(data.get("road_segment_result"))
            except:
                pass
    
    # Report final completion with comprehensive statistics
    q.put(json.dumps({
        "status": "completed",
        "message": f"Completed processing all {total_tiles} tiles",
        "total_tiles": total_tiles,
        "total_violations_processed": total_violations_processed,
        "total_attr_fixed": total_attr_fixed,
        "total_segment_fixed": total_segment_fixed,
        "total_fixed": total_attr_fixed + total_segment_fixed,
        "attribution_correction_percentage": round((total_attr_fixed / max(total_violations_processed, 1)) * 100, 2),
        "segment_correction_percentage": round((total_segment_fixed / max(total_violations_processed, 1)) * 100, 2),
        "overall_correction_percentage": round(((total_attr_fixed + total_segment_fixed) / max(total_violations_processed, 1)) * 100, 2)
    }, default=str))
    
    # Signal completion
    q.put(None)

@app.route('/start-processing', methods=['POST'])
def start_processing():
    """Start a new processing task for all tiles and return its ID."""
    # Create a unique task ID
    task_id = f"task_{int(time.time())}"
    
    # Create a queue for this task
    progress_queues[task_id] = queue.Queue()
    
    # Start the processing in a separate thread
    thread = threading.Thread(target=process_tiles, args=(task_id,))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "task_id": task_id,
        "status": "started"
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

# Serve static files from the root directory
@app.route('/')
def index():
    """Serve the main dashboard HTML page."""
    return send_from_directory('.', 'index.html')

@app.route('/index.css')
def serve_css():
    """Serve the CSS file."""
    return send_from_directory('.', 'index.css')

@app.route('/index.js')
def serve_js():
    """Serve the JavaScript file."""
    return send_from_directory('.', 'index.js')

if __name__ == '__main__':
    app.run(debug=True)
