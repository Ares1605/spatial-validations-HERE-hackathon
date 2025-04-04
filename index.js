// DOM elements
const startBtn = document.getElementById('start-btn');
const statusEl = document.getElementById('status');
const progressBar = document.getElementById('progress-bar');
const progressText = document.getElementById('progress-text');
const currentTileInfo = document.getElementById('current-tile-info');
const currentTileEl = document.getElementById('current-tile');
const tileProgressEl = document.getElementById('tile-progress');
const currentStepEl = document.getElementById('current-step');
const tileGridEl = document.getElementById('tile-grid');
const logsEl = document.getElementById('logs');
const rawJsonEl = document.getElementById('raw-json');
const toggleJsonBtn = document.getElementById('toggle-json-btn');
const summaryEl = document.getElementById('summary');
const totalTilesProcessedEl = document.getElementById('total-tiles-processed');
const totalViolationsEl = document.getElementById('total-violations');
const totalFixedEl = document.getElementById('total-fixed');
const attrFixedEl = document.getElementById('attr-fixed');
const segmentFixedEl = document.getElementById('segment-fixed');
const fixPercentageEl = document.getElementById('fix-percentage');
const attrPercentageEl = document.getElementById('attr-percentage');
const segmentPercentageEl = document.getElementById('segment-percentage');
const chartAttrEl = document.getElementById('chart-attr');
const chartSegmentEl = document.getElementById('chart-segment');
const chartRemainingEl = document.getElementById('chart-remaining');
const correctionsDetailsEl = document.getElementById('corrections-details');
const toggleDetailsBtn = document.getElementById('toggle-details-btn');
const correctionsContentEl = document.getElementById('corrections-content');
const attrCorrectionsTableEl = document.getElementById('attr-corrections-table').querySelector('tbody');
const segmentCorrectionsTableEl = document.getElementById('segment-corrections-table').querySelector('tbody');

// Toggle raw JSON display
toggleJsonBtn.addEventListener('click', () => {
    if (rawJsonEl.style.display === 'none') {
        rawJsonEl.style.display = 'block';
        toggleJsonBtn.textContent = 'Hide';
    } else {
        rawJsonEl.style.display = 'none';
        toggleJsonBtn.textContent = 'Show';
    }
});

// Toggle details display
toggleDetailsBtn.addEventListener('click', () => {
    if (correctionsContentEl.style.display === 'none') {
        correctionsContentEl.style.display = 'block';
        toggleDetailsBtn.textContent = 'Hide';
    } else {
        correctionsContentEl.style.display = 'none';
        toggleDetailsBtn.textContent = 'Show';
    }
});

// Function to format JSON with syntax highlighting
function formatJson(json) {
    if (typeof json !== 'string') {
        json = JSON.stringify(json, null, 2);
    }
    
    // Replace key-value pairs with styled spans
    return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, 
        function (match) {
            let cls = 'json-number';
            if (/^"/.test(match)) {
                if (/:$/.test(match)) {
                    cls = 'json-key';
                    // Remove the colon from the key
                    match = match.replace(/:$/, '') + ':';
                } else {
                    cls = 'json-string';
                }
            } else if (/true|false/.test(match)) {
                cls = 'json-boolean';
            } else if (/null/.test(match)) {
                cls = 'json-null';
            }
            
            // For keys, keep the colon outside the colored span
            if (cls === 'json-key') {
                return '<span class="' + cls + '">' + match.replace(/:$/, '') + '</span>:';
            }
            return '<span class="' + cls + '">' + match + '</span>';
        });
}

// Variables to track progress
let tileStatuses = {};
let totalViolations = 0;
let totalFixedCount = 0;
let attrFixedCount = 0;
let segmentFixedCount = 0;

// Arrays to store correction details
let attrCorrections = [];
let segmentCorrections = [];

// Array to store all raw JSON messages
let allJsonMessages = [];

// Add a log entry
function addLog(message, type = '') {
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${type}`;
    logEntry.textContent = message;
    logsEl.appendChild(logEntry);
    logsEl.scrollTop = logsEl.scrollHeight;
}

// Update the tile grid
function updateTileGrid(tiles, currentTileIndex) {
    // Clear existing tiles
    if (!tileGridEl.hasChildNodes()) {
        // Only create tiles if they don't exist yet
        tiles.forEach((tile, index) => {
            const tileBox = document.createElement('div');
            tileBox.className = 'tile-box pending';
            tileBox.id = `tile-${tile}`;
            tileBox.textContent = tile;
            tileBox.title = tile;
            tileGridEl.appendChild(tileBox);
        });
    }
    
    // Update states based on current progress
    tiles.forEach((tile, index) => {
        const tileEl = document.getElementById(`tile-${tile}`);
        if (index + 1 === currentTileIndex) {
            tileEl.className = 'tile-box current';
        } else if (index + 1 < currentTileIndex) {
            tileEl.className = tileStatuses[tile] === 'error' ? 
                'tile-box error' : 'tile-box completed';
        }
    });
}

// Update the corrections details tables
function updateCorrectionsDetails() {
    // Clear existing rows
    attrCorrectionsTableEl.innerHTML = '';
    segmentCorrectionsTableEl.innerHTML = '';
    
    // Add Road Attribution Corrections
    attrCorrections.forEach(correction => {
        Object.entries(correction.data).forEach(([violationId, details]) => {
            const row = document.createElement('tr');
            
            // Create shortened IDs for readability
            const shortViolationId = violationId.split(':').pop();
            const shortTopologyId = details.topology_id.split(':').pop();
            
            row.innerHTML = `
                <td title="${violationId}">${shortViolationId}</td>
                <td title="${details.topology_id}">${shortTopologyId}</td>
                <td>${details.current_pedestrian_access ? 'Yes' : 'No'}</td>
                <td>${details.recommended_pedestrian_access ? 'Yes' : 'No'}</td>
                <td>${(details.confidence * 100).toFixed(1)}%</td>
            `;
            attrCorrectionsTableEl.appendChild(row);
        });
    });
    
    // Add Road Segment Corrections
    segmentCorrections.forEach(correction => {
        Object.entries(correction.data).forEach(([violationId, details]) => {
            const row = document.createElement('tr');
            
            // Create shortened IDs for readability
            const shortViolationId = violationId.split(':').pop();
            const shortSignId = details.sign_id.split(':').pop();
            const shortCurrentTopologyId = details.current_topology_id.split(':').pop();
            const shortNewTopologyId = details.new_topology_id.split(':').pop();
            
            row.innerHTML = `
                <td title="${violationId}">${shortViolationId}</td>
                <td title="${details.sign_id}">${shortSignId}</td>
                <td title="${details.current_topology_id}">${shortCurrentTopologyId}</td>
                <td title="${details.new_topology_id}">${shortNewTopologyId}</td>
                <td>${(details.confidence_score * 100).toFixed(1)}%</td>
            `;
            segmentCorrectionsTableEl.appendChild(row);
        });
    });
    
    // Show the corrections details section if there are any corrections
    if (attrCorrections.length > 0 || segmentCorrections.length > 0) {
        correctionsDetailsEl.style.display = 'block';
    }
}

// Update the visualization chart
function updateViolationsChart(totalViolations, attrFixed, segmentFixed) {
    if (totalViolations > 0) {
        const attrPercent = (attrFixed / totalViolations) * 100;
        const segmentPercent = (segmentFixed / totalViolations) * 100;
        const remainingPercent = 100 - attrPercent - segmentPercent;
        
        chartAttrEl.style.width = `${attrPercent}%`;
        chartSegmentEl.style.width = `${segmentPercent}%`;
        chartRemainingEl.style.width = `${remainingPercent}%`;
    }
}

// Start processing
startBtn.addEventListener('click', async () => {
    try {
        // Reset UI elements
        startBtn.disabled = true;
        statusEl.textContent = 'Starting processing...';
        tileGridEl.innerHTML = '';
        logsEl.innerHTML = '';
        rawJsonEl.innerHTML = '';
        summaryEl.style.display = 'none';
        correctionsDetailsEl.style.display = 'none';
        attrCorrectionsTableEl.innerHTML = '';
        segmentCorrectionsTableEl.innerHTML = '';
        
        // Reset counters and data
        totalFixedCount = 0;
        attrFixedCount = 0;
        segmentFixedCount = 0;
        totalViolations = 0;
        tileStatuses = {};
        allJsonMessages = [];
        attrCorrections = [];
        segmentCorrections = [];
        
        // Start the task
        addLog('Starting processing task...');
        const response = await fetch('/start-processing', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        const taskId = data.task_id;
        addLog(`Task started with ID: ${taskId}`);
        
        // Connect to the event stream
        const eventSource = new EventSource(`/task-progress/${taskId}`);
        
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            // Store raw JSON for developer view
            allJsonMessages.push(data);
            rawJsonEl.innerHTML = formatJson(allJsonMessages);
            
            // Handle different status updates
            switch (data.status) {
                case 'started':
                    statusEl.textContent = data.message;
                    addLog(`${data.message}`, 'step');
                    if (data.tiles && data.tiles.length > 0) {
                        updateTileGrid(data.tiles, data.current_tile_index);
                    }
                    break;
                    
                case 'processing':
                    // Update progress bar
                    if (data.progress_percent !== undefined) {
                        progressBar.style.width = `${data.progress_percent}%`;
                        progressText.textContent = `${data.progress_percent}%`;
                    }
                    
                    // Update current tile info
                    if (data.current_tile) {
                        currentTileInfo.style.display = 'flex';
                        currentTileEl.textContent = data.current_tile;
                        tileProgressEl.textContent = `Tile ${data.current_tile_index} of ${data.total_tiles}`;
                        
                        // Update tile grid
                        updateTileGrid(data.tiles || Object.keys(tileStatuses), data.current_tile_index);
                    }
                    
                    // Update current step
                    if (data.current_step) {
                        currentStepEl.textContent = data.current_step;
                    }
                    
                    // Add log message
                    addLog(data.message);
                    
                    // Track violation counts
                    if (data.total_violations) {
                        totalViolations += data.total_violations;
                    }
                    
                    // Track fix counts and store detailed correction data
                    if (data.road_attr_result && data.road_attr_result.type === 'road-attribution-correction') {
                        attrFixedCount = Object.keys(data.road_attr_result.data).length;
                        attrCorrections.push(data.road_attr_result);
                        
                        // Log details about the road attribution corrections
                        addLog(`Found ${attrFixedCount} road attribution corrections`, 'success');
                        
                        // Update corrections details view
                        updateCorrectionsDetails();
                    }
                    
                    if (data.road_segment_result && data.road_segment_result.type === 'road-segment-corrector') {
                        segmentFixedCount = Object.keys(data.road_segment_result.data).length;
                        segmentCorrections.push(data.road_segment_result);
                        
                        // Log details about the road segment corrections
                        addLog(`Found ${segmentFixedCount} road segment corrections`, 'success');
                        
                        // Update corrections details view
                        updateCorrectionsDetails();
                    }
                    
                    if (data.total_fixed_count) {
                        totalFixedCount = data.total_fixed_count;
                    }
                    
                    statusEl.textContent = data.message;
                    break;
                    
                case 'error':
                    addLog(`ERROR: ${data.message}`, 'error');
                    if (data.current_tile) {
                        tileStatuses[data.current_tile] = 'error';
                        updateTileGrid(Object.keys(tileStatuses), data.current_tile_index);
                    }
                    break;
                    
                case 'completed':
                    // Update final progress
                    progressBar.style.width = '100%';
                    progressText.textContent = '100%';
                    statusEl.textContent = data.message;
                    addLog(data.message, 'success');
                    startBtn.disabled = false;
                    
                    // Show summary
                    summaryEl.style.display = 'block';
                    
                    // Calculate total counts from all corrections
                    let totalAttrViolations = 0;
                    let totalSegmentViolations = 0;
                    
                    attrCorrections.forEach(correction => {
                        totalAttrViolations += Object.keys(correction.data).length;
                    });
                    
                    segmentCorrections.forEach(correction => {
                        totalSegmentViolations += Object.keys(correction.data).length;
                    });
                    
                    // Use data from the server if available, otherwise use our calculated totals
                    const finalTotalViolations = data.total_violations_processed || totalViolations;
                    const finalAttrFixed = data.total_attr_fixed || totalAttrViolations;
                    const finalSegmentFixed = data.total_segment_fixed || totalSegmentViolations;
                    const finalTotalFixed = finalAttrFixed + finalSegmentFixed;
                    
                    // Update the summary display
                    totalTilesProcessedEl.textContent = data.total_tiles || 0;
                    totalViolationsEl.textContent = finalTotalViolations;
                    totalFixedEl.textContent = finalTotalFixed;
                    attrFixedEl.textContent = finalAttrFixed;
                    segmentFixedEl.textContent = finalSegmentFixed;
                    
                    // Calculate and display percentages
                    if (finalTotalViolations > 0) {
                        const overallPercent = data.overall_correction_percentage || 
                                              Math.round((finalTotalFixed / finalTotalViolations) * 100);
                        const attrPercent = data.attribution_correction_percentage || 
                                           Math.round((finalAttrFixed / finalTotalViolations) * 100);
                        const segmentPercent = data.segment_correction_percentage || 
                                              Math.round((finalSegmentFixed / finalTotalViolations) * 100);
                        
                        fixPercentageEl.textContent = `${overallPercent}% of violations fixed`;
                        attrPercentageEl.textContent = `${attrPercent}% of total violations`;
                        segmentPercentageEl.textContent = `${segmentPercent}% of total violations`;
                        
                        // Update the visualization chart
                        updateViolationsChart(finalTotalViolations, finalAttrFixed, finalSegmentFixed);
                    }
                    break;
            }
        };
        
        eventSource.onerror = () => {
            addLog('Connection closed or error occurred.', 'error');
            startBtn.disabled = false;
            eventSource.close();
        };
        
    } catch (error) {
        addLog(`Error: ${error.message}`, 'error');
        statusEl.textContent = `Error: ${error.message}`;
        startBtn.disabled = false;
    }
});
