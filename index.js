// index.js

// DOM elements (keep existing selections)
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
const summaryEl = document.getElementById('summary'); // Summary section
const totalFixedEl = document.getElementById('total-fixed'); // Total fixed element
const attrFixedEl = document.getElementById('attr-fixed'); // Attribution fixed element
const segmentFixedEl = document.getElementById('segment-fixed'); // Segment fixed element
const fixPercentageEl = document.getElementById('fix-percentage'); // Total fix percentage
const attrPercentageEl = document.getElementById('attr-percentage'); // Attr fix percentage
const segmentPercentageEl = document.getElementById('segment-percentage'); // Segment fix percentage
const chartAttrEl = document.getElementById('chart-attr');
const chartSegmentEl = document.getElementById('chart-segment');
const chartRemainingEl = document.getElementById('chart-remaining');
const correctionsDetailsEl = document.getElementById('corrections-details');
const toggleDetailsBtn = document.getElementById('toggle-details-btn');
const correctionsContentEl = document.getElementById('corrections-content');
const attrCorrectionsTableEl = document.getElementById('attr-corrections-table').querySelector('tbody');
const segmentCorrectionsTableEl = document.getElementById('segment-corrections-table').querySelector('tbody');

// Toggle raw JSON display (keep existing)
toggleJsonBtn.addEventListener('click', () => {
    if (rawJsonEl.style.display === 'none') {
        rawJsonEl.style.display = 'block';
        toggleJsonBtn.textContent = 'Hide';
    } else {
        rawJsonEl.style.display = 'none';
        toggleJsonBtn.textContent = 'Show';
    }
});

// Toggle details display (keep existing)
toggleDetailsBtn.addEventListener('click', () => {
    if (correctionsContentEl.style.display === 'none') {
        correctionsContentEl.style.display = 'block';
        toggleDetailsBtn.textContent = 'Hide';
    } else {
        correctionsContentEl.style.display = 'none';
        toggleDetailsBtn.textContent = 'Show';
    }
});

// Function to format JSON with syntax highlighting (keep existing)
function formatJson(json) {
    if (typeof json !== 'string') {
        json = JSON.stringify(json, null, 2);
    }
    return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
        function (match) {
            let cls = 'json-number';
            if (/^"/.test(match)) {
                if (/:$/.test(match)) {
                    cls = 'json-key';
                    match = match.replace(/:$/, '') + ':';
                } else {
                    cls = 'json-string';
                }
            } else if (/true|false/.test(match)) {
                cls = 'json-boolean';
            } else if (/null/.test(match)) {
                cls = 'json-null';
            }
            if (cls === 'json-key') {
                return '<span class="' + cls + '">' + match.replace(/:$/, '') + '</span>:';
            }
            return '<span class="' + cls + '">' + match + '</span>';
        });
}

// --- Updated State Variables ---
let tileStatuses = {};
let totalViolations = 0; // Running total violations processed
let totalFixedCount = 0; // Running total fixed violations
let attrFixedCount = 0; // Running total attribution fixes
let segmentFixedCount = 0; // Running total segment fixes
let tilesProcessedCount = 0; // Count of tiles fully processed

// Arrays to store correction details (keep existing)
let attrCorrections = [];
let segmentCorrections = [];

// Array to store all raw JSON messages (keep existing)
let allJsonMessages = [];

// Add a log entry (keep existing)
function addLog(message, type = '') {
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${type}`;
    logEntry.textContent = message;
    logsEl.appendChild(logEntry);
    logsEl.scrollTop = logsEl.scrollHeight;
}

// Update the tile grid (keep existing)
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
            // Initialize status
            if (!tileStatuses[tile]) {
                tileStatuses[tile] = 'pending';
            }
        });
    }

    // Update states based on current progress
    tiles.forEach((tile, index) => {
        const tileEl = document.getElementById(`tile-${tile}`);
        if (!tileEl) return; // Skip if element not found yet

        if (index + 1 === currentTileIndex && tileStatuses[tile] !== 'error' && tileStatuses[tile] !== 'completed') {
             tileEl.className = 'tile-box current';
             tileStatuses[tile] = 'current';
        } else if (tileStatuses[tile] === 'completed') {
            tileEl.className = 'tile-box completed';
        } else if (tileStatuses[tile] === 'error') {
             tileEl.className = 'tile-box error';
        } else if (tileStatuses[tile] === 'pending' && index + 1 < currentTileIndex) {
            // Mark implicitly completed tiles if we moved past without explicit completion (e.g., after error)
            tileEl.className = 'tile-box completed';
            tileStatuses[tile] = 'completed';
        } else if (tileStatuses[tile] === 'pending') {
             tileEl.className = 'tile-box pending';
        }
    });
}


// Update the corrections details tables (keep existing)
function updateCorrectionsDetails() {
    // Clear existing rows
    attrCorrectionsTableEl.innerHTML = '';
    segmentCorrectionsTableEl.innerHTML = '';

    // Add Road Attribution Corrections
    attrCorrections.forEach(correction => {
        if (correction && correction.data) {
             Object.entries(correction.data).forEach(([violationId, details]) => {
                const row = document.createElement('tr');
                const shortViolationId = violationId.split(':').pop();
                const shortTopologyId = details.topology_id ? details.topology_id.split(':').pop() : '-';
                row.innerHTML = `
                    <td title="${violationId}">${shortViolationId}</td>
                    <td title="${details.topology_id}">${shortTopologyId}</td>
                    <td>${details.current_pedestrian_access !== undefined ? (details.current_pedestrian_access ? 'Yes' : 'No') : '-'}</td>
                    <td>${details.recommended_pedestrian_access !== undefined ? (details.recommended_pedestrian_access ? 'Yes' : 'No') : '-'}</td>
                    <td>${details.confidence !== undefined ? (details.confidence * 100).toFixed(1) + '%' : '-'}</td>
                `;
                attrCorrectionsTableEl.appendChild(row);
            });
        }
    });

    // Add Road Segment Corrections
    segmentCorrections.forEach(correction => {
         if (correction && correction.data) {
            Object.entries(correction.data).forEach(([violationId, details]) => {
                const row = document.createElement('tr');
                const shortViolationId = violationId.split(':').pop();
                const shortSignId = details.sign_id ? details.sign_id.split(':').pop() : '-';
                const shortCurrentTopologyId = details.current_topology_id ? details.current_topology_id.split(':').pop() : '-';
                const shortNewTopologyId = details.new_topology_id ? details.new_topology_id.split(':').pop() : '-';
                row.innerHTML = `
                    <td title="${violationId}">${shortViolationId}</td>
                    <td title="${details.sign_id}">${shortSignId}</td>
                    <td title="${details.current_topology_id}">${shortCurrentTopologyId}</td>
                    <td title="${details.new_topology_id}">${shortNewTopologyId}</td>
                    <td>${details.confidence_score !== undefined ? (details.confidence_score * 100).toFixed(1) + '%' : '-'}</td>
                `;
                segmentCorrectionsTableEl.appendChild(row);
            });
        }
    });

    // Show the corrections details section if there are any corrections
    if (attrCorrections.length > 0 || segmentCorrections.length > 0) {
        correctionsDetailsEl.style.display = 'block';
    } else {
         correctionsDetailsEl.style.display = 'none';
    }
}


// --- Updated Chart Function ---
// Update the visualization chart (calculates remaining based on total V)
function updateViolationsChart(currentTotalViolations, currentAttrFixed, currentSegmentFixed) {
    if (currentTotalViolations > 0) {
        const totalFixed = currentAttrFixed + currentSegmentFixed;
        const attrPercent = Math.min(100, (currentAttrFixed / currentTotalViolations) * 100);
        const segmentPercent = Math.min(100 - attrPercent, (currentSegmentFixed / currentTotalViolations) * 100);
        const remainingPercent = Math.max(0, 100 - attrPercent - segmentPercent);

        chartAttrEl.style.width = `${attrPercent}%`;
        chartSegmentEl.style.width = `${segmentPercent}%`;
        chartRemainingEl.style.width = `${remainingPercent}%`;
    } else {
        chartAttrEl.style.width = `0%`;
        chartSegmentEl.style.width = `0%`;
        chartRemainingEl.style.width = `100%`;
        chartRemainingEl.style.backgroundColor = '#ecf0f1';
    }
}


// --- Updated UI Update Function ---
// Function to update the summary statistics UI incrementally
function updateSummaryUI() {
    totalFixedEl.textContent = totalFixedCount;
    attrFixedEl.textContent = attrFixedCount;
    segmentFixedEl.textContent = segmentFixedCount;

    const totalFixPercent = totalViolations > 0 ? ((totalFixedCount / totalViolations) * 100).toFixed(1) : 0;
    const attrFixPercent = totalViolations > 0 ? ((attrFixedCount / totalViolations) * 100).toFixed(1) : 0;
    const segmentFixPercent = totalViolations > 0 ? ((segmentFixedCount / totalViolations) * 100).toFixed(1) : 0;

    fixPercentageEl.textContent = `${totalFixPercent}%`;
    attrPercentageEl.textContent = `${attrFixPercent}%`;
    segmentPercentageEl.textContent = `${segmentFixPercent}%`;

    updateViolationsChart(totalViolations, attrFixedCount, segmentFixedCount);

    if (summaryEl.style.display === 'none' && (tilesProcessedCount > 0 || totalViolations > 0) ) {
         summaryEl.style.display = 'block';
         chartRemainingEl.style.backgroundColor = '#e74c3c';
    }
}


// Start processing (keep most existing logic, update SSE handler)
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
        progressBar.style.width = '0%';
        progressText.textContent = '0%';

        // --- Reset counters and data ---
        totalFixedCount = 0;
        attrFixedCount = 0;
        segmentFixedCount = 0;
        totalViolations = 0;
        tilesProcessedCount = 0;
        tileStatuses = {};
        allJsonMessages = [];
        attrCorrections = [];
        segmentCorrections = [];
        // Reset summary UI text explicitly
        updateSummaryUI();


        // Start the task
        addLog('Starting processing task...');
        // ***** THIS IS THE CORRECTED FETCH CALL *****
        const response = await fetch('/start-processing', {
             method: 'POST', // Specify the POST method
             headers: {
                 'Content-Type': 'application/json' // Keep the header
             }
             // No request body is needed for this endpoint
        });
        // ***** END OF CORRECTION *****

        // Check if the initial request failed
        if (!response.ok) {
            throw new Error(`Failed to start task: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        const taskId = data.task_id;
        addLog(`Task started with ID: ${taskId}`);

        // Connect to the event stream
        const eventSource = new EventSource(`/task-progress/${taskId}`);

        // --- Updated SSE onmessage Handler ---
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
                        // Initialize tile statuses
                        data.tiles.forEach(tile => { tileStatuses[tile] = 'pending'; });
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

                        // Mark tile as current in the grid (if not already error/completed)
                        if (tileStatuses[data.current_tile] === 'pending') {
                             tileStatuses[data.current_tile] = 'current';
                        }

                         // Ensure tiles list is available for updating the grid
                         const tilesForGrid = allJsonMessages[0]?.tiles || Object.keys(tileStatuses);
                         if (tilesForGrid.length > 0) {
                              updateTileGrid(tilesForGrid, data.current_tile_index);
                         }
                    }

                    // Update current step
                    if (data.current_step) {
                        currentStepEl.textContent = data.current_step;
                    }

                    // Add log message
                    addLog(data.message);

                    // --- Incremental Summary Update ---
                    if (data.road_segment_fixed_count !== undefined) {
                        // This message marks the completion of processing for one tile
                        totalViolations += data.total_violations || 0;
                        attrFixedCount += data.road_attr_fixed_count || 0;
                        segmentFixedCount += data.road_segment_fixed_count || 0;
                        totalFixedCount = attrFixedCount + segmentFixedCount;

                        // Mark tile completed only if not already marked as error
                        if (data.current_tile && tileStatuses[data.current_tile] !== 'error') {
                             tileStatuses[data.current_tile] = 'completed';
                        }
                        tilesProcessedCount++;

                        updateSummaryUI();

                         // Update grid after marking tile completed/error
                        const tilesForGrid = allJsonMessages[0]?.tiles || Object.keys(tileStatuses);
                         if (tilesForGrid.length > 0) {
                             updateTileGrid(tilesForGrid, data.current_tile_index + 1); // Update grid, show next as potentially current
                         }
                    }
                    // --- End Incremental Summary Update ---


                    // Collect detailed corrections
                    if (data.road_attr_result && data.road_attr_result.data && Object.keys(data.road_attr_result.data).length > 0) {
                        attrCorrections.push(data.road_attr_result);
                        updateCorrectionsDetails();
                    }
                    if (data.road_segment_result && data.road_segment_result.data && Object.keys(data.road_segment_result.data).length > 0) {
                        segmentCorrections.push(data.road_segment_result);
                        updateCorrectionsDetails();
                    }
                    break;

                case 'error':
                    addLog(`ERROR: ${data.message}`, 'error');
                    if (data.current_tile) {
                        tileStatuses[data.current_tile] = 'error';
                        // Increment processed count even on error
                        tilesProcessedCount++;
                         // Update summary to reflect processed tile count change
                         updateSummaryUI();
                         // Update tile grid immediately
                        const tilesForGrid = allJsonMessages[0]?.tiles || Object.keys(tileStatuses);
                         if (tilesForGrid.length > 0) {
                           updateTileGrid(tilesForGrid, data.current_tile_index + 1); // Update grid after marking error
                         }
                    }
                    // Update overall progress bar slightly
                    if (data.current_tile_index && data.total_tiles) {
                         const errorProgress = Math.round(((data.current_tile_index) / data.total_tiles) * 100, 1);
                         progressBar.style.width = `${errorProgress}%`;
                         progressText.textContent = `${errorProgress}%`;
                    }
                    break;

                case 'completed':
                    // Final update using definitive totals from backend
                    progressBar.style.width = '100%';
                    progressText.textContent = '100%';
                    statusEl.textContent = data.message;
                    currentTileInfo.style.display = 'none';

                    totalViolations = data.total_violations_processed;
                    attrFixedCount = data.total_attr_fixed;
                    segmentFixedCount = data.total_segment_fixed;
                    totalFixedCount = data.total_fixed;
                    tilesProcessedCount = data.total_tiles;

                    updateSummaryUI();

                    // Final grid update
                     const finalTiles = allJsonMessages[0]?.tiles || Object.keys(tileStatuses);
                     if (finalTiles.length > 0) {
                         finalTiles.forEach(tile => {
                            if (tileStatuses[tile] !== 'error') {
                                tileStatuses[tile] = 'completed';
                            }
                         });
                         updateTileGrid(finalTiles, data.total_tiles + 1); // Ensure all shown as completed/error
                     }

                    startBtn.disabled = false;
                    eventSource.close();
                    addLog('Processing finished.', 'success');
                    break;
            }
        };

        // SSE Error Handler
        eventSource.onerror = (error) => {
            console.error("EventSource failed:", error);
            addLog('Error connecting to progress stream. Please try again.', 'error');
            statusEl.textContent = 'Connection error.';
            eventSource.close();
            startBtn.disabled = false;
        };

    } catch (error) {
        console.error('Error starting processing:', error);
        statusEl.textContent = 'Failed to start processing.';
        addLog(`Error: ${error.message}`, 'error');
        startBtn.disabled = false;
    }
}); // End of startBtn listener
