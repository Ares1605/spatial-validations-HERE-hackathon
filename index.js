
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
const totalTilesProcessedEl = document.getElementById('total-tiles-processed');
const totalViolationsEl = document.getElementById('total-violations'); // Total violations element
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
    // ... existing code ...
});

// Toggle details display (keep existing)
toggleDetailsBtn.addEventListener('click', () => {
    // ... existing code ...
});

// Function to format JSON with syntax highlighting (keep existing)
function formatJson(json) {
    // ... existing code ...
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
    // ... existing code ...
}

// Update the tile grid (keep existing)
function updateTileGrid(tiles, currentTileIndex) {
    // ... existing code ...
}

// Update the corrections details tables (keep existing)
function updateCorrectionsDetails() {
    // ... existing code ...
}

// --- Updated Chart Function ---
// Update the visualization chart (calculates remaining based on total V)
function updateViolationsChart(currentTotalViolations, currentAttrFixed, currentSegmentFixed) {
    if (currentTotalViolations > 0) {
        const totalFixed = currentAttrFixed + currentSegmentFixed;
        // Ensure percentages don't exceed 100 due to potential calculation nuances
        const attrPercent = Math.min(100, (currentAttrFixed / currentTotalViolations) * 100);
        const segmentPercent = Math.min(100 - attrPercent, (currentSegmentFixed / currentTotalViolations) * 100);
        const remainingPercent = Math.max(0, 100 - attrPercent - segmentPercent);

        chartAttrEl.style.width = `${attrPercent}%`;
        chartSegmentEl.style.width = `${segmentPercent}%`;
        chartRemainingEl.style.width = `${remainingPercent}%`;
    } else {
        // Reset chart if no violations yet
        chartAttrEl.style.width = `0%`;
        chartSegmentEl.style.width = `0%`;
        chartRemainingEl.style.width = `100%`; // Show remaining bar initially full or empty based on pref
        chartRemainingEl.style.backgroundColor = '#ecf0f1'; // Make it less prominent until data arrives
    }
}


// --- Updated UI Update Function ---
// Function to update the summary statistics UI incrementally
function updateSummaryUI() {
    totalTilesProcessedEl.textContent = tilesProcessedCount; // Display processed tile count
    totalViolationsEl.textContent = totalViolations;
    totalFixedEl.textContent = totalFixedCount;
    attrFixedEl.textContent = attrFixedCount;
    segmentFixedEl.textContent = segmentFixedCount;

    // Calculate and display percentages
    const totalFixPercent = totalViolations > 0 ? ((totalFixedCount / totalViolations) * 100).toFixed(1) : 0;
    const attrFixPercent = totalViolations > 0 ? ((attrFixedCount / totalViolations) * 100).toFixed(1) : 0;
    const segmentFixPercent = totalViolations > 0 ? ((segmentFixedCount / totalViolations) * 100).toFixed(1) : 0;

    fixPercentageEl.textContent = `${totalFixPercent}%`;
    attrPercentageEl.textContent = `${attrFixPercent}%`;
    segmentPercentageEl.textContent = `${segmentFixPercent}%`;

    // Update the violations chart
    updateViolationsChart(totalViolations, attrFixedCount, segmentFixedCount);

    // Make summary visible if it's not already
    if (summaryEl.style.display === 'none' && (tilesProcessedCount > 0 || totalViolations > 0) ) {
         summaryEl.style.display = 'block';
         chartRemainingEl.style.backgroundColor = '#e74c3c'; // Set correct color once data starts showing
    }
}


// Start processing (keep most existing logic, update SSE handler)
startBtn.addEventListener('click', async () => {
    try {
        // Reset UI elements (keep existing)
        startBtn.disabled = true;
        statusEl.textContent = 'Starting processing...';
        tileGridEl.innerHTML = '';
        logsEl.innerHTML = '';
        rawJsonEl.innerHTML = '';
        summaryEl.style.display = 'none'; // Hide summary initially
        correctionsDetailsEl.style.display = 'none';
        attrCorrectionsTableEl.innerHTML = '';
        segmentCorrectionsTableEl.innerHTML = '';
        progressBar.style.width = '0%'; // Reset progress bar
        progressText.textContent = '0%';

        // --- Reset counters and data ---
        totalFixedCount = 0;
        attrFixedCount = 0;
        segmentFixedCount = 0;
        totalViolations = 0;
        tilesProcessedCount = 0; // Reset processed tile count
        tileStatuses = {};
        allJsonMessages = [];
        attrCorrections = [];
        segmentCorrections = [];
        // Reset summary UI text explicitly
        updateSummaryUI();


        // Start the task (keep existing)
        addLog('Starting processing task...');
        const response = await fetch('/start-processing', { /* ... */ });
        const data = await response.json();
        const taskId = data.task_id;
        addLog(`Task started with ID: ${taskId}`);

        // Connect to the event stream (keep existing)
        const eventSource = new EventSource(`/task-progress/${taskId}`);

        // --- Updated SSE onmessage Handler ---
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);

            // Store raw JSON for developer view (keep existing)
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
                     // Optionally make summary visible early, or wait for first processing update
                    // summaryEl.style.display = 'block';
                    break;

                case 'processing':
                    // Update progress bar (keep existing)
                    if (data.progress_percent !== undefined) {
                        progressBar.style.width = `${data.progress_percent}%`;
                        progressText.textContent = `${data.progress_percent}%`;
                    }

                    // Update current tile info (keep existing)
                    if (data.current_tile) {
                        currentTileInfo.style.display = 'flex';
                        currentTileEl.textContent = data.current_tile;
                        tileProgressEl.textContent = `Tile ${data.current_tile_index} of ${data.total_tiles}`;
                        // Update tile grid status
                         // Ensure tiles array is available for updateTileGrid
                        const tilesForGrid = data.tiles || Object.keys(tileStatuses).length > 0 ? Object.keys(tileStatuses) : [];
                        if(tilesForGrid.length === 0 && allJsonMessages.length > 0 && allJsonMessages[0].tiles) {
                            // Fallback to get tiles from the first message if needed
                             updateTileGrid(allJsonMessages[0].tiles, data.current_tile_index);
                        } else if (tilesForGrid.length > 0) {
                            updateTileGrid(tilesForGrid, data.current_tile_index);
                        }
                    }

                    // Update current step (keep existing)
                    if (data.current_step) {
                        currentStepEl.textContent = data.current_step;
                    }

                    // Add log message (keep existing)
                    addLog(data.message);

                    // --- Incremental Summary Update ---
                    // Check if this message contains counts for a completed tile
                    // (We'll use the presence of road_segment_fixed_count as the signal
                    // because it's sent after both steps for a tile are done)
                    if (data.road_segment_fixed_count !== undefined) {
                        // Increment running totals with the counts *for this specific tile*
                        totalViolations += data.total_violations || 0;
                        attrFixedCount += data.road_attr_fixed_count || 0;
                        segmentFixedCount += data.road_segment_fixed_count || 0;
                        totalFixedCount = attrFixedCount + segmentFixedCount; // Recalculate total fixed
                        tilesProcessedCount++; // Increment fully processed tile count

                        // Update the summary UI elements
                        updateSummaryUI();
                    }
                    // --- End Incremental Summary Update ---


                    // Collect detailed corrections (keep existing)
                    if (data.road_attr_result && data.road_attr_result.data && Object.keys(data.road_attr_result.data).length > 0) {
                        attrCorrections.push(data.road_attr_result);
                        updateCorrectionsDetails(); // Update details table
                    }
                    if (data.road_segment_result && data.road_segment_result.data && Object.keys(data.road_segment_result.data).length > 0) {
                        segmentCorrections.push(data.road_segment_result);
                        updateCorrectionsDetails(); // Update details table
                    }
                    break;

                case 'error':
                    addLog(`ERROR: ${data.message}`, 'error');
                    if (data.current_tile) {
                        tileStatuses[data.current_tile] = 'error'; // Mark tile status
                        const tileEl = document.getElementById(`tile-${data.current_tile}`);
                        if (tileEl) {
                            tileEl.className = 'tile-box error';
                        }
                         // Increment processed count even on error to keep progress moving
                         tilesProcessedCount++;
                         // Update summary to reflect processed tile count change
                         updateSummaryUI();
                    }
                     // Update overall progress bar slightly to show progression despite error
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
                    currentTileInfo.style.display = 'none'; // Hide current tile info

                    // Update final counts from the 'completed' message data
                    totalViolations = data.total_violations_processed;
                    attrFixedCount = data.total_attr_fixed;
                    segmentFixedCount = data.total_segment_fixed;
                    totalFixedCount = data.total_fixed;
                    tilesProcessedCount = data.total_tiles; // Set to total tiles

                    // Update UI with final numbers
                    updateSummaryUI();

                    // Make sure all tiles are marked completed (except errors)
                    if (allJsonMessages[0] && allJsonMessages[0].tiles) {
                         allJsonMessages[0].tiles.forEach(tile => {
                            if (tileStatuses[tile] !== 'error') {
                                const tileEl = document.getElementById(`tile-${tile}`);
                                if(tileEl) tileEl.className = 'tile-box completed';
                            }
                         });
                    }


                    startBtn.disabled = false;
                    eventSource.close();
                    addLog('Processing finished.', 'success');
                    break;
            }
        };

        // SSE Error Handler (keep existing)
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
