#!/bin/bash
# Run the application with webkit compatibility and detailed logging

echo "Starting AirImpute Pro with webkit compatibility and logging..."

# Create log directory
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# Set up logging
LOG_FILE="$LOG_DIR/airimpute-$(date +%Y%m%d-%H%M%S).log"
ERROR_LOG="$LOG_DIR/airimpute-errors-$(date +%Y%m%d-%H%M%S).log"

echo "Logs will be written to:"
echo "  Main log: $LOG_FILE"
echo "  Error log: $ERROR_LOG"
echo ""

# Run with webkit compatibility script and capture all output
echo "Starting application..."
./tauri-dev.sh 2>&1 | tee "$LOG_FILE" | while IFS= read -r line; do
    # Also write errors to separate log
    if [[ "$line" =~ "error"|"Error"|"ERROR"|"failed"|"Failed"|"FAILED" ]]; then
        echo "$line" >> "$ERROR_LOG"
    fi
    # Show output in terminal too
    echo "$line"
done

# Check exit status
EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "Application exited with error code: $EXIT_CODE"
    echo "Check logs for details:"
    echo "  $LOG_FILE"
    echo "  $ERROR_LOG"
    
    # Show last 10 lines of error log
    if [ -s "$ERROR_LOG" ]; then
        echo ""
        echo "Recent errors:"
        tail -10 "$ERROR_LOG"
    fi
fi

exit $EXIT_CODE