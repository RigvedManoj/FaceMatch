#!/bin/bash

export PYTHONPATH=$(pwd)/..

# Start the Python server in the background
python ../src/facematch/face_match_server.py &
server_pid=$!
echo "Server started with PID $server_pid"

# Wait for the server to start (adjust time as necessary)
sleep 10

# Start timer
start_time=$(date +%s)

# Define the directory containing the files and the output CSV file
input_directory="\\path\\to\\input\\directory" # Path to directory of query images
output_csv="\\path\\to\\output\\csv" # Path to csv file containing top-n matches
similarity_threshold = 0 #Change this value appropriately if needed
# Sample directory path
# input_directory = "<path to dataset folder>\\LFWdataset\\sample_queries"
# output_csv = "<path to dataset folder>\\LFWdataset\\output.csv"

# Initialize the CSV file with headers
echo "filename,result" > "$output_csv"

# Iterate over each file in the directory
for file in "$input_directory"/*; do
    # Ensure it's a file (not a directory)
    if [[ -f "$file" ]]; then
        # Get the filename
        filename=$(basename "$file")

        # Call the Python script and capture its output
        result=$(python ../src/Sample_Client/sample_find_face_client.py --file_paths "$file" --similarity_threshold $similarity_threshold --database_name "sample_db" | grep -v "Matches found" | tr '\n' ' ')

        # Append the filename and result as a new row in the CSV file
        echo "$filename,$result" >> "$output_csv"
    fi
done

# Calculate total time taken
end_time=$(date +%s)
total_time=$((end_time - start_time))

# Stop the server
kill $server_pid
echo "Server stopped"

# Print total time taken
echo "Total time taken: $total_time seconds"

read -p "Press any key to exit..."