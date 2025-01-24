#!/bin/bash
#!/bin/bash

# Install necessary Python packages
pip install -r requirements.txt

# Function to recursively unzip files
unzip_recursive() {
    local dir=$1
    for file in "$dir"/*.gz; do
        if [[ -f "$file" ]]; then
            gunzip "$file"
            #unzip_recursive "${file%.*}"
        fi
    done
}

# Define the base directory
BASE_DIR="Task01_BrainTumour"

# Unzip files in the specified folders
for folder in "imagesTr" "imagesTs" "labelsTr"; do
    unzip_recursive "$BASE_DIR/$folder"
done