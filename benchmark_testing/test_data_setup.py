import os
import random
import shutil

# Supported image extensions
image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}


# Function to check if a file is an image
def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in image_extensions


# Main function to process directories recursively
def process_directory(root_dir, test_dir, verify_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        # Filter image files in the current directory
        image_files = [f for f in filenames if is_image_file(f)]

        # If there are multiple image files in the directory, process them
        if len(image_files) > 1:
            # Randomly select two images: one for test, one for verify
            selected_images = random.sample(image_files, 2)
            test_image, verify_image = selected_images

            # Ensure destination directories exist
            os.makedirs(test_dir, exist_ok=True)
            os.makedirs(verify_dir, exist_ok=True)

            # Move selected images to test and verify directories
            shutil.move(
                os.path.join(dirpath, test_image), os.path.join(test_dir, test_image)
            )
            shutil.move(
                os.path.join(dirpath, verify_image),
                os.path.join(verify_dir, verify_image),
            )
            print(
                f"Moved '{test_image}' to test directory and '{verify_image}' to verify directory."
            )


# Function to count files in a directory
def count_files(directory):
    return len(
        [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    )


# Specify the root directory you want to start from
root_directory = "\\path\\to\\root\\directory"
test_directory = "\\path\\to\\test\\directory"
verify_directory = "\\path\\to\\verify\\directory"
process_directory(root_directory, test_directory, verify_directory)

# Print the number of files in the test and verify directories
print(f"Number of files in test directory: {count_files(test_directory)}")
print(f"Number of files in verify directory: {count_files(verify_directory)}")