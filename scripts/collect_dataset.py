import os
import shutil
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("codellama/CodeLlama-7B-hf")
min_tokens = 1000

python_files = []
total_files = 0
processed_files = 0

# Create data/train directory if it doesn't exist
output_dir = "data/train"
os.makedirs(output_dir, exist_ok=True)

for root, dirs, files in os.walk("cloned_repo"):
    for file in files:
        total_files += 1
        if file.endswith(".py"):
            processed_files += 1
            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                try:
                    code = f.read()
                    tokens = tokenizer.encode(code).tokens
                    if len(tokens) >= min_tokens:
                        # Copy the file to data/train directory
                        dest_path = os.path.join(output_dir, file)
                        # Handle duplicate filenames
                        if os.path.exists(dest_path):
                            base, ext = os.path.splitext(file)
                            i = 1
                            while os.path.exists(os.path.join(output_dir, f"{base}_{i}{ext}")):
                                i += 1
                            dest_path = os.path.join(output_dir, f"{base}_{i}{ext}")
                        
                        shutil.copy(file_path, dest_path)
                        python_files.append(file_path)
                        print(f"Copied {file_path} to {dest_path}")
                        
                        if len(python_files) == 100:
                            break
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

# Generate the report
report_path = "dataset_report.txt"
with open(report_path, "w") as report_file:
    report_file.write("Dataset Collection Report\n")
    report_file.write("=========================\n")
    report_file.write(f"Total files scanned: {total_files}\n")
    report_file.write(f"Python files processed: {processed_files}\n")
    report_file.write(f"Python files with >= {min_tokens} tokens: {len(python_files)}\n")
    report_file.write(f"Files copied to {output_dir}:\n")
    for file in python_files:
        report_file.write(f"- {file}\n")

print(f"Report generated: {report_path}")
print(f"Copied {len(python_files)} Python files to {output_dir}")
