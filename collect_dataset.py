import os
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("codellama/CodeLlama-7B-hf")
min_tokens = 1000

python_files = []
for root, dirs, files in os.walk("cloned_repo"):
    for file in files:
        if file.endswith(".py"):
            with open(os.path.join(root, file), "r") as f:
                code = f.read()
                tokens = tokenizer.encode(code).tokens
                if len(tokens) >= min_tokens:
                    python_files.append(file)
                    if len(python_files) == 100:
                        break
