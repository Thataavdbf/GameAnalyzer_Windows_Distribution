import os
import ast

def scan_codebase(root_dir):
    print(f"\n📂 Scanning: {root_dir}\n")
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(subdir, file)
                print(f"🧠 File: {file_path}")
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        tree = ast.parse(f.read())
                        for node in tree.body:
                            if isinstance(node, ast.ClassDef):
                                print(f"  🧱 Class: {node.name}")
                            elif isinstance(node, ast.FunctionDef):
                                print(f"  ⚙️  Function: {node.name}")
                    except Exception as e:
                        print(f"  ⚠️  Could not parse: {e}")
                print()

if __name__ == "__main__":
    scan_codebase(".")
