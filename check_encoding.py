# -*- coding: utf-8 -*-

import os

def is_utf8_encoded(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read()
        return True
    except UnicodeDecodeError:
        return False
    except Exception as e:
        print(f"[×] Unexpected error reading {file_path}: {e}")
        return False

def convert_to_utf8(file_path, original_encoding='gbk'):
    try:
        with open(file_path, 'r', encoding=original_encoding, errors='ignore') as f:
            content = f.read()
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[✓] Converted to UTF-8: {file_path}")
    except Exception as e:
        print(f"[×] Failed to convert {file_path}: {e}")

def check_and_fix_encoding(root_path, convert=False):
    print(f"🔍 Scanning: {root_path}")
    count = 0
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                print(f"Checking: {full_path}")
                if not is_utf8_encoded(full_path):
                    print(f"[!] Not UTF-8 encoded: {full_path}")
                    if convert:
                        convert_to_utf8(full_path)
                    count += 1
    if count == 0:
        print("✅ All .py files are UTF-8 encoded.")

# 设置路径
project_path = '/home/hbq/genb_main'
check_and_fix_encoding(project_path, convert=True)
