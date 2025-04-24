#!/usr/bin/env python
import sys
import json

def clean_notebook(notebook):
    """清理 notebook 的輸出"""
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    return notebook

if __name__ == '__main__':
    notebook = json.load(sys.stdin)
    cleaned = clean_notebook(notebook)
    json.dump(cleaned, sys.stdout, indent=1)