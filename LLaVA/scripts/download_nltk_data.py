#!/usr/bin/env python3
"""Download NLTK corpora required by eval_scripts/eval_utils/chair.py."""
import nltk

for resource in ("wordnet", "omw-1.4"):
    nltk.download(resource, quiet=True)
print("NLTK data ready: wordnet, omw-1.4")
