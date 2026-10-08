"""Lets the tests import the project modules the same way src/pipeline.py does."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("", "audio", "text", "fusion"):
    sys.path.insert(0, os.path.join(ROOT, "src", sub))