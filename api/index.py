from serve import app as application
import sys
import os

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Vercel needs "app" variable
app = application
