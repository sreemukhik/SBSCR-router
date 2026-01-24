"""
Simple HTTP server to run the SBSCR demo
Run this script and open http://localhost:8080 in your browser
"""

import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

# Configuration
PORT = 8080
DEMO_DIR = Path(__file__).parent

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

def run_server():
    # Change to demo directory
    os.chdir(DEMO_DIR)
    
    # Create server
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print("=" * 60)
        print("🎨 SBSCR DEMO SERVER")
        print("=" * 60)
        print(f"\n✅ Server running at: http://localhost:{PORT}")
        print(f"📁 Serving from: {DEMO_DIR}")
        print("\n📝 Instructions:")
        print("1. Make sure SBSCR server is running (python serve.py)")
        print("2. Open http://localhost:8080 in your browser")
        print("3. Start chatting to see the router in action!")
        print("\n⏹️  Press Ctrl+C to stop the server")
        print("=" * 60 + "\n")
        
        # Try to open browser automatically
        try:
            webbrowser.open(f'http://localhost:{PORT}')
            print("🌐 Opening browser automatically...\n")
        except:
            print("⚠️  Could not open browser automatically")
            print(f"   Please open http://localhost:{PORT} manually\n")
        
        # Start serving
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Server stopped. Goodbye!")

if __name__ == "__main__":
    run_server()
