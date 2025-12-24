#!/bin/bash
# Clear Streamlit cache and restart fresh

echo "🧹 Cleaning Streamlit cache..."
rm -rf ~/.streamlit/cache 2>/dev/null
rm -rf .streamlit/cache 2>/dev/null

echo "🔄 Restarting Dashboard..."
echo "Press Ctrl+C in the terminal running 'streamlit run dashboard.py'"
echo "Then run: streamlit run dashboard.py --server.runOnSave true"
echo ""
echo "Or just refresh your browser with Cmd+Shift+R (hard reload)"
