#!/bin/bash
# Migration script: Clean up old logs/ directory after reorganization

echo "🧹 Cleaning up old logs/ directory..."
echo ""

cd "$(dirname "$0")/.."

# Check if logs directory exists and has files
if [ ! -d "logs" ]; then
    echo "✅ No logs/ directory found. Nothing to clean."
    exit 0
fi

file_count=$(find logs -type f | wc -l | tr -d ' ')

if [ "$file_count" -eq 0 ]; then
    echo "✅ logs/ directory is empty."
    echo "   You can safely remove it with: rm -rf logs/"
    exit 0
fi

echo "📊 Found $file_count files in logs/ directory"
echo ""
echo "Files breakdown:"
echo "  .zip files: $(find logs -name "*.zip" | wc -l | tr -d ' ')"
echo "  .pkl files: $(find logs -name "*.pkl" | wc -l | tr -d ' ')"
echo "  .csv files: $(find logs -name "*.csv*" | wc -l | tr -d ' ')"
echo "  .log files: $(find logs -name "*.log" | wc -l | tr -d ' ')"
echo "  .txt files: $(find logs -name "*.txt" | wc -l | tr -d ' ')"
echo ""

echo "⚠️  These files should have been migrated to data/ directory."
echo "   If you've verified the migration, you can remove logs/ directory."
echo ""
echo "Options:"
echo "  1. Keep logs/ directory (do nothing)"
echo "  2. Archive logs/ to logs_backup_$(date +%Y%m%d).tar.gz"
echo "  3. Delete logs/ directory permanently"
echo ""

read -p "Choose option (1/2/3): " choice

case $choice in
    1)
        echo "✅ Keeping logs/ directory"
        ;;
    2)
        archive_name="logs_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
        echo "📦 Creating archive: $archive_name"
        tar -czf "$archive_name" logs/
        if [ $? -eq 0 ]; then
            echo "✅ Archive created successfully"
            echo "   You can now safely delete logs/ with: rm -rf logs/"
        else
            echo "❌ Failed to create archive"
        fi
        ;;
    3)
        echo "⚠️  This will permanently delete logs/ directory!"
        read -p "Are you sure? Type 'yes' to confirm: " confirm
        if [ "$confirm" = "yes" ]; then
            rm -rf logs/
            echo "✅ logs/ directory deleted"
        else
            echo "❌ Cancelled"
        fi
        ;;
    *)
        echo "❌ Invalid option"
        ;;
esac

echo ""
echo "📁 Current data/ structure:"
du -sh data/* 2>/dev/null || echo "  (data/ directory not found)"
