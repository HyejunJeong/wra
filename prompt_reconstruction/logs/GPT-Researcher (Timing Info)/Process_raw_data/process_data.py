#!/usr/bin/env python3
"""
Process all session trace txt files in the directory to create CSV files with domain sequences and timing information.
Each log entry gets a row with idx (1-N) and domain_sequence showing domains with relative timing.
"""

import csv
import re
import os
import glob
from datetime import datetime
from collections import defaultdict

def parse_timestamp(timestamp_str):
    """Parse timestamp string to datetime object."""
    return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')

def calculate_time_diff(start_time, current_time):
    """Calculate time difference in seconds from start time."""
    diff = current_time - start_time
    return diff.total_seconds()

def process_trace_file(input_file, output_file):
    """Process the trace file and create CSV output."""
    
    # Dictionary to store data for each log
    logs_data = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Parse line: log_XXX | timestamp | domain | URL
            parts = line.split(' | ')
            if len(parts) != 4:
                continue
                
            log_id = parts[0]
            timestamp_str = parts[1]
            domain = parts[2]
            url = parts[3]
            
            try:
                timestamp = parse_timestamp(timestamp_str)
                logs_data[log_id].append((timestamp, domain))
            except ValueError as e:
                print(f"Warning: Could not parse timestamp '{timestamp_str}' in {input_file}: {e}")
                continue
    
    # Process each log and create CSV rows
    csv_rows = []
    
    for log_id in sorted(logs_data.keys()):
        # Extract log number (001, 002, etc.)
        log_num = int(log_id.split('_')[1])
        
        # Sort entries by timestamp
        entries = sorted(logs_data[log_id], key=lambda x: x[0])
        
        if not entries:
            continue
            
        # First timestamp is the reference (0.0)
        start_time = entries[0][0]
        
        # Build domain sequence with timing
        domain_sequence = []
        for timestamp, domain in entries:
            time_diff = calculate_time_diff(start_time, timestamp)
            domain_sequence.append(f"{domain}({time_diff:.3f})")
        
        # Join all domains into one cell
        domain_sequence_str = ", ".join(domain_sequence)
        
        csv_rows.append([log_num, domain_sequence_str])
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['idx', 'domain_sequence'])
        writer.writerows(csv_rows)
    
    print(f"Processed {len(csv_rows)} logs and saved to {output_file}")
    return len(csv_rows)

def find_trace_files():
    """Find all txt files in the current directory that look like trace files."""
    # Look for files that contain 'trace' in the name or have the expected format
    txt_files = glob.glob("*.txt")
    trace_files = []
    
    for txt_file in txt_files:
        # Check if it's likely a trace file by looking at the first few lines
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line and 'log_' in first_line and ' | ' in first_line:
                    trace_files.append(txt_file)
        except Exception as e:
            print(f"Warning: Could not read {txt_file}: {e}")
            continue
    
    return trace_files

def main():
    """Process all trace files in the directory."""
    
    # Find all trace files
    trace_files = find_trace_files()
    
    if not trace_files:
        print("No trace files found in the current directory!")
        print("Make sure you have .txt files with the expected format (log_XXX | timestamp | domain | URL)")
        return
    
    print(f"Found {len(trace_files)} trace files to process:")
    for file in trace_files:
        print(f"  - {file}")
    print()
    
    total_logs = 0
    
    # Process each trace file
    for trace_file in trace_files:
        print(f"Processing {trace_file}...")
        
        # Create output filename
        base_name = os.path.splitext(trace_file)[0]
        output_file = f"{base_name}_processed.csv"
        
        try:
            num_logs = process_trace_file(trace_file, output_file)
            total_logs += num_logs
            print(f"✓ Completed {trace_file} -> {output_file} ({num_logs} logs)")
        except Exception as e:
            print(f"✗ Error processing {trace_file}: {e}")
        
        print()
    
    print(f"Processing complete! Total logs processed: {total_logs}")
    print(f"Generated CSV files:")
    
    # List all generated CSV files
    csv_files = glob.glob("*_processed.csv")
    for csv_file in sorted(csv_files):
        file_size = os.path.getsize(csv_file)
        print(f"  - {csv_file} ({file_size:,} bytes)")

if __name__ == "__main__":
    main()
