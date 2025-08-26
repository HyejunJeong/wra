#!/usr/bin/env python3
"""
Finalize_Data.py - Replace domain sequences in Evaluation and Training CSV files
with corresponding timing information from Process_raw_data directory.

This script:
1. Finds all CSV files in Evaluation and Training directories
2. Matches them with corresponding processed CSV files in Process_raw_data
3. Replaces the domain_sequence column with timing information
4. Saves updated files with "_finalized" suffix
"""

import os
import csv
import glob
import re
from pathlib import Path

def extract_dataset_name(filename):
    """Extract dataset name from filename to match with processed files."""
    # Remove common prefixes and suffixes
    name = filename.replace("Evaluation Set - ", "").replace("Training Set - ", "")
    name = name.replace(".csv", "")
    
    # Extract the core dataset identifier (e.g., SESSION14-GR-local -> session14_local)
    if "SESSION14" in name:
        if "local" in name:
            return "session14_local"
        elif "GPT4" in name:
            return "session14_gpt4"
    elif "FEDWEB13" in name:
        if "local" in name:
            return "fedweb13_local"
        elif "GPT4" in name:
            return "fedweb13_gpt4"
    elif "DD16" in name:
        if "local" in name:
            return "dd16_local"
        elif "GPT4" in name:
            return "dd16_gpt4"
    
    return None

def find_processed_file(dataset_name, process_raw_data_dir):
    """Find the corresponding processed CSV file."""
    if not dataset_name:
        return None
    
    # Look for files with the pattern: {dataset_name}_trace_processed.csv
    pattern = f"{dataset_name}_trace_processed.csv"
    processed_files = glob.glob(os.path.join(process_raw_data_dir, pattern))
    
    if processed_files:
        return processed_files[0]
    
    return None

def load_processed_data(processed_file_path):
    """Load the processed CSV data into a dictionary mapping idx to domain_sequence."""
    processed_data = {}
    
    try:
        with open(processed_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row['idx'])
                domain_sequence = row['domain_sequence']
                processed_data[idx] = domain_sequence
        
        print(f"  Loaded {len(processed_data)} rows from {os.path.basename(processed_file_path)}")
        return processed_data
    
    except Exception as e:
        print(f"  Error loading {processed_file_path}: {e}")
        return {}

def update_csv_file(input_file_path, processed_data, output_file_path):
    """Update the CSV file by replacing domain_sequence column with processed data."""
    updated_rows = 0
    total_rows = 0
    skipped_rows = 0
    
    try:
        # Read the input file
        with open(input_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            # Write the output file
            with open(output_file_path, 'w', newline='', encoding='utf-8') as out_f:
                writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                writer.writeheader()
                
                for row in reader:
                    total_rows += 1
                    
                    # Handle empty or invalid Idx values
                    try:
                        if not row['Idx'] or row['Idx'].strip() == '':
                            # Skip rows with empty Idx
                            skipped_rows += 1
                            writer.writerow(row)
                            continue
                        
                        idx = int(row['Idx'])
                        
                        # Replace domain_sequence if we have processed data for this idx
                        if idx in processed_data:
                            row['domain_sequence'] = processed_data[idx]
                            updated_rows += 1
                        
                        writer.writerow(row)
                        
                    except (ValueError, KeyError) as e:
                        # Skip rows with invalid Idx values
                        skipped_rows += 1
                        writer.writerow(row)
                        continue
        
        print(f"  Updated {updated_rows}/{total_rows} rows (skipped {skipped_rows} invalid rows)")
        return updated_rows, total_rows
    
    except Exception as e:
        print(f"  Error updating {input_file_path}: {e}")
        return 0, 0

def process_directory_files(directory_path, process_raw_data_dir, output_dir):
    """Process all CSV files in a directory."""
    print(f"\nProcessing directory: {directory_path}")
    
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    if not csv_files:
        print(f"  No CSV files found in {directory_path}")
        return
    
    print(f"  Found {len(csv_files)} CSV files")
    
    total_updated = 0
    total_files = 0
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        print(f"\n  Processing: {filename}")
        
        # Extract dataset name
        dataset_name = extract_dataset_name(filename)
        if not dataset_name:
            print(f"    Could not extract dataset name from {filename}")
            continue
        
        print(f"    Dataset name: {dataset_name}")
        
        # Find corresponding processed file
        processed_file = find_processed_file(dataset_name, process_raw_data_dir)
        if not processed_file:
            print(f"    No processed file found for {dataset_name}")
            continue
        
        print(f"    Found processed file: {os.path.basename(processed_file)}")
        
        # Load processed data
        processed_data = load_processed_data(processed_file)
        if not processed_data:
            print(f"    No processed data loaded")
            continue
        
        # Create output filename
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}_finalized.csv"
        output_path = os.path.join(output_dir, output_filename)
        
        # Update the CSV file
        updated, total = update_csv_file(csv_file, processed_data, output_path)
        
        if updated > 0:
            total_updated += updated
            total_files += 1
            print(f"    Successfully created: {output_filename}")
        else:
            print(f"    Failed to update file")
    
    return total_files, total_updated

def main():
    """Main function to process all directories."""
    print("Finalize_Data.py - Domain Sequence Replacement Tool")
    print("=" * 60)
    
    # Define directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    evaluation_dir = os.path.join(base_dir, "Evaluation")
    training_dir = os.path.join(base_dir, "Training")
    process_raw_data_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create output directory
    output_dir = os.path.join(base_dir, "Finalized_Data")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Base directory: {base_dir}")
    print(f"Evaluation directory: {evaluation_dir}")
    print(f"Training directory: {training_dir}")
    print(f"Process raw data directory: {process_raw_data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if directories exist
    if not os.path.exists(evaluation_dir):
        print(f"Error: Evaluation directory not found: {evaluation_dir}")
        return
    
    if not os.path.exists(training_dir):
        print(f"Error: Training directory not found: {training_dir}")
        return
    
    if not os.path.exists(process_raw_data_dir):
        print(f"Error: Process raw data directory not found: {process_raw_data_dir}")
        return
    
    # Process Evaluation directory
    eval_files, eval_updated = process_directory_files(evaluation_dir, process_raw_data_dir, output_dir)
    
    # Process Training directory
    train_files, train_updated = process_directory_files(training_dir, process_raw_data_dir, output_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Evaluation files processed: {eval_files}")
    print(f"Training files processed: {train_files}")
    print(f"Total files processed: {eval_files + train_files}")
    print(f"Total rows updated: {eval_updated + train_updated}")
    print(f"Output directory: {output_dir}")
    
    if eval_files + train_files > 0:
        print("\n✅ Processing completed successfully!")
        print(f"Check the '{output_dir}' directory for finalized files.")
    else:
        print("\n❌ No files were processed.")

if __name__ == "__main__":
    main()
