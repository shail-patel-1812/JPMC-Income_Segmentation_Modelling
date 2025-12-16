# General utilities
import warnings
import csv

warnings.filterwarnings('ignore')

def convert_to_csv(data_file, columns_file, output_file, delimiter=None):
    """
    Convert .data and .columns files to CSV.
    
    Args:
        data_file: Path to the .data file
        columns_file: Path to the .columns file
        output_file: Path for the output CSV file
        delimiter: Delimiter used in .data file (auto-detects if None)
    """
    # Read column names from .columns file
    with open(columns_file, 'r') as f:
        # Handle different formats: one per line or comma-separated
        content = f.read().strip()
        if '\n' in content:
            columns = [col.strip() for col in content.split('\n')]
        else:
            columns = [col.strip() for col in content.split(',')]
    
    print(f"Found {len(columns)} columns: {columns}")
    
    # Read data from .data file
    with open(data_file, 'r') as f:
        lines = f.readlines()
    
    # Auto-detect delimiter if not specified
    if delimiter is None:
        first_line = lines[0] if lines else ""
        if '\t' in first_line:
            delimiter = '\t'
        elif ',' in first_line:
            delimiter = ','
        elif ';' in first_line:
            delimiter = ';'
        else:
            delimiter = None  # whitespace
    
    # Parse data rows
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if delimiter:
            values = line.split(delimiter)
        else:
            values = line.split()  # split on whitespace
        rows.append(values)
    
    print(f"Found {len(rows)} data rows")
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    
    print(f"Successfully created: {output_file}")


# === USAGE ===
if __name__ == "__main__":
    # Update these paths to match your files
    data_file = r"census-bureau.data"
    columns_file = r"census-bureau.columns"
    output_file = "output.csv"
    
    convert_to_csv(data_file, columns_file, output_file)