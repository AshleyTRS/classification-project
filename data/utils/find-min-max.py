import csv
import os

def find_min_max(file_path):
    """
    Computes the minimum and maximum value for each column in the CSV file.
    Time complexity: O(n log x) where n is number of columns and x is number of rows.
    """
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Get column names
        columns = [[] for _ in header]  # List of lists for each column
        
        for row in reader:
            for i, value in enumerate(row):
                try:
                    columns[i].append(float(value))
                except ValueError:
                    # Skip non-numeric values
                    pass
        
        for i, col_values in enumerate(columns):
            if col_values:
                sorted_values = sorted(col_values)
                min_val = sorted_values[0]
                max_val = sorted_values[-1]
                print(f"{header[i]}, Min: {min_val}, Max: {max_val}")
            else:
                print(f"{header[i]}, No numeric values")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, '..', 'raw', 'X.csv')
    find_min_max(file_path)

if __name__ == "__main__":
    main()