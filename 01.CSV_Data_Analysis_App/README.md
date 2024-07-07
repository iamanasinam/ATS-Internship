# CSV Data Analysis App

This repository contains a Python script and a Streamlit application for analyzing data from a CSV file named `try.csv`. The script allows for various operations including displaying column names, inspecting rows by index, generating statistical summaries for columns, and plotting data. The Streamlit application offers a web interface for interacting with the CSV data.

### Requirements

- Python 3.x
- Pandas
- Matplotlib
- Colorama
- Streamlit

You can install the required packages using pip:

``` bash
pip install pandas matplotlib colorama streamlit
```
### Usage

#### Python Script

The Python script provided in this repository performs the following operations:

**1. Read the CSV File**: Loads data from the file `try.csv` located in the same directory as the script.

**2. Display Column Names**: Outputs the names of all columns present in the CSV file.

**3. Display Index Values**: Lists the index values for each column in the DataFrame.

**4. Row Inspection**: Prompts the user to input an index value and displays the row data corresponding to that index. The user can repeatedly inspect rows by index until they choose to stop.

**5. Display Column Statistics**: Calculates and displays statistical values for a selected column, including:
- **Minimum Value**: The lowest value in the column.
- **Maximum Value**: The highest value in the column.
- **Mean Value**: The average value of the column.
- **Mode Value**: The most frequently occurring value in the column.
- **Median Value**: The middle value of the column when sorted. Handles both numeric columns and columns containing monetary values (formatted as strings).

**6. Plot Columns**: Allows the user to plot data from one column against another column or the index .Creates a scatter plot with the specified columns.

**7. Continue or Exit**: Prompts the user to decide whether to continue using the program or exit. If the user chooses to continue, they can perform additional operations; otherwise, the program will terminate.

### Running the Script

To run the script, use the following command in your terminal or command prompt:

```bash
python index.py
```
### Streamlit Application

The Streamlit app provides a web-based interface for the same operations performed by the Python script:

**1. Display Column Names**: Lists all column names in the CSV file.

**2. View Row by Index**: Allows you to select a column and view data for a specific row index.

**3. Display Column Statistics**: Shows statistical summaries (e.g., minimum, maximum, mean, median, mode) for a selected column.


**4. Plot Columns**: Enables plotting one column against another column or the index.

**5. Exit**: Provides an option to stop the application.

### Running the Streamlit App
To run the Streamlit application, use the following command:

``` bash
streamlit run app.py
```


## Code Overview

### Python Script

The Python script includes:

- **Data Loading and Initial Inspection**: Loads data from `try.csv` and performs an initial check.
- **Functions**:
  - Display column names.
  - Show index values for each column.
  - View data for a specific row by index.
  - Display statistical summaries for selected columns.
  - Plot data from one column against another or the index.
- **User Interaction**: Allows users to choose columns and indices through command-line input.
- **Statistics Display**: Provides statistics for both numeric and non-numeric columns.

### Streamlit Application

The Streamlit app includes:

- **Functions**:
  - Display column names.
  - View rows by index.
  - Display statistical summaries for selected columns.
  - Plot data from one column against another.
- **Interactive Widgets**: Provides a web-based interface for user input and displaying results.


### Example
Here's an example of how you might interact with the script:

``` bash
Requirement 1 - Column Names:
Column1
Column2
...

Requirement 2 - Index Values:
Index Values for Column: Column1 - Range of index values
...

Do you want to find See row by index? (yes/no): yes
Enter the index value to get the row: 5
Value at Row 5 - Data for row 5

Enter a column name: Column1
Statistics for Column: Column1
Minimum Value: 10
Maximum Value: 100
Mean Value: 55
...

Enter a column name for X-axis: Column1
Enter a column name for Y-axis: Column2

```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## Contact
For any inquiries or issues, please contact me at iamanasinam@gmail.com
