import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore, init

init()

# Reading the csv file
df = pd.read_csv("Data.csv")

# Detects and displays column names - use for loop to display column names
print(Fore.YELLOW + "Requirement 1 - Column Names:", Fore.GREEN)
for column in df.columns:
    print(column)

# Detects and displays index values for each column
print(Fore.YELLOW + "\nRequirement 2 - Index Values: ")
for column in df.columns:
    print(
        Fore.YELLOW + "Index Values for Column:",
        Fore.GREEN + "",
        column,
        Fore.YELLOW + " - ",
        Fore.GREEN + "",
        df[column].index,
    )

for column in df.columns:
    while True:
        user_input = input(
            Fore.YELLOW
            + f"\nDo you want to find See row by index? (yes/no): {Fore.GREEN}",
        )
        if user_input.lower() == "yes":
            user_input = input(
                Fore.YELLOW + f"Enter the index value to get the row: {Fore.GREEN} "
            )
            if user_input.isdigit():
                row_index = int(user_input)
                if row_index < len(df):
                    print(
                        Fore.YELLOW + "Value at Row",
                        Fore.GREEN + "",
                        user_input,
                        Fore.YELLOW + " - \n",
                        Fore.GREEN + "",
                        df.iloc[row_index, :],
                    )
                else:
                    print(Fore.RED + "Invalid row index!")
            else:
                print(Fore.RED + "Invalid index value! Please enter a number.")
        elif user_input.lower() == "no":
            break
        else:
            print(Fore.RED + "Invalid input! Please enter 'yes' or 'no'.")
    break


# Allows the user to display statistical values for any selected column
def display_stats(column):
    try:
        print(Fore.YELLOW + "\nStatistics for Column: " + Fore.GREEN + str(column))

        # Check if the column is numeric or not
        if pd.api.types.is_numeric_dtype(df[column]):
            if column in [
                "Discounts",
                "Units Sold",
            ]:
                print(
                    Fore.YELLOW + "Minimum Value:",
                    Fore.GREEN + str(df[column].min()),
                    Fore.YELLOW + "and the name of Product:",
                    Fore.GREEN
                    + df[df[column] == df[column].min()]["Product"].values[0],
                )
                print(
                    Fore.YELLOW + "Maximum Value:",
                    Fore.GREEN + str(df[column].max()),
                    Fore.YELLOW + "and the name of Product:",
                    Fore.GREEN
                    + df[df[column] == df[column].max()]["Product"].values[0],
                )
                print(Fore.YELLOW + "Mean Value:", Fore.GREEN + str(df[column].mean()))
                print(
                    Fore.YELLOW + "Mode Value:",
                    Fore.GREEN + str(df[column].mode().values[0]),
                )
                print(
                    Fore.YELLOW + "Median Value:", Fore.GREEN + str(df[column].median())
                )
        elif column in [
            "Manufacturing Price",
            "Sale Price",
            "Gross Sales",
            "Sales",
            "COGS",
            "Profit",
        ]:
            # Check if the column contains string values with '$' and remove '$' before converting to numeric
            if df[column].dtype == "object":
                numeric_values = (
                    df[column]
                    .str.replace("$", "")
                    .str.replace(",", "")
                    .apply(pd.to_numeric, errors="coerce")
                )
                numeric_values = numeric_values.dropna()
                if len(numeric_values) > 0:
                    print(
                        Fore.YELLOW + "Minimum Value:",
                        Fore.GREEN + str(df[column].min()),
                        Fore.YELLOW + "and the name of Product:",
                        Fore.GREEN
                        + df[df[column] == df[column].min()]["Product"].values[0],
                    )
                    print(
                        Fore.YELLOW + "Maximum Value:",
                        Fore.GREEN + str(df[column].max()),
                        Fore.YELLOW + "and the name of Product:",
                        Fore.GREEN
                        + df[df[column] == df[column].max()]["Product"].values[0],
                    )
                    print(
                        Fore.YELLOW + "Mean Value:",
                        Fore.GREEN + str(numeric_values.mean()),
                    )
                    print(
                        Fore.YELLOW + "Median Value:",
                        Fore.GREEN + str(numeric_values.median()),
                    )
                    print(
                        Fore.YELLOW + "Mode Value:",
                        Fore.GREEN + str(numeric_values.mode().values[0]),
                    )
                else:
                    print(Fore.YELLOW + "No numeric values found.")
            else:
                print(Fore.RED + "Column does not contain string values.")
        else:
            print(Fore.YELLOW + "Selected column is not numeric or not available.")

    except TypeError:
        print(Fore.RED + "Selected column contains non-numeric values.")
    except Exception as e:
        print(Fore.RED + "An error occurred: ", str(e))


# Asking the user to input a column name
column = input(Fore.YELLOW + "\nEnter a column name: " + Fore.GREEN)

# Calling the function to display the statistical values
display_stats(column)


# Allows the user to plot any column against any other column or the index
def plot_column(column1, column2):
    try:
        print(
            Fore.YELLOW
            + "\nPlotting Column: "
            + Fore.GREEN
            + str(column1)
            + " vs. "
            + Fore.GREEN
            + str(column2)
        )
        df.plot(x=column1, y=column2, kind="scatter")
    except TypeError:
        print(Fore.RED + "Selected columns contain non-numeric values.")
    except Exception as e:
        print(Fore.RED + "An error occurred: ", str(e))


# Asking the user to input two column names
column1 = input(Fore.YELLOW + "\nEnter a column name for X-axis: " + Fore.GREEN)
column2 = input(Fore.YELLOW + "\nEnter a column name for Y-axis: " + Fore.GREEN)

# Calling the function to plot the column
plot_column(column1, column2)

# Show the plot
plt.show()

# After the user is done, ask if they want to continue using the program or exit
while True:
    user_input = input(
        Fore.YELLOW + "\nDo you want to continue? (yes/no): " + Fore.GREEN
    )
    if user_input.lower() == "yes":
        break
    elif user_input.lower() == "no":
        print(Fore.YELLOW + "Exiting the program.")
        break
    else:
        print(Fore.RED + "Invalid input! Please enter 'yes' or 'no'.")
