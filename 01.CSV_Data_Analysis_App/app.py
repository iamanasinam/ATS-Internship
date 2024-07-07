import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# Function to load the CSV file
def load_data():
    return pd.read_csv("Data.csv")


# Function to display column names
def display_column_names(df):
    st.write("### Column Names")
    # st.write(df.columns.tolist())
    st.write(df.columns)


# Function to display row by index
def display_row_by_index(df):
    st.write("### View Row by Index")
    row_index = st.number_input(
        "Enter the index value to get the row",
        min_value=0,
        max_value=len(df) - 1,
        step=1,
    )

    if st.button("Show Row"):
        if row_index < len(df):
            st.write(df.iloc[row_index])
        else:
            st.error("Invalid row index!")


# Function to display statistics for a column
def display_statistics(df):
    st.write("### Column Statistics")
    column = st.selectbox("Select a Column for Statistics", df.columns)

    if st.button("Show Statistics"):
        try:
            if pd.api.types.is_numeric_dtype(df[column]):
                st.write(f"Minimum Value: {df[column].min()}")
                st.write(f"Maximum Value: {df[column].max()}")
                st.write(f"Mean Value: {df[column].mean()}")
                st.write(f"Mode Value: {df[column].mode().values[0]}")
                st.write(f"Median Value: {df[column].median()}")
            else:
                # Handle columns with '$' values
                if df[column].dtype == "object":
                    numeric_values = (
                        df[column]
                        .str.replace("$", "", regex=False)
                        .str.replace(",", "", regex=False)
                        .apply(pd.to_numeric, errors="coerce")
                    )
                    if not numeric_values.dropna().empty:
                        st.write(f"Minimum Value: {numeric_values.min()}")
                        st.write(f"Maximum Value: {numeric_values.max()}")
                        st.write(f"Mean Value: {numeric_values.mean()}")
                        st.write(f"Median Value: {numeric_values.median()}")
                        st.write(f"Mode Value: {numeric_values.mode().values[0]}")
                    else:
                        st.warning("No numeric values found.")
                else:
                    st.warning(
                        "Selected column is not numeric or does not contain valid numeric values."
                    )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


# Function to plot columns
def plot_columns(df):
    st.write("### Plot Columns")
    column1 = st.selectbox("Select Column for X-axis", df.columns)
    column2 = st.selectbox("Select Column for Y-axis", df.columns)

    if st.button("Plot"):
        try:
            df.plot(x=column1, y=column2, kind="scatter")
            plt.title(f"Scatter Plot of {column1} vs. {column2}")
            plt.xlabel(column1)
            plt.ylabel(column2)
            st.pyplot()
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


# Main function to run the Streamlit app
def main():
    st.title("CSV Data Analysis App")

    df = load_data()

    display_column_names(df)
    # display_index_values(df)
    display_row_by_index(df)
    display_statistics(df)
    plot_columns(df)

    if st.button("Exit"):
        st.write("Exiting the program.")
        st.stop()


if __name__ == "__main__":
    main()
