import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_data():
    # Loading DataSet
    data = pd.read_excel('DataSet-fi.xlsx')
    return data

def show_histogram(data, column):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def show_scatter_plot(data, x_column, y_column):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x=x_column, y=y_column)
    plt.title(f'Scatter Plot of {x_column} vs {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()

def show_bar_plot(data, column):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=column, data=data)
    plt.title(f'Bar Plot of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()

def main():
    print("Welcome to the EndoResearch Final Project!")
    
    # Load data
    data = load_data()
    
    # Show different graphs
    show_histogram(data, 'Pelvic pain')  
    show_scatter_plot(data, 'Menstrual pain (Dysmenorrhea)', 'Pelvic pain')  
    show_bar_plot(data, 'Painful cramps during period')  

if __name__ == "__main__":
    main()