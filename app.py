from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64

app = Flask(__name__)

def load_data():
    data = pd.read_excel('DataSet-fi.xlsx')
    return data

def plot_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = base64.b64encode(buf.getvalue()).decode('ascii')
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/histogram')
def show_histogram():
    data = load_data()
    column = request.args.get('column', default='Pelvic pain')
    fig = plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    img = plot_to_img(fig)
    return render_template('plot.html', plot_img=img, plot_title=f'Histogram of {column}')

@app.route('/scatter')
def show_scatter_plot():
    data = load_data()
    x_column = request.args.get('x_column', default='Menstrual pain (Dysmenorrhea)')
    y_column = request.args.get('y_column', default='Pelvic pain')
    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x=x_column, y=y_column)
    plt.title(f'Scatter Plot of {x_column} vs {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    img = plot_to_img(fig)
    return render_template('plot.html', plot_img=img, plot_title=f'Scatter Plot of {x_column} vs {y_column}')

@app.route('/bar')
def show_bar_plot():
    data = load_data()
    column = request.args.get('column', default='Painful cramps during period')
    fig = plt.figure(figsize=(10, 6))
    sns.countplot(x=column, data=data)
    plt.title(f'Bar Plot of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    img = plot_to_img(fig)
    return render_template('plot.html', plot_img=img, plot_title=f'Bar Plot of {column}')

if __name__ == "__main__":
    app.run(debug=True)