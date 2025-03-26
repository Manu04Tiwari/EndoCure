from flask import Flask, jsonify, render_template
import pandas as pd

app = Flask(__name__)

# Route to serve the dataset
@app.route('/dataset')
def get_dataset():
    # Read the Excel file
    df = pd.read_excel('dataset-fi.xlsx')  # Replace with your Excel file path
    return df.to_json(orient='records')  # Convert to JSON format

# Route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)