from flask import Flask, render_template, request
import pandas as pd
import joblib  # For loading the trained model

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Load the dataset
data = pd.read_csv('dataset.csv')

@app.route('/')
def index():
    # Get a list of unique student names for the dropdown menu
    student_names = data['Student_Name'].unique()
    return render_template('index.html', student_names=student_names)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the selected student name from the dropdown
    selected_student = request.form.get('student_name')
    
    # Find the student's data
    student_data = data[data['Student_Name'] == selected_student]
    
    # Extract features for prediction (you may need to adjust this based on your data)
    features = student_data.drop(['Student_Name', 'Dropout_Label'], axis=1)
    
    # Make the prediction
    prediction = model.predict(features)
    
    # Display the prediction on a new page (create a corresponding HTML template)
    return render_template('prediction.html', student=selected_student, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=False)
