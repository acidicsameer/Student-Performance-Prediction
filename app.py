from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
pipe = joblib.load("student_performance_pipeline.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "Hours_Studied": [float(request.form["Hours_Studied"])],
        "Attendance": [float(request.form["Attendance"])],
        "Parental_Involvement": [request.form["Parental_Involvement"]],
        "Access_to_Resources": [request.form["Access_to_Resources"]],
        "Extracurricular_Activities": [request.form["Extracurricular_Activities"]],
        "Sleep_Hours": [float(request.form["Sleep_Hours"])],
        "Previous_Scores": [float(request.form["Previous_Scores"])],
        "Motivation_Level": [request.form["Motivation_Level"]],
        "Internet_Access": [request.form["Internet_Access"]],
        "Tutoring_Sessions": [float(request.form["Tutoring_Sessions"])],
        "Family_Income": [request.form["Family_Income"]],
        "Teacher_Quality": [request.form["Teacher_Quality"]],
        "School_Type": [request.form["School_Type"]],
        "Peer_Influence": [request.form["Peer_Influence"]],
        "Physical_Activity": [float(request.form["Physical_Activity"])],
        "Learning_Disabilities": [request.form["Learning_Disabilities"]],
        "Parental_Education_Level": [request.form["Parental_Education_Level"]],
        "Distance_from_Home": [request.form["Distance_from_Home"]],
        "Gender": [request.form["Gender"]]
    }

    input_df = pd.DataFrame(data)

    prediction = pipe.predict(input_df)

    return render_template(
        "index.html",
        prediction_text=f"Predicted Exam Score: {prediction[0]:.2f}"
    )


if __name__ == "__main__":
    app.run(debug=True)