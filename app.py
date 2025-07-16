from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
pipe = pickle.load(open('pipe.pkl', 'rb'))

teams = [
    'Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa', 
    'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka'
]

cities = [
    'Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town', 
    'London', 'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban', 
    'St Lucia', 'Wellington', 'Lauderhill', 'Hamilton', 'Centurion', 
    'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton', 
    'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi', 
    'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts', 'Cardiff', 
    'Christchurch', 'Trinidad'
]

weather_descriptions = ['Clear', 'Cloudy', 'Rainy', 'Foggy', 'Snowy', 'Windy']  # Example weather options

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        batting_team = request.form['batting_team']
        bowling_team = request.form['bowling_team']
        city = request.form['city']
        current_score = int(request.form['current_score'])
        overs = float(request.form['overs'])
        wickets = int(request.form['wickets'])
        last_five = int(request.form['last_five'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        wind_speed = float(request.form['wind_speed'])
        precipitation = float(request.form['precipitation'])
        weather_description = request.form['weather_description']

        # Calculate additional features
        balls_left = 120 - (overs * 6)
        current_run_rate = current_score / overs

        # Prepare input data for the model
        input_df = pd.DataFrame({
            'batting_team': [batting_team], 
            'bowling_team': [bowling_team], 
            'city': [city],
            'current_score': [current_score], 
            'balls_left': [balls_left], 
            'wicket_left': [wickets], 
            'current_run_rate': [current_run_rate], 
            'last_six': [last_five],
            'temperature': [temperature], 
            'humidity': [humidity], 
            'wind_speed': [wind_speed], 
            'precipitation': [precipitation],
            'weather_description': [weather_description]
        })

        # Predict using the model
        result = pipe.predict(input_df)
        prediction = int(result[0])

    return render_template('index.html', teams=sorted(teams), cities=sorted(cities), 
                           weather_descriptions=sorted(weather_descriptions), prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
