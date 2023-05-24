import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import json
import requests
import random
import datetime
from datetime import date, datetime, timedelta
import time
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import joblib
import meteostat
from meteostat import Stations, Daily, Monthly
from db_settings import update_table, get_all_data_from_db, connect_to_db
import warnings
import sqlite3
warnings.filterwarnings('ignore')

# def speak(text):
#     # Create gTTS object
#     tts = gTTS(text=text, lang='en')

#     # Save the audio file
#     tts.save('output.mp3')

#     # Play the audio file
#     playsound.playsound('output.mp3')

st.title('Weatherbot Predictor')
st.markdown('My name is 9jaWeatherbot! I will be your weather predictor today. \n\n Wouldn\'t you want to know what the weather says, so you can decide if to go out or stay? :smile:')
st.image('images/weather_unsplash.jpg', caption='Source: Unsplash')

username = st.text_input('What is your name?', key='name')
button = st.button('Please click me to submit.')
if button:
    if username != '':
        # speak(f'Hello, {username}')
        st.markdown(f'Hello, {username}!')
        # speak('Please select a date and location to get your weather prediction...')
    else:
        # speak('Please enter your name...')
        st.warning('Please input your username to continue.')

st.sidebar.markdown('**<u>About 9jaWeatherbot</u>**', unsafe_allow_html=True)
text1 = '''
<justify>9jaWeatherbot was created by Rofiatu Alli, a student of the Pumpkin Redeemer\'s Cohort of the GoMyCode Data Science class. This project was created as her final project for the programme, and while she knows it is not a perfect app, she put in all of the resources that had been taught to her in the course of her programme to come up with this near-masterpiece, and she hopes you find some joy while using it (in the absence of being able to find the most accurate weather data).</justify>
'''
st.sidebar.write(text1, unsafe_allow_html=True)
st.sidebar.markdown('**<u>Making 9jaWeatherbot</u>**', unsafe_allow_html=True)
text2 = '''
<justify> In making 9jaWeatherbot, Rofiatu used daily historical data from the Meteostat API from the period January 1 2012 to May 18 2023. This data was used to build a model that leveraged the XGBoost Regressor to predict temperature for the date selected. The API only had historical data for 26 states, and a few features. As such, this app may be limited to the extent of the historical data available for use in training the model. </justify>
'''
st.sidebar.write(text2, unsafe_allow_html=True)

def get_nigerian_data():

    station = Stations()
    global nig_stations
    nig_stations = station.region('NG') #...................................... Filter stations by country code (NG for Nigeria)
    nig_stations = nig_stations.fetch() # ...................................... Fetch the station information

    return nig_stations

def get_weather_data(nig_stations):

    start_date = datetime(2012, 1, 1)
    end_date = datetime(2023, 5, 18)

    weather_data = Daily(nig_stations, start_date, end_date)
    weather_data = weather_data.fetch()
    weather_data = pd.DataFrame(weather_data).reset_index()
    nig_stations = nig_stations.reset_index()
    weather_data.drop(['snow', 'wpgt', 'tsun'], axis=1, inplace=True)
    weather_data = weather_data.rename(columns={'station': 'id', 'time': 'date', 'tavg': 'avg_temp', 'tmin': 'min_temp', 'tmax': 'max_temp', 'prcp': 'precipitation', 'wdir': 'wind_direction', 'wspd': 'windspeed', 'pres': 'pressure'})
    weather_data = weather_data.merge(nig_stations[['id', 'latitude', 'longitude']], on='id')
    reordered_colums = ['date', 'latitude', 'longitude', 'avg_temp', 'min_temp', 'max_temp', 'pressure']
    weather_data = weather_data[reordered_colums]
    weather_data['date'] = pd.to_datetime(weather_data['date'])
    
    return weather_data

def clean_data(weather_data):

    # replace null values with mean of the respective columns
    columns_data = weather_data.drop(['date'], axis=1)
    cols = list(columns_data.columns)

    for i in cols:
        weather_data[i].fillna(weather_data[i].mean(),inplace=True)
    weather_data.isnull().sum().sort_values(ascending=False)

    # # plot the data to find any incidence of multicolinearity
    # sns.heatmap(weather_data.corr(), annot=True)

    return weather_data

# standardizing data

def standardize(weather_data):

    import datetime as dt

    weather_data['date'] = weather_data['date'].map(dt.datetime.timestamp)

    return weather_data

# modelling

def modelling(weather_data):

    # find min_temp, which is y1
    x1 = weather_data[['date', 'latitude', 'longitude']]
    y1 = weather_data['min_temp']
    
    x1_train, x1_valid, y1_train, y1_valid = train_test_split(x1, y1, test_size=0.2, random_state=20)
        
    xgb1 = XGBRegressor()
    xgb1.fit(x1_train, y1_train)
    y_pred1 = xgb1.predict(x1_valid)
    y_pred1 = pd.Series(y_pred1)
    r2 = r2_score(y_pred1, y1_valid)
    print("R-squared Score:", r2)

    # find max_temperature, which is y2
    x2 = weather_data[['date', 'latitude', 'longitude']]
    y2 = weather_data['max_temp']
    
    x2_train, x2_valid, y2_train, y2_valid = train_test_split(x2, y2, test_size=0.2, random_state=20)
        
    xgb2 = XGBRegressor()
    xgb2.fit(x2_train, y2_train)
    y_pred2 = xgb2.predict(x2_valid)
    y_pred2 = pd.Series(y_pred2)
    r2 = r2_score(y_pred2, y2_valid)
    print("R-squared Score:", r2)

    # find pressure, which is y3
    x3 = weather_data[['date', 'latitude', 'longitude']]
    y3 = weather_data['pressure']
    
    x3_train, x3_valid, y3_train, y3_valid = train_test_split(x3, y3, test_size=0.2, random_state=20)
        
    xgb3 = XGBRegressor()
    xgb3.fit(x3_train, y3_train)
    y_pred3 = xgb3.predict(x3_valid)
    y_pred3 = pd.Series(y_pred3)
    r2 = r2_score(y_pred3, y3_valid)
    print("R-squared Score:", r2)

    # find avg_temp, which is y4
    x4 = weather_data[['date', 'latitude', 'longitude']]
    y4 = weather_data['avg_temp']
    
    x4_train, x4_valid, y4_train, y4_valid = train_test_split(x4, y4, test_size=0.2, random_state=20)
        
    xgb4 = XGBRegressor()
    xgb4.fit(x4_train, y4_train)
    y_pred4 = xgb4.predict(x4_valid)
    y_pred4 = pd.Series(y_pred4)
    r2 = r2_score(y_pred4, y4_valid)
    print("R-squared Score:", r2)

    # save your model
    joblib.dump(xgb1, 'min_temp_model.pkl')
    joblib.dump(xgb2, 'max_temp_model.pkl')
    joblib.dump(xgb3, 'pressure_model.pkl')
    joblib.dump(xgb4, 'avg_temp_model.pkl')

nig_stations = get_nigerian_data()
list_of_states = [i for i in nig_stations['name']]
# weather_data = get_weather_data(nig_stations)
# original_data = weather_data.copy()
# clean_weather_data = clean_data(weather_data)
# unclean_data = clean_weather_data.copy()
# standard_weather_data = standardize(clean_weather_data)
# modelling(standard_weather_data)

# ------------------------- get user's input ----------------------------------

def get_user_agent(place):
    url = f"https://nominatim.openstreetmap.org/search?q={place}&format=json"
    response = requests.get(url)
    response_json = response.json()
    latitude = response_json[0]["lat"]
    longitude = response_json[0]["lon"]
    return latitude, longitude

def user_selected_date():
    
    # Get the current date
    current_date = datetime.now().date()

    # Calculate the maximum allowed date
    max_date = current_date + timedelta(days=180)

    # Display the date input with restricted range
    selected_date = st.date_input("Select a date: ", min_value=current_date, max_value=max_date, value=current_date)

    if not selected_date:
        selected_date = current_date
    else:
        selected_date = selected_date

    return selected_date

selected_date = user_selected_date()

def user_location():

    # initialize geocoding API
    global place
    # place = st.text_input('Write the name of the city you\'d like to get weather predictions for...', key='location1')
    options = [None] + list_of_states
    place = st.selectbox('Please select the name of the Nigerian city you\'d like to get weather predictions for...', options, key='location1')
    longitude, latitude = 0, 0

    if place:
        latitude, longitude = get_user_agent(place)
        long = st.write(f'The longitude of your location is: {longitude}')
        lat = st.write(f'The latitude of your location is: {latitude}')
        st.success(f'Please click on the button below to get the weather information for {place}...')

    elif place is None:
        st.warning('Please select a State from the drop-down box to proceed.')

    return longitude, latitude


def read_user():

    print("reading user data...")
    pd.set_option('display.max_columns', None)
    global longitude, latitude
    longitude, latitude = user_location()
    date = selected_date.strftime("%Y-%m-%d")
    user_data = pd.DataFrame({'date': [date], 'latitude': [latitude], 'longitude': [longitude]})
    user_data['date'] = pd.to_datetime(user_data['date'])
    
    return user_data

# ---------------------- streamlit implementation -----------------------------

def main():

    import datetime as dt
    print("STARTED ...")
    user_data = read_user()
    cleaned_data = clean_data(user_data)
    standardized_data = standardize(cleaned_data)

    input_variables = standardized_data

    print('printing input variables', input_variables)
    input_v = np.array(input_variables)

    global min_temp_pred, max_temp_pred, avg_temp_pred, pressure_pred

    # load the model
    # predict user's min temp
    model_min_temp = joblib.load(open('min_temp_model.pkl','rb'))
    min_temp_pred = model_min_temp.predict(input_v)
    min_temp_pred = int(min_temp_pred[0])
    print(f'The min_temp value is: {min_temp_pred}')

    # predict user's max temp
    model_max_temp = joblib.load(open('max_temp_model.pkl','rb'))
    max_temp_pred = model_max_temp.predict(input_v)
    max_temp_pred = int(max_temp_pred[0])
    print(f'The max_temp value is: {max_temp_pred}')

    # predict user's pressure
    model_pressure = joblib.load(open('pressure_model.pkl','rb'))
    pressure_pred = model_pressure.predict(input_v)
    pressure_pred = int(pressure_pred[0])
    print(f'The pressure value is: {pressure_pred}')

    # predict user's avg temp
    model_avg_temp = joblib.load(open('avg_temp_model.pkl','rb'))
    avg_temp_pred = model_avg_temp.predict(input_v)
    avg_temp_pred = int(avg_temp_pred[0])
    print(f'The avg_temp value is: {avg_temp_pred}')

    current_date = date.today()

    predict_button = st.button('Predict the weather')

    if place:
        if predict_button:
            # display summary of user's input
            frame = ({'place':[place], 'date': [selected_date], 'longitude': [longitude], 'latitude': [latitude]})
            st.write('These are your input variables: ')
            frame = pd.DataFrame(frame)
            frame = frame.rename(index = {0: 'Value'})
            frame = frame.transpose()
            st.write(frame)

            col1, col2, col3= st.columns(3)
            col1.metric("Low Temperature", f"{min_temp_pred} °C")
            col2.metric("Temperature", f"{avg_temp_pred} °C")
            col3.metric("High Temperature", f"{max_temp_pred} °C")

            if avg_temp_pred >= 0 and avg_temp_pred < 10:
                summary = f'The weather in your selected location {place} is classified as: Cold, with an average temperature of {avg_temp_pred} °C'
                # speak(summary)
                st.warning(f'The weather in your selected location {place} is classified as: Cold U+1F976')
                st.image('images/cold_weather.jpg', caption = 'Source: Unsplash', use_column_width = True)
                st.write('You should probably consider staying in today!')
                suggestion = np.random.choice([
                    "Cozy Movie Marathon: Gather some blankets, make yourself a warm drink, and indulge in a movie marathon. Watch your favorite movies or explore new genres. It's a perfect way to stay warm and entertained indoors.",

                    "Winter Hike or Nature Walk: Bundle up in warm clothing and explore the beauty of winter outdoors. Find a local park or trail with scenic winter views and go for a hike or nature walk. Don't forget to take some hot beverages or snacks to enjoy along the way.",

                    "Indoor Board Games or Puzzles: Invite friends or family over and spend quality time together playing board games or solving puzzles. It's a fun and engaging way to stay entertained while staying cozy indoors. Choose games that are suitable for a group or challenge yourself with a complex puzzle.",

                ])
                intro = 'When the weather is cool, here is an idea on what to do: \n\n'
                outro = '\n\n Remember to adapt these suggestions based on your personal preferences and the resources available to you. Stay warm and enjoy your cold weather activities!'
                message = intro + suggestion + outro
                st.write(message)
                st.info(f'Classified as at: {current_date}')

            elif avg_temp_pred >= 10 and avg_temp_pred < 20:
                summary = f'The weather in your selected location {place} is classified as: Cool, with an average temperature of {avg_temp_pred} °C'
                # speak(summary)
                st.success(f'The weather in your selected location {place} is classified as: Cool :grinning:' )
                st.image('images/cool_weather.jpg', caption = 'Source: Unsplash', use_column_width = True)
                st.write('Today is a good day to go out and play!')
                suggestion = np.random.choice([
                    "Enjoy Outdoor Activities: Cool weather can be perfect for engaging in outdoor activities. You can go for a hike or a nature walk, explore local parks, have a picnic, or even try your hand at outdoor sports like cycling or rock climbing. The cool temperature can make these activities more enjoyable and refreshing.",

                    "Cozy up with a Book or Movie: Cool weather often creates a cozy atmosphere, making it an ideal time to relax indoors with a good book or movie. Grab a warm blanket, make yourself a hot drink, and spend some quality time immersed in a captivating story. You can explore new genres, dive into your favorite series, or even have a movie marathon with friends or family.",

                    "Try New Recipes: Cool weather can inspire you to get creative in the kitchen. Take advantage of the cooler temperature by experimenting with new recipes for comforting dishes like soups, stews, casseroles, or baked goods. You can also try your hand at making hot beverages like mulled cider or hot chocolate. Cooking or baking can be a fun and rewarding activity, and you get to enjoy delicious treats afterward.",

                ])
                intro = 'When the weather is cool, here is an idea on what to do: \n\n'
                outro = '\n\n Remember to consider weather conditions, stay hydrated, and protect yourself from excessive sun exposure by wearing sunscreen and appropriate clothing. Enjoy the clear skies and make the most of your time outdoors!'
                message = intro + suggestion + outro
                st.write(message)
                st.info(f'Classified as at: {current_date}')

            elif avg_temp_pred >= 20 and avg_temp_pred < 25:
                summary = f'The weather in your selected location {place} is classified as: Mild, with an average temperature of {avg_temp_pred} °C'
                # speak(summary)
                st.success(f'The weather in your selected location {place} is classified as: Mild :sunglasses:')
                st.image('images/mild_weather.jpg', caption = 'Source: Unsplash', use_column_width = True)
                st.write('It\'s a perfect outing day...')
                suggestion = np.random.choice([
                    "Go for a hike or nature walk: Take advantage of the pleasant weather and explore the great outdoors. Find a nearby trail or park and embark on a hike or nature walk. Enjoy the beauty of nature, breathe in the fresh air, and appreciate the scenery around you. It's a great way to stay active and connect with nature.", 

                    "Visit a local farmers market: Mild weather often means farmers markets are in full swing. Explore your local farmers market and discover a variety of fresh produce, artisanal products, baked goods, and more. Support local farmers and vendors while enjoying a leisurely stroll through the market. You can also try new flavors, sample local delicacies, and even participate in cooking demonstrations or workshops.",

                    "Have a picnic or outdoor barbecue: Take advantage of the pleasant temperatures by organizing a picnic or outdoor barbecue with friends or family. Find a picturesque spot in a park or set up a cozy outdoor space in your backyard. Prepare delicious food, pack a blanket or set up a grill, and enjoy a relaxing time outdoors. It's an opportunity to socialize, savor tasty food, and enjoy the mild weather.",

                ])
                intro = 'Here is an idea on how to make it great: \n\n'
                outro = '\n\n These activities allow you to enjoy the comfortable weather and engage in outdoor experiences that cater to your interests and preferences. Make the most of the mild weather by embracing the opportunities it offers for outdoor enjoyment and relaxation.'
                message = intro + suggestion + outro
                st.write(message)
                st.info(f'classified as at: {current_date}')

            elif avg_temp_pred >= 25 and avg_temp_pred < 30:
                summary = f'The weather in your selected location {place} is classified as: Warm, with an average temperature of {avg_temp_pred} °C'
                # speak(summary)
                st.success(f'The weather in your selected location {place} is classified as: Warm :smile:')
                st.image('images/warm_weather.jpg', caption = 'Source: Unsplash', use_column_width = True)
                suggestion = np.random.choice([
                    "Go swimming or visit the beach: Take advantage of the warm weather by cooling off in the water. Whether it's a local swimming pool, a nearby lake, or the ocean, swimming can be a refreshing and enjoyable activity. If you're close to a beach, spend a day lounging on the sand, playing beach games, and taking a dip in the water. It's a great way to relax and have fun under the sun.", 

                    "Have a picnic in the park: Enjoy the pleasant weather by having a picnic in a nearby park. Pack a delicious spread of food, bring a blanket or a picnic mat, and find a shady spot to unwind. You can bring along games, books, or a musical instrument to make the experience even more enjoyable. It's a fantastic way to spend quality time outdoors, connect with nature, and enjoy a leisurely meal.",

                    "Explore outdoor markets or festivals: Warm weather often brings out vibrant outdoor markets and festivals. Look for local farmers markets, craft fairs, art festivals, or food festivals happening in your area. Wander through the stalls, browse unique products, sample delicious food, and immerse yourself in the lively atmosphere. It's an opportunity to support local businesses, discover new flavors, and soak up the energy of the community.",

                ])
                intro = 'Here is a tip on how to keep a warm smile on this warm day: \n\n'
                outro = '\n\n These activities allow you to embrace the warmth and make the most of the pleasant weather. Whether it\'s cooling off in the water, enjoying a picnic, or immersing yourself in outdoor events, there are plenty of exciting things to do when the weather is warm.'
                message = intro + suggestion + outro
                st.write(message)
                st.info(f'classified as at: {current_date}')

            elif avg_temp_pred >= 30 and avg_temp_pred < 35:
                summary = f'The weather in your selected location {place} is classified as: Hot, with an average temperature of {avg_temp_pred} °C'
                # speak(summary)
                st.warning(f'The weather in your selected location {place} is classified as: Hot :fire:')
                st.image('images/hot_weather.jpg', caption = 'Source: Unsplash', use_column_width = True)
                st.write('Roses are red, Violets are blue; the Sun is hot, and so are you!')
                suggestion = np.random.choice([
                    "Go to the beach: Enjoy a refreshing dip in the ocean, relax on the sandy shores, and soak up the sun. Swimming, playing beach games, and building sandcastles are all fun activities to do when it's hot outside.", 

                    "Have a picnic in the park: Find a shady spot in a nearby park and have a picnic with family or friends. Pack refreshing drinks, chilled fruits, sandwiches, and snacks to keep yourself hydrated and energized. You can also bring outdoor games or a frisbee to play while enjoying the outdoors.",

                    "Visit a water park or pool: Beat the heat by spending the day at a water park or swimming pool. Slide down thrilling water slides, float along lazy rivers, or simply relax in a pool while staying cool. Water parks and pools offer a range of activities and attractions for all ages to enjoy during hot weather.",

                ])
                intro = 'On this sunny day, here is an idea on how you can slay: \n\n'
                outro = '\n\n Remember to stay hydrated, apply sunscreen, and take necessary precautions to ensure your safety and well-being during hot weather conditions.'
                message = intro + suggestion + outro
                st.write(message)
                st.info(f'classified as at: {current_date}')

            elif avg_temp_pred >= 35:
                summary = f'The weather in your selected location {place} is classified as: Very Hot, with an average temperature of {avg_temp_pred} °C'
                # speak(summary)
                st.warning(f'The weather in your selected location {place} is classified as: Very Hot :warning:')
                st.image('images/sweating.jpg', caption = 'Source: Unsplash', use_column_width = True)
                st.write('The sun rays aren\'t smiling today...')
                suggestion = np.random.choice([
                    "Water Fun Day: Beat the heat by having a water fun day. Set up a sprinkler in your backyard, have a water balloon fight, or set up a small pool or inflatable water slide. You can also visit a nearby water park or swimming pool to cool off and enjoy various water activities.", 

                    "Indoor Movie Marathon: If the heat is unbearable outside, create your own indoor cinema experience. Pick a theme or genre, gather some snacks and refreshments, and binge-watch your favorite movies or TV series in the comfort of air conditioning. Don't forget to grab a cozy blanket for added comfort.",

                    "Ice Cream Social: Embrace the hot weather by indulging in a cool treat. Organize an ice cream social with friends or family. Set up an ice cream station with a variety of flavors, toppings, and cones. You can also make homemade popsicles or create unique ice cream sundaes with different toppings and sauces.",

                ])
                intro = 'Here is a fun idea for this sunny day: \n\n'
                outro = '\n\n Remember to stay hydrated and take necessary precautions to protect yourself from the heat, such as wearing sunscreen and seeking shade when needed. Enjoy your activities and stay cool!'
                message = intro + suggestion + outro
                st.write(message)
                st.info(f'classified as at: {current_date}')

# ------------------------- Database Implementation ------------------------------ #

    data = {
        'username': username,
        'place': place,
        'selected_date': selected_date,
        'min_temp_pred': min_temp_pred,
        'max_temp_pred': max_temp_pred,
        'avg_temp_pred': avg_temp_pred,
        'pressure_pred': pressure_pred,
        'current_date': current_date
    }
    if predict_button:
        conn = connect_to_db()
        update_table(data, conn)
        data_history = get_all_data_from_db(username, conn)

        for i in data_history:
            st.write(current_date, i)

if __name__ == "__main__":
    main()
