import spacy
import requests

api_key = "b35710991c78f784118060233839818a"

def get_weather(city_name):
    api_url = "http://api.openweathermap.org/data/2.5/weather?q={}&appid={}".format(city_name, api_key)
    
    response = requests.get(api_url)
    response_dict = response.json()

    weather = response_dict["weather"][0]["description"]
    
    if response.status_code == 200:
        return weather
    else:
        print('[!] HTTP {0} calling [{1}]'.format(response.status_code, api_url))
        return None

nlp = spacy.load('en_core_web_md')

def chatbot(statement):
  weather = nlp("Current weather in a city")
  statement = nlp(statement)
  min_similarity = 0.40

  if weather.similarity(statement) >= min_similarity:
    for ent in statement.ents:
        if ent.label_ == "GPE": # GeoPolitical Entity
            city = ent.text
            break
        else:
            return "You need to mention the city whose weather condition you are interested in."

    weather = get_weather(city)
    if weather is not None:
        return "In " + city + ", the current weather is: " + weather
    else:
      return "Something went wrong."
  else:
    return "Sorry I don't understand that. Please rephrase your statement."

# weather = get_weather("Maryland")
# print(weather)

statement1 = chatbot("I wonder if it is hot in 'Middle River, Maryland' today.")
print(statement1)
