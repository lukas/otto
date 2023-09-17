from dotenv import load_dotenv
import requests

import os

load_dotenv()


class WeatherSkill:
    function_name = "weather"
    parameter_names = ['location']

    examples = [
        [
            "what is the weather?",
            "weather()"
        ],
        [
            "what is the weather in Boston?",
            "timecheck(location=\"Boston\")"
        ]
    ]

    def __init__(self, message_function):
        self.message_function = message_function
        self.api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        if (self.api_key == None):
            raise Exception(
                "No OpenWeatherMap API key found in .env file! Go to openweathermap.org to get a free key and put in .env as OPENWEATHERMAP_API_KEY")

    def start(self, args: dict[str, str]):

        if 'location' in args:
            location_string = args['location']
        else:
            location_string = "San Francisco"

        self.get_weather(location_string)

    def get_weather(self, city_name):
        print("Getting weather for " + city_name)
        params = {"q": city_name,
                  "appid": self.api_key, "units": "imperial"}
        base_url = "http://api.openweathermap.org/data/2.5/weather"

        response = requests.get(base_url, params=params)

        # Check HTTP status code
        if response.status_code == 200:
            data = response.json()

            # Extract and print relevant data
            main_data = data['main']
            weather_data = data['weather'][0]
            weather_string = \
                f"Weather in {city_name}:\n" \
                f"{weather_data['description']}\n"\
                f"Temperature: {main_data['temp']} Farenheit\n" \
                f"Pressure: {main_data['pressure']} hPa\n" \
                f"Humidity: {main_data['humidity']}%"

            print("Weather data: " + weather_string)
            self.message_function(weather_string)

        else:
            print("Error fetching data from OpenWeatherMap API!")
            print(response)


if __name__ == '__main__':
    # for testing
    weather = WeatherSkill(print)
    weather.start({"location": "san francisco"})
