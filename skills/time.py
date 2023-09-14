from datetime import datetime
import pytz
from dotenv import load_dotenv
import requests

import os

load_dotenv()

function_name = "timecheck"
parameter_names = ['location']


class TimeSkill:
    def __init__(self, message_function):
        self.message_function = message_function
        self.api_key = os.getenv("GEONAMES_USERNAME")
        if (self.api_key == None):
            raise Exception(
                "No geonames username found in .env file! Go to https://www.geonames.org/ and get a user name and put in .env as GEONAMES_USER_NAME")

    def start(self, args: dict[str, str]):
        time_string = args["location"]
        time = ""
        if (time_string == None or time_string == ""):
            time = self._get_current_time()
        else:
            time = self._get_current_time_in_city(time_string)

        formatted_time = time.strftime("%H:%M")

        self.message_function(formatted_time)

    def _get_current_time_in_city(self, city: str) -> datetime:
        # Step 1: Get timezone of the city using GeoNames API
        username = self.api_key
        params = {"username": username, "q": city, "maxRows": 1}
        url = f"http://api.geonames.org/searchJSON"
        response = requests.get(url, params=params)
        data = response.json()

        if not data.get("geonames"):
            raise ValueError(
                f"City {city} not found or GeoNames API error!")

        lat = data["geonames"][0]["lat"]
        long = data["geonames"][0]["lng"]

        url2 = f"http://api.geonames.org/timezoneJSON"
        params2 = {"username": username, "lat": lat, "lng": long}

        response2 = requests.get(url2, params=params2)
        data2 = response2.json()

        timezone_id = data2["timezoneId"]

        # Step 2: Get the current time in that timezone
        timezone = pytz.timezone(timezone_id)
        local_time = datetime.now(timezone)

        return local_time

    def _get_current_time(self):
        return datetime.now()


if __name__ == '__main__':
    # Example
    time = TimeSkill(print)

    city_name = "Tokyo"
    time.start({"location": city_name})