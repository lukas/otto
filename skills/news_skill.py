import requests
import os
from dotenv import load_dotenv

load_dotenv()

# News API endpoint URL
HEADLINE_NEWS_API_URL = 'https://newsapi.org/v2/top-headlines'
NEWS_API_URL = 'https://newsapi.org/v2/everything'

DEFAULT_TOPIC = "general"
DEFAULT_COUNTRY = "us"
# Other options
# ISO 3166-1 country code


class NewsSkill:
    function_name = "news"
    parameter_names = ['topic']

    examples = [
        [
            "what is the news?",
            "news()"
        ],
        [
            "tell me some business headlines",
            "news(topic=\"business\")"
        ],

        [
            "tell me the latest news about the environment",
            "news(topic=\"environment\")"
        ]
    ]

    def __init__(self, message_function):
        self.message_function = message_function
        self.api_key = os.getenv("NEWS_API_KEY")
        if (self.api_key == None):
            raise RuntimeError(
                "No News API key found in .env file! Go to newsapi.org to get a free key and put in .env as NEWS_API_KEY")

    def start(self, args: dict[str, str]):

        if 'topic' in args:
            topic_string = args['topic']
        else:
            topic_string = DEFAULT_TOPIC

        self._get_news(topic_string)

    def _get_titles_from_response(self, response):
        data = response.json()
        titles = []
        # Extract and display the news articles
        articles = data.get('articles', [])
        for idx, article in enumerate(articles, start=1):
            if article['title'] == None:
                continue
            if article['title'] == '[Removed]':
                continue

            titles.append(article['title'])
        return titles

    def _get_news(self, topic_string):

        # Parameters for the request (you can customize these)
        params = {
            'apiKey': self.api_key,
            'pageSize': 5,  # Number of articles to retrieve
        }

        if (topic_string != DEFAULT_TOPIC):
            params['q'] = topic_string

        # Make the GET request to the News API
        response = requests.get(HEADLINE_NEWS_API_URL, params=params)
        titles = []
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the JSON response

            titles = self._get_titles_from_response(response)
            if len(titles) == 0:
                response = requests.get(NEWS_API_URL, params=params)
                titles = self._get_titles_from_response(response)
                if (len(titles) == 0):
                    print("No articles found!")

        else:
            # If the request was not successful, print an error message
            print(
                f"Failed to retrieve news articles. Status code: {response.status_code} Error: {response.text}")

        self.message_function("\n".join(titles))


if __name__ == '__main__':
    # for testing
    news = NewsSkill(print)
    news.start({"topic": "astronomy"})
