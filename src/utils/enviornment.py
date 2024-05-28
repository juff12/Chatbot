import os 
from dotenv import load_dotenv
load_dotenv("creds/credentials.env")
import time

# prevents loading too fast and not getting the creds
time.sleep(1)

class Environment():
    def __init__(self):
        # get creds from env file
        self.channel = os.getenv("CHANNEL")
        self.bot_name = os.getenv("BOT_NAME")
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        self.oauth = os.getenv("OAUTH_TOKEN")

        # these are pre-defined
        self.irc_port = 6667
        self.irc_server = "irc.chat.twitch.tv"

env = Environment()