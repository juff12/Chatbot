import time
import socket
from .enviornment import env

class TwitchClient():
    def __init__(self, server=env.irc_server, port=env.irc_port, oauth_token=env.oauth, 
                bot_name=env.bot_name, channel=env.channel, client_id=env.client_id):
        self.server = server
        self.port = port
        self.oauth_token = oauth_token
        self.bot_name = bot_name
        self.channel = channel
        self.client_id = client_id

    # connect to IRC server and begin checking for messages
    def connect_to_channel(self):
        self.irc = socket.socket()
        self.irc.connect((self.server, self.port))
        print(f"Connecting to {self.server} on port {self.port}")
        time.sleep(1)
        # log in
        self.irc_command(f"CAP REQ :twitch.tv/commands")
        self.irc_command(f"PASS oauth:{self.oauth_token}")
        self.irc_command(f"NICK {self.bot_name.lower()}")
        self.irc_command(f"JOIN #{self.channel.lower()}")
        
    # execute IRC commands
    def irc_command(self, command):
        self.irc.send((command + "\r\n").encode('utf-8'))

    # send privmsg's, which are normal chat messages
    def send_message(self, message):
        self.irc_command(f"PRIVMSG #{self.channel.lower()} :{message}")
        print('Sent "{}" to {}'.format(message,self.channel))