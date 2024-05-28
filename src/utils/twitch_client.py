import time
import socket
import threading
from .enviornment import env
from datetime import datetime, timedelta
import requests

class TwitchClient():
    def __init__(self, server=env.irc_server, port=env.irc_port, oauth_token=env.oauth, 
                bot_name=env.bot_name, channel=env.channel, client_id=env.client_id):
        self.server = server
        self.port = port
        self.oauth_token = oauth_token
        self.bot_name = bot_name
        self.channel = channel
        self.client_id = client_id
        self.stop_event = threading.Event()
        
        # messages
        self.messages = []

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

    def run_message_gather(self):
        self.message_gather_thread = threading.Thread(target=self.gather_messages)
        self.message_gather_thread.start()

    def gather_messages(self):
        message_count = 0
        while not self.stop_event.is_set():
            message_count += 1
            # only get messages that are not welcome messages
            if message_count < 4:
                continue
            message = self.read_message() # get message
            if message is None:
                continue
            # format and store the messages
            message = message.split(f'#{self.channel} :')[-1]
            self.messages.append({'message': message, 'timestamp': datetime.now()})

    def stop_message_gather(self):
        self.stop_event.set()
        self.message_gather_thread.join()

    def get_messages_window(self, window=15):
        temp_data = self.messages.copy()
        end = datetime.now()
        start = end - timedelta(seconds=window)
        return [d['message'].split('\r\n')[0] for d in temp_data if d['timestamp'] >= start and d['timestamp'] <= end]

    def get_channel_status(self):
        url = f'https://api.twitch.tv/helix/streams?user_login={self.channel}'

        headers = {
            'Authorization': f'Bearer {self.oauth_token}',
            'Client-Id': self.client_id
        }
        response = requests.get(url, headers=headers)
        data = response.json()
        return len(data['data']) > 0

    # read messages from the IRC server
    def read_message(self):
        try:
            self.irc.settimeout(5)
            resp = self.irc.recv(2048).decode('utf-8')
            return resp
        except socket.timeout:
            return None
        finally:
            self.irc.settimeout(None)
        
    # send privmsg's, which are normal chat messages
    def send_message(self, message):
        self.irc_command(f"PRIVMSG #{self.channel.lower()} :{message}")
        print('Sent "{}" to {}'.format(message,self.channel))