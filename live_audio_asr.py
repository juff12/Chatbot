from transformers import pipeline, BertTokenizer, BertForSequenceClassification
import torch
import datetime
import keyboard
import argparse
from sentence_transformers import SentenceTransformer, util
from src.chatbot import Chatbot
from src.utils import TwitchClient
import subprocess
import os
import numpy as np
import json

# added here due to conflict in the original code
class AudioPipeline():
    def __init__(self, model_name, device, infile, tempfile='data/audio/temp.wav', outfile='data/audio/output.wav', freq=44100, duration=5):
        self.transcriber = self.create_transcriber(model_name, device)
        self.infile = infile
        self.tempfile = tempfile
        self.outfile = outfile
        self.freq = freq
        self.duration = duration

    def create_transcriber(self, model_name, device):
        # create the transcriber
        transcriber = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            chunk_length_s=30,
            device=device
        )
        return transcriber

    def transcribe(self, audio_file):
        return self.transcriber(audio_file)['text']

    # grab a slice of the audio
    def audio_slice(self):
        # get the current audio as a wav file, save in temp location
        command = ['ffmpeg', '-y', '-i', f'{self.infile}', '-vn', '-acodec',
                'pcm_s16le', '-ar', f'{self.freq}', '-ac', '2', f'{self.tempfile}']
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        # get the last 10 seconds of the audio
        command = ['ffmpeg', '-sseof', f'-{self.duration}', '-y', '-i', f'{self.tempfile}',
                '-vn', '-acodec', 'pcm_s16le', '-ar', f'{self.freq}', '-ac', '2', f'{self.outfile}']
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        # remove the temp file
        os.remove(self.tempfile)
        
class OutputChecker():
    def __init__(self, semantic_model, coherence_model):
        self.semantic_model = self.create_semantic_model(semantic_model)
        self.coherence_model = self.create_coherence_model(coherence_model)
        self.coherence_tokenizer = self.create_coherence_tokenizer(coherence_model)
    
    def create_semantic_model(self, model_name):
        model = SentenceTransformer(model_name)
        return model

    def create_coherence_model(self, model_name):
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)
        return model
    
    def create_coherence_tokenizer(self, model_name):
        tokenizer = BertTokenizer.from_pretrained(model_name)
        return tokenizer

    def check_coherence(self, responses):
        scores = []
        for sentence in responses:
            inputs = self.coherence_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
            outputs = self.coherence_model(**inputs)
            score = outputs.logits.item()
            scores.append(score)
        return scores

    def check_relevancy(self, input, output):
        input_embedding = self.semantic_model.encode(input)
        output_embedding = self.semantic_model.encode(output)
        sim = util.cos_sim(input_embedding, output_embedding)
        return sim.cpu().numpy().flatten()
    
def args():
    parser = argparse.ArgumentParser(description='Live Audio Chatbot')
    parser.add_argument('--message_rate', type=int, default=300, help='The rate at which new messages are sent')
    parser.add_argument('--prompt_format', type=str, default='mistral', help='The format of the model [llama/mistral]')
    parser.add_argument('--tokenizer_name', type=str, default='mistralai/Mistral-7B-Instruct-v0.2', help='The base model')
    parser.add_argument('--model_name', type=str, default='', help='The new model')
    parser.add_argument('--asr_model', type=str, default='openai/whisper-base.en', help='The ASR model to use')
    parser.add_argument('--freq', type=int, default=44100, help='The frequency of the audio')
    parser.add_argument('--duration', type=int, default=15, help='The duration of the audio')
    parser.add_argument('--relevancy_score', type=float, default=0.3, help='The relevancy score to use for the message and response similarity')

    parser.add_argument('--testing', type=bool, default=True, help='If testing no limit on message rate')

    parser.add_argument('--temp_file', type=str, default='data/audio/temp.wav', help='The temp file to hold the wav file while slicing')
    parser.add_argument('--output_file', type=str, default='data/audio/output.wav', help='The output file to use')
    parser.add_argument('--input_file', type=str, default='', help='The input file to use [blank is default]')

    return parser.parse_args()
    
def save_file(data, filename):
    # Write data to the file
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def main():
    # Generate a unique identifier based on current date and time
    # for the save file operation
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename_all = f"data/runs/run_{timestamp}.json"
    filename_accepted = f"data/outputs/run_{timestamp}_output.json"
    filename_actual = f"data/actual/run_{timestamp}_actual.json"
    opt = args()

    # set the device for the transcriber
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if opt.input_file == '':
        # use the first file in the directory
        input_file = [os.path.join('data/audio', file) for file in os.listdir('data/audio') if file.endswith('.mkv')][0]
    else:
        input_file = opt.input_file
    
    # format the files
    input_file = os.path.abspath(input_file)

    try:
        # initialize the aunio pipeline
        audio_pipe = AudioPipeline(opt.asr_model, device, opt.input_file, opt.temp_file,
                                   opt.output_file, freq=opt.freq, duration=opt.duration)
    except Exception as e:
        print("Error loading transcriber: ", e)
        return
    
    try:
        # initialize the chatbot
        chatbot = Chatbot(opt.model_name, opt.tokenizer_name, device, opt.prompt_format)
        # initialize the twitch client
        client = TwitchClient()
        client.connect_to_channel() # connect to the channel
    except Exception as e:
        print("Error loading model or client: ", e)
        return
    
    try:
        # initialize the output checker
        output_checker = OutputChecker("all-MiniLM-L6-v2", "bert-base-uncased")
    except Exception as e:
        print("Error loading output checker: ", e)
        return

    # run the thread
    client.run_message_gather()

    # a message has been sent in the last x seconds
    sent_message = False

    # start and end of the current period
    start = datetime.datetime.now()
    end = start + datetime.timedelta(seconds=opt.message_rate)
    
    generated_data = [] # all the data generated by the chatbot
    accepted_data = [] # the data of messages sent
    actual_data = [] # the actual data from the chatlog

    while client.get_channel_status() == True:
        # start the timer for this simulation
        sim_start = datetime.datetime.now()
        
        # the message limit interval has ended, and testing mode not on
        if datetime.datetime.now() > end:
            sent_message = False
            start = datetime.datetime.now()
            end = start + datetime.timedelta(seconds=opt.message_rate)
        elif opt.testing == True:
            sent_message = False

        # record the audio and save it
        try:
            audio_pipe.audio_slice()
        except Exception as e:
            print("Error slicing the audio: ", e)
            continue

        # convert to text
        audio_text = audio_pipe.transcribe(opt.output_file)

        # if the repsonse is too short, skip
        if len(audio_text.split(' ')) < 5:
            #print('Skipping... (too short)')
            continue
        
        # interact with chatbot
        responses = chatbot.generate(audio_text)

        # save the data periodically
        generated_data.append({"text": audio_text, "responses": responses})
        save_file(generated_data, filename_all)
        
        # check the relevancy of the response
        scores = output_checker.check_relevancy(audio_text, responses)

        # get the most relevant responses
        possible_responses = [{'response': response, 'score': score} for response, score in zip(responses, scores) if score > opt.relevancy_score]
        # get the most coherent responses
        coherency = output_checker.check_coherence(audio_text, [response['response'] for response in possible_responses])
        max_idx = np.argmax(coherency)
        response = possible_responses[max_idx]['response']
        max_score = possible_responses[max_idx]['score']

        # get the max score and response
        #max_score = np.max(scores)
        #response = responses[np.argmax(scores)]

        # message has not been sent in last x seconds and the score is high enough
        if sent_message==False:
            print(f'Delay: {datetime.datetime.now() - sim_start}')
            print(f"User: {audio_text}")
            print(f"Chatbot: {response}")
            print(f"Score: {max_score}")
            #client.send_message(response)
            sent_message = True
            accepted_data.append({"text": audio_text, "response": response, "score": str(max_score)})
        elif max_score >= 0.5:
            print(f'Delay: {datetime.datetime.now() - sim_start}')
            print(f"User: {audio_text}")
            print(f"Chatbot: {response}")
            print(f"Score: {max_score}")
            #client.send_message(response)
            accepted_data.append({"text": audio_text, "response": response, "score": str(max_score)})
        else:
            #print("Skipping... (low score)")
            pass

        # save the accepted data periodically
        save_file(accepted_data, filename_accepted)

        # get the message window
        chatlog = client.get_messages_window(window=opt.duration)
        
        # save the message and chatlog pair
        actual_data.append({"text": audio_text, "chatlog": chatlog})
        save_file(actual_data, filename_actual)

        # exit if key board is pressed
        if keyboard.is_pressed('q'):
            break

    # remove the output file and the recording file
    if os.path.exists(opt.output_file):
        os.remove(opt.output_file)
    # close the threads
    client.stop_message_gather()
    print("Threads have been stopped.")
if __name__ == '__main__':
    main()