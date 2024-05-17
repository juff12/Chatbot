import torch
from transformers import pipeline
import os
import time
import datetime
import keyboard
from src.chatbot import Chatbot, AudioPipeline, TwitchClient
import argparse
from sentence_transformers import SentenceTransformer, util

sent_model = SentenceTransformer("all-MiniLM-L6-v2")

def args():
    parser = argparse.ArgumentParser(description='Live Audio Chatbot')
    parser.add_argument('--message_rate', type=int, default=15, help='The rate at which new messages are sent')
    parser.add_argument('--prompt_format', type=str, default='mistral', help='The format of the model [llama/mistral]')
    parser.add_argument('--tokenizer_name', type=str, default='mistralai/Mistral-7B-Instruct-v0.2', help='The base model')
    parser.add_argument('--model_name', type=str, default='', help='The new model')
    parser.add_argument('--asr_model', type=str, default='openai/whisper-base.en', help='The ASR model to use')
    parser.add_argument('--freq', type=int, default=44100, help='The frequency of the audio')
    parser.add_argument('--duration', type=int, default=10, help='The duration of the audio')

    parser.add_argument('--temp_file', type=str, default='data/audio/temp.wav', help='The temp file to hold the wav file while slicing')
    parser.add_argument('--output_file', type=str, default='data/audio/output.wav', help='The output file to use')
    parser.add_argument('--input_file', type=str, default='', help='The input file to use [blank is default]')

    return parser.parse_args()


def output_checker(input, output):
    # check if the output is related to the input
    input_embedding = sent_model.encode(input)
    output_embedding = sent_model.encode(output)
    sim = util.cos_sim(input_embedding, output_embedding)
    # if the similarity is greater than 0.25, then it is related
    if sim > 0.3:
        return True
    return False

def main():
    # get cli args
    opt = args()
    
    # set the device for the transcriber
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # initialize the transcriber
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=opt.asr_model,
        chunk_length_s=30,
        device=device
    )
    try:
        # initialize the chatbot
        chatbot = Chatbot(opt.model_name, opt.tokenizer_name, device, opt.prompt_format)
        # initialize the twitch client
        client = TwitchClient()
        client.connect_to_channel() # connect to the channel
    except Exception as e:
        print(e)
        return
    
    if opt.input_file == '':
        # use the first file in the directory
        input_file = [os.path.join('data/audio', file) for file in os.listdir('data/audio') if file.endswith('.mkv')][0]
    else:
        input_file = opt.input_file

    # initialize the audio pipeline
    audio_pipe = AudioPipeline(input_file, opt.temp_file, opt.output_file,
                               freq=opt.freq, duration=opt.duration)

    run_process = True

    while run_process:
        print("Running...")
        # start and end of the current period
        start = datetime.datetime.now()
        end = start + datetime.timedelta(seconds=opt.message_rate)

        # record the audio and save it
        try:
            audio_pipe.audio_slice()
        except Exception as e:
            print(e)
            continue

        # convert to text
        audio_text = transcriber(opt.output_file)['text']

        # if the repsonse is too short, skip
        if len(audio_text) < 5:
            print('Skipping...')
            continue

        # interact with chatbot
        response = chatbot.generate(audio_text)

        # example
        if output_checker(audio_text, response):
            print('Delay', datetime.datetime.now() - start)
            print(f"User: {audio_text}")
            print(f"Chatbot: {response}")
            client.send_message(response)
        else:
            print("Skipping...")

        # exit if key board is pressed
        if keyboard.is_pressed('q'):
            break

        # run every x seconds
        while datetime.datetime.now() < end:
            # exit if key board is pressed
            if keyboard.is_pressed('q'):
                run_process = False
                break
            time.sleep(1)
    
    # remove the output file and the recording file
    os.remove(opt.output_file)
    os.remove(input_file)

if __name__ == "__main__":
    main()