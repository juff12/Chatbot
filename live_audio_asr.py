import torch
from transformers import pipeline
import os
import time
import datetime
import keyboard
from src.chatbot import Chatbot, AudioPipeline
import argparse

def args():
    parser = argparse.ArgumentParser(description='Live Audio Chatbot')
    parser.add_argument('--message_rate', type=int, default=30, help='The rate at which new messages are sent')
    parser.add_argument('--prompt_format', type=str, default='llama', help='The format of the model [llama/mistral]')
    parser.add_argument('--base_model', type=str, default='NousResearch/Llama-2-7b-hf', help='The base model')
    parser.add_argument('--trained_model', type=str, default='', help='The new model')
    parser.add_argument('--asr_model', type=str, default='openai/whisper-base.en', help='The ASR model to use')
    parser.add_argument('--freq', type=int, default=44100, help='The frequency of the audio')
    parser.add_argument('--duration', type=int, default=10, help='The duration of the audio')

    parser.add_argument('--temp_file', type=str, default='data/audio/temp.wav', help='The temp file to hold the wav file while slicing')
    parser.add_argument('--output_file', type=str, default='data/audio/output.wav', help='The output file to use')
    parser.add_argument('--input_file', type=str, default='', help='The input file to use [blank is default]')


    return parser.parse_args()

def main():
    # get cli args
    args = args()

    # set the device for the transcriber
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # initialize the transcriber
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=args.asr_model,
        chunk_length_s=30,
        device=device
    )

    # initialize the chatbot
    chatbot = Chatbot(args.base_model, args.trained_model,
                      device=device, format=args.prompt_format)

    if args.input_file == '':
        # use the first file in the directory
        input_file = [os.path.join('data/audio', file) for file in os.listdir('data/audio') if file.endswith('.mkv')][0]
    else:
        input_file = args.input_file

    
    audio_pipe = AudioPipeline(input_file, args.temp_file, args.output_file,
                               freq=args.freq, duration=args.duration)


    run_process = True

    while run_process:
        # start and end of the current period
        start = datetime.datetime.now()
        end = start + datetime.timedelta(seconds=args.message_rate)

        # record the audio and save it
        try:
            audio_pipe.audio_slice()
        except Exception as e:
            print(e)
            continue

        # convert to text
        audio_text = transcriber(args.output_file)['text']

        # interact with chatbot
        response = chatbot.generate(audio_text)

        # example
        print(f"User: {audio_text}")
        print(f"Chatbot: {response}")
        
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
    os.remove(args.output_file)
    os.remove(input_file)

if __name__ == "__main__":
    main()