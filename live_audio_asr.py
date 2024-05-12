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
    parser.add_argument('--model_format', type=str, default='llama', help='The format of the model [llama/mistral]')
    parser.add_argument('--base_model', type=str, default='NousResearch/Llama-2-7b-hf', help='The base model')
    parser.add_argument('--trained_model', type=str, default='', help='The new model')
    parser.add_argument('--asr_model', type=str, default='openai/whisper-base.en', help='The ASR model to use')

    return parser.parse_args()


def main():
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

    chatbot = Chatbot(args.base_model, args.trained_model, device=device)

    # get the '.mkv' file
    input_file = [os.path.join('data/audio', file) for file in os.listdir('data/audio') if file.endswith('.mkv')][0]
    # set the temp and output files
    temp_file = 'data/audio/temp.wav'
    output_file = 'data/audio/output.wav'
    
    audio_pipe = AudioPipeline(input_file, temp_file, output_file)


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
        audio_text = transcriber(output_file)['text']

        # interact with chatbot
        response = chatbot.generate(audio_text, args.model_format)

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
    os.remove(output_file)
    os.remove(input_file)

if __name__ == "__main__":
    main()