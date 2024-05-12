import subprocess
import sounddevice as sd
import wavio as wv
import os

class AudioPipeline():
    def __init__(self, infile, tempfile='data/audio/temp.wav', outfile='data/audio/output.wav', freq=44100, duration=5):
        self.infile = infile
        self.tempfile = tempfile
        self.outfile = outfile
        self.freq = freq
        self.duration = duration
    
    def audio_recording_microphone(self):
        print("Recording...")
        # start recording the audio
        recording = sd.rec(int(self.freq * self.duration),
                        samplerate=self.freq, channels=2)
        sd.wait() # wait until the recording is finished
        # save the recording
        wv.write(self.outfile, recording, self.freq, sampwidth=2)

    # grab a slice of the audio
    def audio_slice(self):
        # get the current audio as a wav file, save in temp location
        command = ['ffmpeg', '-y', '-i', self.infile, '-vn', '-acodec',
                   'pcm_s16le', '-ar', f'{self.freq}', '-ac', '2', self.tempfile]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        # get the last 10 seconds of the audio
        command = ['ffmpeg', '-sseof', f'-{self.duration}', '-y', '-i', self.tempfile,
                   '-vn', '-acodec', 'pcm_s16le', '-ar', f'{self.freq}', '-ac', '2', self.outfile]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        # remove the temp file
        os.remove(self.tempfile)
