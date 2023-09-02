"""
AinuralBeat class
Creates a type of binural beat with music gen AI

Using the melody model
"""

import os
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import uuid

class AinuralBeat:
    def __init__(self, beat_type, duration):
        self.valid_types = ["sleep", "meditate", "relax"]
        
        data_path = os.path.abspath("data/")

        if beat_type not in self.valid_types:
            print(f"{beat_type} not a valid type")
            raise AttributeError
        else:
            self.beat_type = beat_type
        
        self.duration = duration
        self.model = None

        self.descriptions = {
            "relax": ["Relax, unwind, down tempo, loop, quiet, binaural beats, 1.8hz range, high quality sound, deep, low bpm, heart beat, low energy, chill"],
            "meditation": ['Meditation, loop, focused, low tempo, soft, singing bowls,  introspective, thinking, slow, high quality sound, deep, low bpm, heart beat'],
            "sleep": ['Sleep, rest, night time, loop, down tempo, quiet, binaural beats, 1.8hz range, high quality sound, deep, low bpm, heart beat, low energy, chill']
        }
        
        self.beat_examples = {
            "sleep": "./assets/sleepcut.mp3",
            "meditate": "./assets/bowlmeditate.mp3",
            "relax": "./assets/relaxcut.mp3"
        }
        
        # check if there is a data folder and if not make one
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        file_id = str(uuid.uuid4).replace("-", "")
        self.output_file = f"{data_path}/{beat_type}_{file_id}.wav"

    def generate_beat(self):
        """
        Generate music of valid beat type with MusicGen
        """
        print("Generating beat...")
        try:
            self.model = MusicGen.get_pretrained('melody')
            self.model.set_generation_params(duration=60)  
            wav = self.model.generate(self.descriptions[self.beat_type])

            waveform, sample_rate = torchaudio.load(
                self.beat_examples[self.beat_type]
            )

            # expand depends on the number of descriptions and melody matching up
            # waveform[None].expand(1, -1, -1) 
            # 1 description = 1 in the x or i of the tensor
            wav = self.model.generate_with_chroma(
                self.descriptions, 
                waveform[None].expand(1, -1, -1), 
                sample_rate
            )

            for one_wav in wav:
                # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
                audio_write(
                    self.output_file, 
                    one_wav.cpu(), 
                    self.model.sample_rate, 
                    strategy="loudness", 
                    loudness_compressor=True
                )
            
            print(f"Beat of type {self.beat_type} has been generated @ {self.output_file}")
        except Exception as err:
            print(f"Error generating beat: {err}")
            raise err
        
