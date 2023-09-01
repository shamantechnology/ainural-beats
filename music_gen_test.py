"""
Test MusicGen model 
"""
import os
import unittest
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

class TestMusicGenGPU(unittest.TestCase):
    
    def test_relax_beat(self):
        model = MusicGen.get_pretrained('melody')
        model.set_generation_params(duration=60)  
        descriptions = ['Relax, unwind, down tempo, loop, quiet, binaural beats, 1.8hz range, high quality sound, deep, low bpm, heart beat, low energy, chill']
        wav = model.generate(descriptions)

        waveform, sample_rate = torchaudio.load('../assets/relaxcut.mp3')

        # expand depends on the number of descriptions and melody matching up
        # waveform[None].expand(1, -1, -1) 1 description = 1 in the x or i of the tensor
        wav = model.generate_with_chroma(descriptions, waveform[None].expand(1, -1, -1), sample_rate)

        for one_wav in wav:
            # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
            audio_write('../data/relax_test', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

        self.assertTrue(os.path.exists('../data/relax_test.wav'))
    
    def test_meditation_beat(self):
        model = MusicGen.get_pretrained('melody')
        model.set_generation_params(duration=60)  
        descriptions = ['Meditation, loop, focused, low tempo, soft, singing bowls,  introspective, thinking, slow, high quality sound, deep, low bpm, heart beat']
        wav = model.generate(descriptions)

        waveform, sample_rate = torchaudio.load('../assets/bowlmeditate.mp3')

        # expand depends on the number of descriptions and melody matching up
        # waveform[None].expand(1, -1, -1) 1 description = 1 in the x or i of the tensor
        wav = model.generate_with_chroma(descriptions, waveform[None].expand(1, -1, -1), sample_rate)


        for one_wav in wav:
            # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
            audio_write('../data/meditate_test2', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

        self.assertTrue(os.path.exists('../data/meditate_test2.wav'))

    def test_sleep_beat(self):
        model = MusicGen.get_pretrained('melody')
        model.set_generation_params(duration=60)  
        descriptions = ['Sleep, rest, night time, loop, down tempo, quiet, binaural beats, 1.8hz range, high quality sound, deep, low bpm, heart beat, low energy, chill']
        wav = model.generate(descriptions)

        waveform, sample_rate = torchaudio.load('../assets/sleeptoo.mp3')

        # expand depends on the number of descriptions and melody matching up
        # waveform[None].expand(1, -1, -1) 1 description = 1 in the x or i of the tensor
        wav = model.generate_with_chroma(descriptions, waveform[None].expand(1, -1, -1), sample_rate)

        for one_wav in wav:
            # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
            audio_write('../data/sleep_test2', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

        self.assertTrue(os.path.exists('../data/sleep_test2.wav'))

if __name__=='__main__':
	unittest.main()