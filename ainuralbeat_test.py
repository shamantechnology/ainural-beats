"""
Test of class AinuralBeat
"""
import os
import unittest
from ainuralbeat import AinuralBeat

class TestMusicGenGPU(unittest.TestCase):
    def test_relax_beat(self):
        abobj = AinuralBeat("relax", 60)
        try:
            abobj.generate_beat()
        except Exception as err:
            raise err

        self.assertTrue(os.path.exists(abobj.output_file))

    def test_sleep_beat(self):
        abobj = AinuralBeat("sleep", 60)
        try:
            abobj.generate_beat()
        except Exception as err:
            raise err

        self.assertTrue(os.path.exists(abobj.output_file))
    
    def test_meditate_beat(self):
        abobj = AinuralBeat("meditate", 60)
        try:
            abobj.generate_beat()
        except Exception as err:
            raise err

        self.assertTrue(os.path.exists(abobj.output_file))

if __name__=='__main__':
	unittest.main()