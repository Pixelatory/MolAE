import selfies as sf
import random

alphabet=sf.get_semantic_robust_alphabet() # Gets the alphabet of robust symbols
print(alphabet)
rnd_selfies=''.join(random.sample(list(alphabet), 9))
rnd_smiles=sf.decoder(rnd_selfies)
print(rnd_smiles)