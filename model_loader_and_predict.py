import numpy as np


def tetsting_unit():
    tester = []
    import librosa
    test, ans = librosa.load("D:/Dev_stuff/audio_dataset/rachit2.wav")  # provide path of  wave file
    mfccs = np.mean(librosa.feature.mfcc(test, ans, n_mfcc=40).T, axis=0)
    tester.append(mfccs)
    tester = np.array(tester)
    return tester #return Mfcss extracted arrray 


import pickle  # importing pickle to load saved model

load_model = pickle.load(open('phase1_model.sav', 'rb'))  # loading phase_1 model (noise vs speech)
result = load_model.predict(tetsting_unit())  # predicting if result[0]==1 then noise else human sound
load_model2 = pickle.load(open('phase2_model.sav', 'rb'))  # loading phase2 model 
if result[0] == 2:  # checking sound noise or human
    print("Phase-1 clear")
    ok = load_model2.predict(tetsting_unit())  # using second phase_model
    if ok[0] == 1:
        print("Phase-2 clear")
        print('Scream')
    else:
        print('speech')
else:
    print("noise")
