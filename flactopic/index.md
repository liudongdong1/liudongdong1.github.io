# FlacToPic


#### 1. flactoPic

```python
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from librosa import display as ld

def drawSpecgramFlac(filename,savefilename):
    '''
        绘制语谱图，使用plt.specgram 功能
    '''
    data, samplerate = sf.read(filename)
    times = np.linspace(0,len(data),len(data))/samplerate
    plt.clf()
    plt.plot(times,data)
    Pxx, freqs, bins, im = plt.specgram(data, Fs=samplerate)
    plt.xlim(44,46)
    plt.ylim(2000,6000)
    plt.axis('off')
    
    plt.savefig('{}.png'.format(savefilename),dpi=1080)

def drawSpecgramFlacV1(filename,savefilename):
    '''
        绘制语谱图，使用librosa.stft 相关库
    '''
    data, samplerate = sf.read(filename)
    times = np.linspace(0,len(data),len(data))/samplerate
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    ld.specshow(Xdb, sr=samplerate, x_axis='time', y_axis='hz')
    #plt.colorbar()
    #plt.show()
    plt.axis('off')
    plt.savefig('{}.jpg'.format(savefilename))
    plt.clf()
#drawSpecgramFlacV1("/home/zkx/data/trainSmall/9b532eb80.flac","/home/zkx/data/trainSmallPng/9b532eb80.flac")
#drawSpecgramFlac("/home/zkx/data/trainSmall/9b532eb80.flac","/home/zkx/data/trainSmallPng/9b532eb80.flac1")
def generatePng(basefolder,targetfolder):
    for file in os.listdir(basefolder):
        if file.find("jpg") !=-1:
            continue
        filename=os.path.join(basefolder,file)
        drawSpecgramFlac(filename,os.path.join(targetfolder,file))
#generatePng("/home/zkx/data/trainSmall","/home/zkx/data/trainSmallPng")
```

```python
import pandas as pd
import numpy as np
df=pd.read_csv('/home/zkx/data/train_tp.csv') #filename可以直接从盘符开始，标明每一级的文件夹直到csv文件，header=None表示头部为空，sep=' '表示数据间使用空格作为分隔符，如果分隔符是逗号，只需换成 ‘，’即可。
print(df.head())
print(df.shape)
print(df.columns.values)
print(df['recording_id'][0])
print(df['recording_id'][1])
df.shape[0]
for i in range(0,df.shape[0]):
    if df['species_id'][i]>9:
        continue
    basefolder=os.path.join("/home/zkx/data/pngdataset",str(df['species_id'][i]))
    #print(basefolder)
    #if not os.path.exits(basefolder):
    #    os.mkdirs(basefolder)
    try:
        strcom='scp zkx@192.168.2.56:/home/zkx/data/train/'+df['recording_id'][i]+'.flac'+" /home/zkx/data/temp/"+str(df['species_id'][i])
        print(strcom)
        if os.system(strcom)!=0:
            continue
        #print("/home/zkx/data/temp/"+str(df['species_id'][i])+"/"+df['recording_id'][i]+'.flac',basefolder+"/"+df['recording_id'][i])
        drawSpecgramFlac("/home/zkx/data/temp/"+str(df['species_id'][i])+"/"+df['recording_id'][i]+'.flac',basefolder+"/"+df['recording_id'][i])
    except Exception as e:
        print("异常报错：",e)
```



---

> 作者: liudongdong1  
> URL: https://liudongdong1.github.io/flactopic/  

