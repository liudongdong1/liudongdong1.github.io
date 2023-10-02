# CharDataSimulation


### 1. 图片生成

> 利用电脑中字体文件生成a-z 字符图片数据。

```python
# 测试文件
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import matplotlib.pyplot as plt
import numpy as np
import os
# 随机字母:
def rndChar():
    return chr(random.randint(65, 90))
# 随机颜色1:
def rndColor():
    return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))
# 随机颜色2:
def rndColor2():
    return (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))

letters='abcdefghiJKLmnopqrstuvwxyzCFIJKLOPSUVWXYZ'
str = [letters[i] for i in range(0,len(letters))]
num = len(str)
print(letters,num)
width = 64*num
height = 64
step_size = 64
for file in os.listdir('C:/windows/fonts/'):
    try:
        tmp=os.path.join('C:/windows/fonts/',file)
        print(tmp)
        # 创建Image
        image = Image.new('RGB', (width, height), (255, 255, 255))
        #选择字体
        font = ImageFont.truetype(tmp, 80)
        # 创建Font对象:
        # 创建Draw对象:
        draw = ImageDraw.Draw(image)
        # 输出文字:
        for t in range(num):
            draw.text((step_size * t, 0), str[t], font=font, fill=(0,0,0))
        # 模糊:
        # image = image.filter(ImageFilter.BLUR)
        image.save("./png/"+file+'.jpg','jpeg')
    except Exception as err:
        pass
```

```python
# 生成原始数据集
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import matplotlib.pyplot as plt
import numpy as np
import os

#数据集根目录
rootpath="./data/"
#windows上字体文件路径
fontpath='C:/windows/fonts/'
#图片旋转角度
rotateAngel=[-20,-15,-10,-5,5,10,15,20]
#生成图片大小
width = 96
height = 96
#需要生成得字体字符  abcdefghIJkLmnopqrstUVWxYZ
letters='w'
CharData = [letters[i] for i in range(0,len(letters))]

# 黑色，用户显示字体
def blackColor():
    return (0,0,0)

# 根据data数据生成目录文件,data=[a,b,c,d,] 这种字符数列
def mkdir_for_imgs(data):
    for tm in data:
        if os.path.isdir(rootpath+tm):
            pass
        else:
            print(rootpath+tm)
            os.makedirs(rootpath+tm)


#罗列和手势分开字符比较相似的不同font 名称
def getFontList(fontpath):
    listfontnames=os.listdir(fontpath)
    return [name[:-4] for name in listfontnames]
#电脑上总共有663个字体返回的有效字体199个  ['50416___.TTF', '65659___.TTF', '75678___.TTF', '75749___.TTF', 'AGENCYR.TTF', 'ANTQUAB.TTF', 'ANTQUABI.TTF', 'ANTQUAI.TTF', 'arial.ttf', 'ariali.ttf', 'ARIALN.TTF', 'ARIALNB.TTF', 'ARIALNBI.TTF', 'ARIALNI.TTF', 'ARIALUNI.TTF', 'bahnschrift.ttf', 'BASKVILL.TTF', 'BELL.TTF', 'BELLB.TTF', 'BELLI.TTF', 'BKANT.TTF', 'BOD_B.TTF', 'BOD_BI.TTF', 'BOD_CI.TTF', 'BOD_I.TTF', 'BOD_R.TTF', 'BOOKOS.TTF', 'BOOKOSI.TTF', 'BRADHITC.TTF', 'calibri.ttf', 'calibrib.ttf', 'calibrii.ttf', 'calibril.ttf', 'calibrili.ttf', 'CALIFI.TTF', 'CALIFR.TTF', 'CALIST.TTF', 'CALISTB.TTF', 'CALISTBI.TTF', 'CALISTI.TTF', 'cambriai.ttf', 'Candara.ttf', 'Candarab.ttf', 'Candarai.ttf', 'Candaral.ttf', 'Candarali.ttf', 'Candaraz.ttf', 'CENSCBK.TTF', 'CENTAUR.TTF', 'CENTURY.TTF', 'comic.ttf', 'comici.ttf', 'comicz.ttf', 'consola.ttf', 'consolab.ttf', 'consolai.ttf', 'consolaz.ttf', 'constan.ttf', 'constanb.ttf', 'constani.ttf', 'corbel.ttf', 'corbelb.ttf', 'corbeli.ttf', 'corbell.ttf', 'corbelli.ttf', 'cour.ttf', 'courbd.ttf', 'courbi.ttf', 'couri.ttf', 'Deng.ttf', 'Dengb.ttf', 'Dengl.ttf', 'DUBAI-LIGHT.TTF', 'DUBAI-MEDIUM.TTF', 'DUBAI-REGULAR.TTF', 'ebrima.ttf', 'ebrimabd.ttf', 'ERASDEMI.TTF', 'ERASLGHT.TTF', 'ERASMD.TTF', 'FRABK.TTF', 'FRABKIT.TTF', 'FRADMCN.TTF', 'FRADMIT.TTF', 'FRAMDCN.TTF', 'framdit.ttf', 'FREESCPT.TTF', 'FTLTLT.TTF', 'FZSTK.TTF', 'FZYTK.TTF', 'Gabriola.ttf', 'GARA.TTF', 'GARABD.TTF', 'GARAIT.TTF', 'georgia.ttf', 'georgiai.ttf', 'GILC____.TTF', 'GILI____.TTF', 'GIL_____.TTF', 'GLECB.TTF', 'GOTHICI.TTF', 'GOUDOS.TTF', 'GOUDOSI.TTF', 'himalaya.ttf', 'HTOWERT.TTF', 'HTOWERTI.TTF', 'INFROMAN.TTF', 'Inkfree.ttf', 'LBRITED.TTF', 'LBRITEDI.TTF', 'LBRITEI.TTF', 'LEELAWAD.TTF', 'LEELAWDB.TTF', 'LeelawUI.ttf', 'LeelUIsl.ttf', 'LFAX.TTF', 'LFAXD.TTF', 'LFAXI.TTF', 'LSANSDI.TTF', 'LSANSI.TTF', 'LTYPE.TTF', 'LTYPEB.TTF', 'LTYPEBO.TTF', 'LTYPEO.TTF', 'lucon.ttf', 'MAIAN.TTF', 'malgun.ttf', 'malgunbd.ttf', 'malgunsl.ttf', 'mingliub.ttc', 'mmrtext.ttf', 'MOD20.TTF', 'monbaiti.ttf', 'msgothic.ttc', 'msjh.ttc', 'msjhl.ttc', 'MSUIGHUB.TTF', 'MSUIGHUR.TTF', 'msyhl.ttc', 'msyi.ttf', 'Nirmala.ttf', 'NirmalaS.ttf', 'ntailu.ttf', 'OCRAEXT.TTF', 'palabi.ttf', 'palai.ttf', 'PERI____.TTF', 'PER_____.TTF', 'phagspa.ttf', 'REFSAN.TTF', 'ROCKI.TTF', 'SCHLBKI.TTF', 'segoeuii.ttf', 'segoeuil.ttf', 'segoeuisl.ttf', 'seguiemj.ttf', 'seguihis.ttf', 'seguili.ttf', 'seguisb.ttf', 'seguisbi.ttf', 'seguisli.ttf', 'seguisym.ttf', 'simfang.ttf', 'simhei.ttf', 'simkai.ttf', 'SIMLI.TTF', 'simsun.ttc', 'simsunb.ttf', 'SIMYOU.TTF', 'Sitka.ttc', 'SitkaI.ttc', 'STFANGSO.TTF', 'STKAITI.TTF', 'STSONG.TTF', 'STXINWEI.TTF', 'STZHONGS.TTF', 'sylfaen.ttf', 'tahoma.ttf', 'taile.ttf', 'TCCM____.TTF', 'TCMI____.TTF', 'TCM_____.TTF', 'times.ttf', 'timesbi.ttf', 'timesi.ttf', 'trebucit.ttf', 'tt0142m_.ttf', 'tt0143m_.ttf', 'tt0200m_.ttf', 'tt0372m_.ttf', 'tt0395m_.ttf', 'tt0849m_.ttf', 'TT1139M_.TTF', 'tt2002m_.ttf', 'verdana.ttf', 'verdanai.ttf', 'YuGothL.ttc', 'YuGothM.ttc', 'YuGothR.ttc']

#fontname: 字体得名称
# chardata: 需要生成图片的字符
# picturename： 生成图片的名字
#outputdir: 生成图片存储目录
# 字符居中，生成单张图片
def generateBlackPicuture(fontname,chardata,picturename,outputdir):
    try:
        fullfontpath=os.path.join(fontpath,fontname)
        # 创建Image
        image = Image.new('RGB', (width, height), (255, 255, 255))
        #选择字体
        font = ImageFont.truetype(fullfontpath, 80)
        # 创建Font对象:
        # 创建Draw对象:
        draw = ImageDraw.Draw(image)
        #设置字体居中对齐
        imwidth, imheight = image.size
        font_width, font_height = draw.textsize(chardata, font)
        draw.text(((imwidth - font_width-font.getoffset(chardata)[0]) / 2, (imheight - font_height-font.getoffset(chardata)[1]) / 2),chardata,font=font,fill=blackColor())
        #image = image.filter(ImageFilter.BLUR)
        image.save(os.path.join(outputdir,picturename+'.png'))  
        #进行旋转操作并进行处理
       # rotateImage(outputdir,picturename+'.png',outputdir)

        # 模糊:
        # image = image.filter(ImageFilter.BLUR)
    except Exception as err:
        pass
   
#图片旋转操作
#inputdir: the inputdir of dataset
#filename: the name of picture to handle
# outputdir: the name of outputdir to save rotated file
def rotateImage(inputdir,filename,outputdir):
    # original image 
    img = Image.open(os.path.join(inputdir,filename))
    # converted to have an alpha layer 
    im2 = img.convert('RGBA') 
    for i in range(0,24,8):
        savepath=os.path.join(outputdir,filename+str(i) +'.png')
        # rotated image 
        rot = im2.rotate(i, expand=1) 
        # a white image same size as rotated image 
        fff = Image.new('RGBA', rot.size, (255,)*4) 
        # create a composite image using the alpha layer of rot as a mask 
        out = Image.composite(rot, fff, rot) 
        out=out.resize((96,96),Image.ANTIALIAS)
        # save your work (converting back to mode='1' or whatever..) 
        out.convert(img.mode).save(savepath) 
    for i in range(336,360,8):
        savepath=os.path.join(outputdir,filename+str(i)+'.png')
            # rotated image 
        rot = im2.rotate(i, expand=1) 
        # a white image same size as rotated image 
        fff = Image.new('RGBA', rot.size, (255,)*4) 
        # create a composite image using the alpha layer of rot as a mask 
        out = Image.composite(rot, fff, rot) 
        out=out.resize((96,96),Image.ANTIALIAS)
        # save your work (converting back to mode='1' or whatever..) 
        out.convert(img.mode).save(savepath) 
#生成单个字符数据集
# charname： 字符名称
#charnamedir: 字符数据集存储对应的文件夹目录
def generateSingleCharDataset(charname,charnamedir):
    for fonttemp in getFontList("./png"):
        generateBlackPicuture(fonttemp,charname,fonttemp.split('.')[0],charnamedir)

#生成所有字符数据集
def generateAlldata():
    mkdir_for_imgs(CharData)
    for charname in CharData:
        generateSingleCharDataset(charname,os.path.join(rootpath,charname))

def generateAllRotatePicture(rootdir):
    for dirname in os.listdir(rootdir):
        tempdir=os.path.join(rootdir,dirname)
        filenames=os.listdir(tempdir)
        for filename in filenames:
            rotateImage(tempdir,filename,tempdir)


if __name__ == "__main__":
   #generateAllRotatePicture("./data/")
   generateAlldata()
```



### 2. label generate

> 读取标注xml文件，绘制label 图像文件。

```python
import xml.etree.ElementTree as ET
import os
import sys
from xml.dom.minidom import parse
import numpy as np
import cv2
def readPointXML(filepath):
    domTree = parse(filepath)
    # 文档根元素
    rootNode = domTree.documentElement
    objects = rootNode.getElementsByTagName("object")
    #print("objects=",objects)
    point=[]
    for item in objects:
        bndboxs=item.getElementsByTagName('bndbox')
        #print("bndbox",bndboxs)
        if bndboxs and len(bndboxs)>0:
            xmin=(int)(bndboxs[0].getElementsByTagName("xmin")[0].firstChild.data)
            ymin=(int)(bndboxs[0].getElementsByTagName("ymin")[0].firstChild.data)
            xmax=(int)(bndboxs[0].getElementsByTagName("xmax")[0].firstChild.data)
            ymax=(int)(bndboxs[0].getElementsByTagName("ymax")[0].firstChild.data)
            point.append((int((xmin+xmax)/2),int((ymin+ymax)/2)))
    return point

def dealsingleImage():
    points=readPointXML('./calibrili.xml')
    print(points)
    temp=np.zeros([96,96,3])
    for point in points:
        temp[point[1],point[0]]=[255,255,255]
    cv2.imwrite("11.png",temp)
    
#path="H:\chardigitDataset\mnist\characterdigit\codes\data\xtfk\train\data\\"  saverootdir="H:\chardigitDataset\mnist\characterdigit\codes\data\xtfk\train\label"
def generateLabel(path,saverootdir):
    for charfoder in os.listdir(path):
        print(charfoder)
        tempfolder=os.path.join(path,charfoder)
        tempsavefolder=os.path.join(saverootdir,charfoder)
        if not os.path.exists(tempsavefolder):
             os.makedirs(tempsavefolder)
        for filename in os.listdir(tempfolder):
            if filename[-1]=="l":
                points=readPointXML(os.path.join(tempfolder,filename))
                temp=np.zeros([96,96,3])
                for point in points:
                    temp[point[1],point[0]]=[255,255,255]
                savefile=os.path.join(tempsavefolder,filename.split(".")[0]+".png")
                print(savefile)
                cv2.imwrite(savefile,temp)


if __name__ == "__main__":
    generateLabel("H:\\chardigitDataset\\mnist\\characterdigit\\codes\\data\\xtfk\\train\\data\\","H:\\chardigitDataset\\mnist\\characterdigit\\codes\\data\\xtfk\\train\\label\\")
    dealsingleImage()
```

> 针对k,t,x,f 这四个字符当提取出来关键点之后，使用一次函数将关键点连接

```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

#根据字符的特性，最上面点，最上面右边点， 根据y的距离大小 来过滤相邻的点

#可视化显示某张字符检测效果
def CornerPointDetectionTest(imgpath):
    img=cv2.imread(imgpath)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    corners=cv2.goodFeaturesToTrack(gray,10,0.3,6)
    corners=np.int0(corners)
    corners=np.array([[i[0][0],i[0][1]] for i in corners])
    print(corners)
    corners=corners[corners[:,0].argsort()]
    print(corners)
    for x in corners:
        cv2.circle(img,(x[0],x[1]),2,255,-1)
    #cv2.circle(img,(4,8),2,255,-1)   图片左上角 （0，0）， 行x，列y
    plt.imshow(img)
    plt.show()


def getCornerPoint(imgpath):
    img=cv2.imread(imgpath)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    corners=cv2.goodFeaturesToTrack(gray,30,0.2,2)
    corners=np.int0(corners)
    corners=np.array([[i[0][0],i[0][1]] for i in corners])
    print("orgin",corners)
    pointA,pointB=[0,0],[0,0]
    temp=corners[corners[:,1].argsort()]
    ymin=temp[0][1]
    ymax=temp[len(temp)-1][1]
    steplen=(ymax-ymin)/3
    print("y order",temp,ymin,ymax,steplen)
    corners=corners[corners[:,0].argsort()][::-1]
    print("orderx",corners)
    a,b=-1,-1
    pointA=corners[0]

    temp=corners[corners[:,1].argsort()][::-1]
    corners=corners[corners[:,0].argsort()][::-1]
    pointT=corners[0]
    pointMiddle=[int((pointA[0]+pointT[0])/2),int((pointA[1]+pointT[1])/2)]
    for t in corners[1:]:
        if abs(t[1]-pointMiddle[1]) < steplen:
            pointB=t
            break
        # if t[1]>ymin-1 and t[1]<ymin+steplen and a==-1:
        #     pointA=t
        #     a=1
        # elif t[1]>=ymin+steplen and t[1]<ymin+2*steplen and b==-1:
        #     pointB=t
        #     b=1
        # else:
        #     pass
    return pointA,pointB

#曲线拟合为 y=ax+b
def handleCurve(rootdir,picpath):
    imgpath=os.path.join(rootdir,picpath)
    print(imgpath)
    corners=getCornerPoint(imgpath)
    img=cv2.imread(imgpath)
    print(corners[0],corners[1])
    #cv2.line(img,(corners[0][0],corners[0][1]),(corners[1][0],corners[1][1]),(255,255,255))   #(255,255,255)  白色
    pt=(corners[0][0],corners[1][1])
    tut=np.array([(corners[0][0],corners[0][1]),(pt[0]-1,pt[1]-2),(corners[1][0],corners[1][1])])
    tut=tut.reshape(-1,1,2)
    print(tut)
    cv2.polylines(img,[tut],False,(255,255,255))
    savepath="11"+picpath.split(".")[0]+".png"
    print(savepath)
    cv2.imwrite(savepath,img)
   
#处理一个目录文件夹文件
def handle(rootdir):
    templist=os.listdir(rootdir)
    for file in templist:
        try:
            handleCurve(rootdir,file)
        except Exception as err :
            print(err)


def calculateAB(pointA,pointB):
    a=(pointA[1]-pointB[1])/(pointA[0]-pointB[0])
    b=pointA[1]-a*pointA[0]
    return a,b



if __name__ == "__main__":
    #handle("H:\\chardigitDataset\\mnist\\characterdigit\\codes\\data\\Xihua\\k\\")
    #handleCurve("","./corbel.png")
    #print(getCornerPoint("./calibrili.png"))
    CornerPointDetectionTest("./calibrili.png")
#img=np.array(Image.open("./calibrili.png"))
#index=np.argwhere(img>1)
#print(index)
# plt.figure("lena")
# plt.imshow(img,cmap='gray')
# plt.axis('off')
# plt.show()
```

### 3. 图片旋转细化操作

```python
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
#图片旋转操作
#inputdir: the inputdir of dataset
#filename: the name of picture to handle
# outputdir: the name of outputdir to save rotated file
def rotateImage(inputdir,filename,outputdir):
    # original image 
    img = Image.open(os.path.join(inputdir,filename))
    # converted to have an alpha layer 
    im2 = img.convert('RGBA') 
    for i in range(0,17,8):
        savepath=os.path.join(outputdir,filename+"rot"+str(i) +'.png')
        # rotated image 
        rot = im2.rotate(i, expand=1) 
        # a white image same size as rotated image 
        fff = Image.new('RGBA', rot.size, (255,)*4) 
        # create a composite image using the alpha layer of rot as a mask 
        out = Image.composite(rot, fff, rot) 
        out=out.resize((96,96),Image.ANTIALIAS)
        # save your work (converting back to mode='1' or whatever..) 
        out.convert(img.mode).save(savepath) 
    for i in range(336,360,8):
        savepath=os.path.join(outputdir,filename+"rot"+str(i)+'.png')
            # rotated image 
        rot = im2.rotate(i, expand=1) 
        # a white image same size as rotated image 
        fff = Image.new('RGBA', rot.size, (255,)*4) 
        # create a composite image using the alpha layer of rot as a mask 
        out = Image.composite(rot, fff, rot) 
        out=out.resize((96,96),Image.ANTIALIAS)
        # save your work (converting back to mode='1' or whatever..) 
        out.convert(img.mode).save(savepath) 

def pictureExpand(imgfile, savefilename):
    try:
        img=cv2.imread(imgfile,0)
        kernels=[np.ones((2,2),np.uint8),np.ones((3,3),np.uint8)]
        j=0
        for i in [2,3]:
            for kernel in kernels:
                dilation=cv2.dilate(img,kernel,iterations=i)
                gray =255-dilation
                savefile=savefilename+str(j)+".png"
                print(savefile)
                cv2.imwrite(savefile,gray)
                j=j+1
    except Exception as err:
        print(err)
```

```python
#处理单张图片
def handimage(inputpath,savepath):
    if inputpath.find(".png")==-1:
        return
    img=mpimg.imread(inputpath)
    img=color.rgb2gray(img)
    img=1-img
    # shape=img.shape
    # for i in range(0,shape[0]):
    #     for j in range(0,shape[1]):
    #         if img[i][j]<0.5:
    #             img[i][j]=0
    #         else:
    #             img[i][j]=1
    skeleton=morphology.skeletonize(img)
    imsave(savepath,  img_as_uint(skeleton))
```

### 4. 文件操作

```python
#移动文件 folder 目录格式: data/a/1.xml, pathfromfolder:       
def moveFile():
    pathfromfolder="H:\\chardigitDataset\\mnist\\characterdigit\\codes\\data\\xtfk\\train\\label\\"
    savefromfolder="H:\\chardigitDataset\\mnist\\characterdigit\\codes\\data\\xtfk\\train\\data\\"
    sourcefolder="H:\\chardigitDataset\\mnist\\characterdigit\\codes\\data\\xtfk\\Xihua\\"
    # if not os.path.isdir(pathfromfolder):
    #     print("folder not exit")
    # if not os.path.isfile(pathtofolder):
    #     os.mkdirs(pathtofolder)
    for tmfoler in os.listdir(pathfromfolder):
        temppath=os.path.join(pathfromfolder,tmfoler)
        for filename in os.listdir(temppath):
            tmp=os.path.join(sourcefolder,tmfoler,filename)
            try:
                shutil.move(tmp,os.path.join(savefromfolder,tmfoler,filename))
            except Exception as err :
                print(tmp+" is not exit")

def dataSplit(sourcedir,targetdir):
    for tmfolder in os.listdir(sourcedir):
        sourcechardir=os.path.join(sourcedir,tmfolder)
        traindir=os.path.join(targetdir,"train",tmfolder)
        validdir=os.path.join(targetdir,"valid",tmfolder)
        testdir=os.path.join(targetdir,"test",tmfolder)
        print(sourcechardir)
        if not os.path.exists(traindir):
            os.makedirs(traindir)
        if not os.path.exists(validdir):
            os.makedirs(validdir)
        if not os.path.exists(testdir):
            os.makedirs(testdir)
        i=0
        for filename in os.listdir(sourcechardir):
            if i%10 ==9:
                shutil.move(os.path.join(sourcechardir,filename),os.path.join(testdir,filename))
            elif i%10<7:
                shutil.move(os.path.join(sourcechardir,filename),os.path.join(traindir,filename))
            else:
                shutil.move(os.path.join(sourcechardir,filename),os.path.join(validdir,filename))
            i=i+1
```





---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/chardatasimulation/  

