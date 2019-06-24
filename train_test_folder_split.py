import os
import random
import shutil

def mkdir():
    fileDir = 'CCSN_dataset'
    train_data_Dir = 'train_test/train/'
    test_data_Dir = 'train_test/test/'
    for dir in os.listdir(fileDir):
        if dir == '.git':
            continue
        if os.path.exists(train_data_Dir + dir):
            continue
        else:
            os.mkdir(train_data_Dir + dir)
        if os.path.exists(test_data_Dir + dir):
            continue
        else:
            os.mkdir(test_data_Dir + dir)


def moveFile(fileDir, trainDir, testDir):
        pathDir = os.listdir(fileDir)    #取图片的原始路径
        filenumber = len(pathDir)
        rate = 0.3    #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
        picknumber = int(filenumber*rate) #按照rate比例从文件夹中取一定数量图片
        test = random.sample(pathDir, picknumber)  #随机选取picknumber数量的样本图片
        train = []
        for pic in pathDir:
            if pic in test:
                continue
            else:
                train.append(pic)
        for name in train:
            shutil.move(fileDir+name, trainDir+name)
        for name in test:
            shutil.move(fileDir+name, testDir+name)

if __name__ == '__main__':
    sourcefile = "CCSN_v2/"
    for subDir in os.listdir(sourcefile):
        if subDir == '.git':
            continue
        subDirpath = os.path.join(sourcefile, subDir)
        subDirpath = subDirpath + '/'
        #print(subDirpath + '/')
        trainDir = 'CCSN_dataset/train/'
        testDir = 'CCSN_dataset/test/'
        #print(trainDir+subDir+'/', testDir+subDir+'/')
        moveFile(subDirpath, trainDir+subDir+'/', testDir+subDir+'/')

