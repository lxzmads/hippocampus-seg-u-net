# coding: utf-8

# import built-in modules
import os,pickle,random,gc
import hashlib
import traceback
###########################

# import third parties' modules
import tensorlayer as tl
import nibabel as nib
import numpy as np
###########################

# import my modules

###########################
class DataPreprocessor(object):
    """
    DataPreprocessor class.
    @pythonVersion: 3.5.2
    @methods:
            __init__        Initiate data preprocessor instance.
            getAllData      Get training data and test data. Define the ratio of training data
                            and test data by set 'trainTestRate' parameter and 'dataSize' as size
                            of data to use.
    @author: XZ.Liu
    @creation: 2018-03-18
    @modified: 2018-03-20
    @version: 0.1
    """
    def __init__(self, imgsPath='atlas-part1/', segsPath='label-part1/', saveDir='data/train_test_all/'):
        self.__imgsPath = imgsPath
        self.__segsPath = segsPath
        self.__saveDir = saveDir
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

    def __checkDim(self,imgData):
        # check the shape of image
        if imgData.shape != (256,256,180):
            x_pad = (256-imgData.shape[0])//2
            y_pad = (256-imgData.shape[1])//2
            z_pad = (180-imgData.shape[2])//2
            imgData = np.pad(imgData, ((x_pad,x_pad),(y_pad,y_pad),(z_pad,z_pad)),'constant',constant_values=((-1,-1),(-1,-1),(-1,-1)))
        return imgData
    def __loadImg(self, left, right):
        """
        Load the images' 3D data to a dict classfied by images' different dimensions.
        Returns :list: dataList object

        :param left: Left index of __imgsPath's image list to load.
        :param right: Right index of __imgsPath's image list to load.
        """
        imgList = os.listdir(self.__imgsPath)[left:right]
        dataList = []
        # load image data by image shape
        for imgName in imgList:
            imgPath = self.__imgsPath + imgName
            img = nib.load(imgPath)
            imgData = img.get_data() #(256, 256, 166, 1)
            # get the first three dimension - 3D data
            imgData = imgData[:,:,:,0]
            imgData = self.__checkDim(imgData)
            # if the list of the specific dimension data already exists, we just append data to it.
            dataList.append(imgData)
        return dataList # shape [(256, 256, 180)]

    def __getNormalizeParam(self, dataList):
        """
        Get the parameters we need to normalize the images' data.
        Returns :dict: dataMeanStdDict

        :param dataList: Data list to use for calculating the normalizing parameters.
        """
        dataList = np.asarray(dataList)
        # calculate mean and std
        m = np.mean(dataList)
        s = np.std(dataList)
        dataMeanStdDict = {'mean': 0.0, 'std': 1.0}
        dataMeanStdDict['mean'] = m
        dataMeanStdDict['std'] = s
        return dataMeanStdDict

    def __save(self, object, name = 'mean_std_dict.pickle'):
        """ Save file to pickle. """
        with open(self.__saveDir + name, 'wb') as f:
            pickle.dump(object, f, protocol=4)

    def __getData(self, imgList, dataMeanStdDict):
        """
        Get normalized image data.
        Returns :tuple: X_input,X_target

        :param imgList: Image list used to normalize and construct the return tuple.
        :param dataMeanStdDict: Dict of image list's parameters for normalizing.
        """
        X_input = []
        X_target = []
        for imgName in imgList:
            imgPath = self.__imgsPath + imgName
            imgData = nib.load(imgPath).get_data()
            imgData = imgData[:, :, :, 0]
            imgData = self.__checkDim(imgData)
            # dimIndex = str(hashlib.md5(str(imgData.shape).encode()).hexdigest())[:6]
            imgData = (imgData -  dataMeanStdDict['mean']) / dataMeanStdDict['std']
            imgData = imgData.astype(np.float32)

            # default segentation image name is the same as image name
            segName = imgName
            segImgData = nib.load(self.__segsPath + segName).get_data()
            segImgData = segImgData[:,:,:,0]
            segImgData = self.__checkDim(segImgData)
            for j in range(imgData.shape[2]):
                tmpArray = imgData[:, :, j]
                tmpArray.astype(np.float32)
                X_input.append(tmpArray)
                seg2d = segImgData[:, :, j]
                seg2d.astype(int)
                X_target.append(seg2d)
            gc.collect()
            print("finished {}".format(imgName))
        return X_input,X_target

    def getAllData(self,trainTestRate = 0.8, dataSize = 'small'):
        """
        Get training data and testing data.
        Returns :tuple: X_trainInput,X_trainTarget,X_testInput,X_testTarget

        :param trainTestRate: The value of (training data / testing data)
        :param: dataSize: The size of images used. legal values are 'small' 'half' 'all'
        """
        try:
            num = len(os.listdir(self.__imgsPath))
            if dataSize == 'small':
                dataSize = num // 10
            elif dataSize == 'half':
                dataSize = num // 2
            elif dataSize == 'all':
                dataSize = num
            else:
                raise ValueError('Illegal dataSize')
        except Exception as e:
            traceback.print_exc()
        finally:
            print('Using data size: {}'.format(dataSize))

        try:
            if int(trainTestRate) < 0 or int(trainTestRate) > 1:
                raise ValueError('Error trainTestRate')
            trainIndex = int(dataSize * trainTestRate)
        except ValueError as e:
            traceback.print_exc()
        finally:
            print('train index by {}'.format(trainIndex))

        trDataList = self.__loadImg(0, trainIndex)
        txDataList = self.__loadImg(trainIndex, dataSize)
        trDataMeanStdDict = self.__getNormalizeParam(trDataList)
        txDataMeanStdDict = self.__getNormalizeParam(txDataList)
        # self.__save(trDataMeanStdDict, name = 'tr_mean_std_dict.pickle')
        # self.__save(txDataMeanStdDict, name = 'tx_mean_std_dict.pickle')

        try:
            segList = os.listdir(self.__segsPath)[:dataSize - 1]
            imgList = os.listdir(self.__imgsPath)[:dataSize - 1]
            trImgList = imgList[:trainIndex]
            txImgList = imgList[trainIndex:]
            trSegList = segList[:trainIndex]
            txSegList = segList[trainIndex:]
        except Exception as e:
            traceback.print_exc()

        print('Getting training data...')
        X_trainInput,X_trainTarget = self.__getData(trImgList, trDataMeanStdDict)
        print('Getting testing data...')
        X_testInput,X_testTarget = self.__getData(txImgList, txDataMeanStdDict)

        # for i in X_trainInput:
        #     print(i.shape)
        X_trainInput = np.asarray(X_trainInput, dtype='float32')
        X_trainTarget = np.asarray(X_trainTarget, dtype='float32')
        X_testInput = np.asarray(X_testInput)
        X_testTarget = np.asarray(X_testTarget)
        # print(X_trainInput.shape)
        # print(X_trainTarget.shape)
        # print(X_testInput.shape)
        # print(X_testTarget.shape)
        return X_trainInput,X_trainTarget,X_testInput,X_testTarget

if __name__ == '__main__':
    print('import me plz :)')
