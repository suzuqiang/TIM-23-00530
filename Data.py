from CF import *
#%%
def DateRead():
    FilePath = Configure.FD + "Data" + LIST
    OriginalDataFilesName= os.listdir(FilePath + "Raw Data" + LIST)
    for i in OriginalDataFilesName:
        Data = scipy.io.loadmat(FilePath + "Raw Data" + LIST + i)
        Data = Data['SourceData1'][np.arange(Configure.WC*4,(Configure.WC+1)*4)].reshape(1,-1,Configure.SL,2)
        AlllData = Data if i == OriginalDataFilesName[0] else np.concatenate((AlllData,Data))
    AlllData = (AlllData.reshape(-1,1,Configure.SL,2))[:,:,:,0]
    AlllLabel = np.arange(len(OriginalDataFilesName)).repeat(len(AlllData)//len(OriginalDataFilesName))
    return AlllData,AlllLabel
#%%
def GetRandomNum(Name,length):
    FilePath = Configure.FD+'Save'+LIST+'RN'+LIST+Name+'.npy'
    if not (os.path.exists(FilePath)):
        RandomNum = random.sample(range(0, length), length)
        np.save(FilePath,RandomNum)
    return FilePath
#%%
def DateSplit(FilePath):
    print('Start dividing data',end= '....  ')
    Data,Label = DateRead()
    Configure.NSC,Configure.NSPT = np.max(Label)+1,len(Label)//(np.max(Label)+1)
    RandomNum = np.load(GetRandomNum('RandonNumDDS_'+str(len(Label))+'_'+str(Configure.RN),len(Label)))
    Data,Label = Data[RandomNum],Label[RandomNum]
    temp = int(Configure.NSPT*Configure.NSC/Configure.NCV)
    SplitedData,SplitedLabel = np.zeros((Configure.NCV,temp,1,Configure.SL)),np.zeros((Configure.NCV,temp))
    temp,index = np.zeros((Configure.NSC),dtype = 'int32'),np.zeros((6),dtype = 'int32')
    index[5] = (Configure.NSPT/Configure.NCV)
    for i in range(len(Label)):
        x  = int(temp[Label[i]]//index[5])
        SplitedData[x,index[x]],SplitedLabel[x,index[x]] = Data[i],Label[i]
        index[x] += 1
        temp[Label[i]] += 1
    np.save(FilePath + 'Data',SplitedData)
    np.save(FilePath + 'Label',SplitedLabel)
    print('Dividing completed！')
#%%
def wgn(data, snr=10):
    Ps = np.sum(abs(data)**2)/len(data)
    Pn = Ps/(10**((snr/10)))
    noise = np.random.normal(0, 1, len(data)) * np.sqrt(Pn)
    signal_add_noise = data + noise
    return signal_add_noise
def DataAddNosie(TimeData,SNR):
    for i in range(len(TimeData)):
        for n in range(len(TimeData[0])):
            TimeData[i,n] = wgn(TimeData[i,n],SNR)
    return TimeData
#%%
def DetaProcess(TimeData,index=6):
    WPTData = np.zeros((len(TimeData),1,2**index,2**index))
    for m in range(len(TimeData)):
        for o in range(1):
            wp = pywt.WaveletPacket(data=TimeData[m,o], wavelet='db1',mode='symmetric',maxlevel=index)
            num = 0
            for i in [node.path for node in wp.get_level(index, 'freq')]:
                WPTData[m,o,num] = wp[i].data
                num = num+1
    return WPTData
#%%
def DataDeal(Data,Label):
    print('Dataset making',end= '....  ')
    from torch.utils.data import Dataset,DataLoader,TensorDataset
    temp = int(Configure.NSPT*Configure.NSC*(Configure.NCV-1)/Configure.NCV)
    TrainRandomNum = np.load(GetRandomNum('TrainRandomNum_'+str(temp),temp))
    TrainDataSetList,TestDataSetList = [],[]
    for i in range(Configure.NCV):
        TrainData  = (Data[np.delete(np.arange(Configure.NCV),i)].reshape((-1,1,Configure.SL)))[TrainRandomNum]
        TrainData = DataAddNosie(TrainData,Configure.SNR)
        TrainData = DetaProcess(TrainData)
        TrainLabel = (Label[np.delete(np.arange(Configure.NCV),i)].reshape((-1)))[TrainRandomNum]
        TrainData,TrainLabel = torch.from_numpy(TrainData).float(),torch.from_numpy(TrainLabel).long()
        TrainDataSetList.append(DataLoader(dataset=TensorDataset(TrainData,TrainLabel), batch_size=Configure.Batch, shuffle=False, drop_last=False))
        del TrainData,TrainLabel
        TestData  = DetaProcess(Data[i])
        TestLabel = Label[i].reshape((-1))
        TestData,TestLabel = torch.from_numpy(TestData).float(),torch.from_numpy(TestLabel).long()
        TestDataSetList.append(DataLoader(dataset=TensorDataset(TestData,TestLabel), batch_size=Configure.Batch, shuffle=False, drop_last=False))
        del TestData,TestLabel
    print('Dataset completed!')
    return TrainDataSetList,TestDataSetList
def SaveProcessedData(Data,Label):
    temp = int(Configure.NSPT*Configure.NSC*(Configure.NCV-1)/Configure.NCV)
    TrainRandomNum = np.load(GetRandomNum('TrainRandomNum_'+str(temp),temp))
    SavedTrainData,SavedTrainLabel = [],[]
    SavedTestData,SavedTestLabel = [],[]
    for i in range(Configure.NCV):
        TrainData  = (Data[np.delete(np.arange(Configure.NCV),i)].reshape((-1,1,Configure.SL)))[TrainRandomNum]
        TrainData = DataAddNosie(TrainData,Configure.SNR)
        TrainData = DetaProcess(TrainData)
        TrainLabel = (Label[np.delete(np.arange(Configure.NCV),i)].reshape((-1)))[TrainRandomNum]
        SavedTrainData.append(TrainData),SavedTrainLabel.append(TrainLabel)
        TestData  = DetaProcess(Data[i])
        TestLabel = Label[i].reshape((-1))
        SavedTestData.append(TestData),SavedTestLabel.append(TestLabel)
    SavedTrainData,SavedTrainLabel = np.array(SavedTrainData),np.array(SavedTrainLabel)
    SavedTestData,SavedTestLabel = np.array(SavedTestData),np.array(SavedTestLabel)
    np.save(FilePath + 'TrainData.npy',SavedTrainData),np.save(FilePath + 'TrainLabel.npy',SavedTrainLabel)
    np.save(FilePath + 'TestData.npy',SavedTestData),np.save(FilePath + 'TestLabel.npy',SavedTestLabel)
def SavedDataDeal(FilePath):
    from torch.utils.data import Dataset,DataLoader,TensorDataset
    TrainDataSetList,TestDataSetList = [],[]
    SavedTrainData,SavedTrainLabel = np.load(FilePath + 'TrainData.npy'),np.load(FilePath + 'TrainLabel.npy')
    SavedTestData,SavedTestLabel = np.load(FilePath + 'TestData.npy'),np.load(FilePath + 'TestLabel.npy')
    for i in range(Configure.NCV):
        TrainData,TrainLabel = torch.from_numpy(SavedTrainData[i]).float(),torch.from_numpy(SavedTrainLabel[i]).long()
        TrainDataSetList.append(DataLoader(dataset=TensorDataset(TrainData,TrainLabel), batch_size=Configure.Batch, shuffle=False, drop_last=False))
        TestData,TestLabel = torch.from_numpy(SavedTestData[i]).float(),torch.from_numpy(SavedTestLabel[i]).long()
        TestDataSetList.append(DataLoader(dataset=TensorDataset(TestData,TestLabel), batch_size=Configure.Batch, shuffle=False, drop_last=False))
    return TrainDataSetList,TestDataSetList
FilePath = Configure.FD+"Data"+LIST+"RN"+str(Configure.RN)
# if not (os.path.exists(FilePath + 'Label.npy')):
#     print('Target data not detected')
#     DateSplit(FilePath)
# else:
#     print('Target data detected')
# Data,Label = np.load(FilePath + 'Data.npy'),np.load(FilePath + 'Label.npy')
# print('Data loading completed！')
# del FilePath

TrainDataSetList,TestDataSetList = SavedDataDeal(FilePath)



