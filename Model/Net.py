from Model.Building import *
class Net(torch.nn.Module):
    def __init__(self, Name='CNN' ,**kwargs):
        super(Net, self).__init__( **kwargs)
        self.blocks = torch.nn.Sequential(
            GetSomeLayer('Blocks2DTorch',1,64),
            torch.nn.Sequential() if Name=='CNN' else GetSomeLayer(Name,64,64),
            GetSomeLayer('Blocks2DTorch',64,128),
            torch.nn.Sequential() if Name=='CNN' else GetSomeLayer(Name,128,128),
            GetSomeLayer('Blocks2DTorch',128,128),
            torch.nn.Sequential() if Name=='CNN' else GetSomeLayer(Name,128,128),
            GetSomeLayer('Blocks2DTorch',128,256),
            torch.nn.Sequential() if Name=='CNN' else GetSomeLayer(Name,256,256),
            torch.nn.Flatten(),
            GetSomeLayer('ClasserTorch',4096,8))
    def forward(self,inputs):
        return self.blocks(inputs)
def ModelTraining(Name,TrainDataSetList,TestDataSetList):
    torch.cuda.empty_cache()
    H = []
    MPlPath = Configure.FD+"Save" + LIST + "Model_Parameter"+LIST
    HPath = Configure.FD+"Save" + LIST + "History"+LIST
    CEL = torch.nn.CrossEntropyLoss()
    print('====================================================================')
    print('Initialization Completed！ '+Name+' Start Training')
    for n in range(Configure.NCV):
        print('The '+str(n+1)+'-th Cross Validation Dataset')
        H.append([])
        Model,Best = Net(Name=Name).to(device),0
        Optimizer = torch.optim.Adam(Model.parameters())
        for epoch in range(Configure.Epoch):
            for step, (TRI,TRL) in enumerate(TrainDataSetList[n]):
                TRI,TRL = TRI.to(device),TRL.to(device)
                TRLG = Model(TRI)
                TRLOSS = CEL(TRLG,TRL)
                Optimizer.zero_grad()
                TRLOSS.backward(retain_graph=True)
                Optimizer.step()
                del TRI,TRL,TRLG,TRLOSS
                if step%Configure.ASS==0:
                    Temp=np.zeros(4)
                    for _, (TRI,TRL) in enumerate(TrainDataSetList[n]):
                        TRI,TRL = TRI.to(device),TRL.to(device)
                        TRLG = Model(TRI)
                        t1 = TRLG.max(1).indices.cpu().detach().numpy()
                        t2 = TRL.cpu().detach().numpy()
                        Temp[0] += TRL.size()[0]
                        Temp[1] += np.sum(np.array(np.equal(t1,t2)))
                        del TRI,TRL,TRLG,t1,t2
                    for _, (TEI,TEL) in enumerate(TestDataSetList[n]):
                        TEI,TEL = TEI.to(device),TEL.to(device)
                        TELG = Model(TEI)
                        t1 = TELG.max(1).indices.cpu().detach().numpy()
                        t2 = TEL.cpu().detach().numpy()
                        Temp[2] += TEL.size()[0]
                        Temp[3] += np.sum(np.array(np.equal(t1,t2)))
                        del TEI,TEL,TELG,t1,t2
                    printtemp = 'Epoch: {0} TrainAcc: {1} TestAcc: {1}'.format(epoch+1,round(100*Temp[1]/Temp[0],2),round(100*Temp[3]/Temp[2],2))
                    print("\r" + printtemp, end="") if Configure.PS else print(printtemp)
                    S = round(100*Temp[3]/Temp[2],2)
                    if Best < S:
                        torch.save(Model.state_dict(),MPlPath + Name + str(n) +".pkl")
                        Best = S
        print("\n")
    np.save(HPath+ Name +".npy",np.array(H))
    print(Name+' Training Completed！')
    print('====================================================================')


