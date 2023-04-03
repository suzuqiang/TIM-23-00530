from Data import *
from Model.Net import Net,ModelTraining
ModelTraining('CNN',TrainDataSetList,TestDataSetList)
ModelTraining('SeNet2DPytorch',TrainDataSetList,TestDataSetList)
ModelTraining('Simam2DPytorch',TrainDataSetList,TestDataSetList)
ModelTraining('CBAM2DPytorch',TrainDataSetList,TestDataSetList)
ModelTraining('FcaNet2DPytorch',TrainDataSetList,TestDataSetList)
ModelTraining('FcaNetPlus2DPytorch',TrainDataSetList,TestDataSetList)
ModelTraining('FcaNetPlusPlus2DPytorch',TrainDataSetList,TestDataSetList)
