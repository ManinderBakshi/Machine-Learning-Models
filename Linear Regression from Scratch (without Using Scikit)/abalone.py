import regression

abX, abY = regression.loadDataSet('abalone.txt')
yHat01=regression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
yHat1=regression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
yHat10=regression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)
print('Error on training data when K=0.1 - ',regression.rssError(abY[0:99],yHat01.T))
print('Error on training data when K=1.0 - ',regression.rssError(abY[0:99],yHat1.T))
print('Error on training data when K=10 - ',regression.rssError(abY[0:99],yHat10.T))

yHat01=regression.lwlrTest(abX[100:199],abX[0:99],abY[0:99],0.1)

print('Error on Test Data when k=0.1: ',regression.rssError(abY[100:199],yHat01.T))

yHat1=regression.lwlrTest(abX[100:199],abX[0:99],abY[0:99],1)
print('Error on Test Data when k=1: ',regression.rssError(abY[100:199],yHat1.T))

yHat10=regression.lwlrTest(abX[100:199],abX[0:99],abY[0:99],10)
print('Error on Test Data when k=10:',regression.rssError(abY[100:199],yHat10.T))
