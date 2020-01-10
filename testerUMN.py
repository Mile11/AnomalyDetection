from anomalydetection import anomalyDetect
import numpy as np

if __name__ == '__main__':

    filePath = 'UMN/Splitted/Crowd-Activity-All_'
    L = 12
    refF = [450, 450, 360, 360, 360, 360, 360, 360, 450, 450, 450]
    gtLimits = [[486, 595], [672, 825], [323, 419], [583, 656], [495, 600], [469, 535], [746, 850], [462, 568], [546, 627], [569, 650], [718, 790]]

    totalError = 0
    totalFrames = 0

    for i in range(1, 12):

        anomalies = anomalyDetect(filePath + str(i) + '.mp4', L=L, useExistingRef=True, refF=refF[i-1], vidFile=True)

        falsePos1 = np.sum(anomalies[:gtLimits[i-1][0]])
        falseNeg = np.size(anomalies[gtLimits[i-1][0]:gtLimits[i-1][1]], axis=0) - np.sum(anomalies[gtLimits[i-1][0]:gtLimits[i-1][1]])
        falsePos2 = np.sum(anomalies[gtLimits[i-1][1]:])
        errors = falsePos1 + falseNeg + falsePos2
        frames = np.size(anomalies, axis=0)

        print("Error for " + filePath + str(i) + ": " + str(errors) + "/" + str(frames))

        totalError += errors
        totalFrames += frames

    print("Total error on UMN: " + str(totalError*100/totalFrames) + "%")