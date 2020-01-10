from anomalydetection import anomalyDetect, getAvgInteractionForceSum
import os.path
import numpy as np
import cv2


def groundTruthAnalysis(folderName, extension):

    gt = []

    # Prepare full file path
    foldPath = folderName + '/'
    i = 1
    prefix = '00'

    file = foldPath + prefix + str(i) + extension
    filealt = foldPath + 'frame' + prefix + str(i) + extension

    if not os.path.isfile(file):
        file = filealt
        foldPath += 'frame'

    # Repeat as long as there are frames:
    while os.path.isfile(file):

        # Load next frame
        frame = cv2.imread(file)

        if np.sum(frame) > 0:
            gt.append(1)
        else:
            gt.append(0)

        # Prepare the prefix for the next file (in this dataset, no sample has over 1000 frames)
        i += 1
        if 10 <= i < 100:
            prefix = '0'
        elif i >= 100:
            prefix = ''

        # Prepare path for next frame, set the current frame to be the previous one
        file = foldPath + prefix + str(i) + extension

    return np.array(gt), i


if __name__ == '__main__':

    bases = ['UCSDped2', 'UCSDped1']
    totrain = ['Train002', 'Train001']
    Ls = [12, 10]
    trueErrors = []
    falsePositives = []

    for b, t, L in zip(bases, totrain, Ls):

        refForce = getAvgInteractionForceSum(b + '/Train/' + t + '/', '.tif', L)

        totalFrameNum = 0
        totalErrors = 0

        file = b + '/Test/Test'
        s1 = file + '001'
        i = 1
        suff = '00'

        print("TESTING THE TEST SET")
        while os.path.isdir(s1):

            if not os.path.isdir(s1 + '_gt') or (b == 'UCSDped1' and (i == 21)):
                i += 1
                if 10 <= i < 100:
                    suff = '0'
                elif i >= 100:
                    suff = ''
                s1 = file + suff + str(i)
                continue

            anomalies = anomalyDetect(s1 + '/', '.tif',  L, True, refForce)
            gt, frameNum = groundTruthAnalysis(s1 + '_gt', '.bmp')

            differences = np.bitwise_xor(anomalies, gt[L:])
            errors = np.sum(differences)
            totalFrameNum += frameNum - L
            totalErrors += errors
            print("Test " + str(i) + " errors: " + str(errors) + "/" + str(frameNum - L))

            i += 1
            if 10 <= i < 100:
                suff = '0'
            elif i >= 100:
                suff = ''

            s1 = file + suff + str(i)

        print('\n' + b + " error rate: " + str(totalErrors*100/totalFrameNum) + "%")
        trueErrors.append(totalErrors*100/totalFrameNum)

        print("CHECKING FOR FALSE POSITIVES ON THE TRAIN SET")

        totalFrameNum = 0
        totalErrors = 0

        file = b + '/Train/Train'
        s1 = file + '001'
        i = 1
        suff = '00'

        while os.path.isdir(s1):

            anomalies = anomalyDetect(s1 + '/', '.tif', L, True, refForce)

            frameNum = np.size(anomalies)
            totalFrameNum += frameNum - L
            errors = np.sum(anomalies)
            totalErrors += errors
            print("Test " + str(i) + " errors: " + str(errors) + "/" + str(frameNum - L))

            i += 1
            if 10 <= i < 100:
                suff = '0'
            elif i >= 100:
                suff = ''

            s1 = file + suff + str(i)

        print(b + " false positive rate: " + str(totalErrors * 100 / totalFrameNum) + "%")
        falsePositives.append(totalErrors * 100 / totalFrameNum)

    print("TRUE ERRORS")
    print(np.array(trueErrors))
    print("FALSE POSITIVES")
    print(np.array(falsePositives))


