from anomalydetection import anomalyDetect, getAvgInteractionForceSum

# These are a few individual tests that might be interesting to show off

if __name__ == '__main__':
    # Tests

    # UMN
    L = 12
    anomalyDetect('UMN/Splitted/Crowd-Activity-All_1.mp4', '.mp4', L, True, 450, vidFile=True)
    anomalyDetect('UMN/Splitted/Crowd-Activity-All_3.mp4', '.mp4', L, True, 360, vidFile=True)
    anomalyDetect('UMN/Splitted/Crowd-Activity-All_9.mp4', '.mp4', L, True, 450, vidFile=True)

    # UCSD Ped 2
    L = 12
    refForce = getAvgInteractionForceSum('UCSDped2/Train/Train002/', '.tif', L)
    anomalyDetect('UCSDped2/Test/Test006/', '.tif',  L, True, refForce, 1.1)
    anomalyDetect('UCSDped2/Test/Test002/', '.tif',  L, True, refForce, 1.1)
    anomalyDetect('UCSDped2/Test/Test008/', '.tif',  L, True, refForce, 1.1)

    # UCSD Ped 1
    L = 10
    refForce = getAvgInteractionForceSum('UCSDped1/Train/Train001/', '.tif', L)
    anomalyDetect('UCSDped1/Test/Test032/', '.tif',  L, True, refForce)
    anomalyDetect('UCSDped1/Test/Test002/', '.tif',  L, True, refForce)