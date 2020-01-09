import cv2
import numpy as np
import os.path
import random


# Creates a set of particles on a grid. Each particle contains a set of coordinates on a frame
# imageDims -- dimensions of the image, to know the interval of the random initial positions
def createParticles(imageDims):

    xCoords = np.linspace(1, imageDims[0]-2, imageDims[0]//3).astype(int)
    yCoords = np.linspace(1, imageDims[1]-2, imageDims[1]//3).astype(int)

    return np.array(np.meshgrid(xCoords, yCoords)).T.reshape(-1, 2)


# Calculate the magnitudes of interaction forces for given particles
# opticFlows -- array of optic flows of all the frames so far (the last one is the most current one)
# L -- number of previous frames to look at when determining the average optical flow
# tau -- relaxing parameter (see: social force equation)
def calcInteractionForces(opticFlows, L, tau=0.5):

    # Calculate the social force (approximated as subtraction between the optic flow of the current and previous frame)
    socialForce = opticFlows[-1] - opticFlows[-2]

    # Calculate the average optical flow from the previous L frames
    # In case there's less than L frames processed in the algorithm, use all of the available optic flows
    if L >= len(opticFlows):
        avgOpt = np.sum(opticFlows, axis=0) / len(opticFlows)
    else:
        avgOpt = np.sum(opticFlows[-L:], axis=0) / L

    # Personal (desired) force of the particle
    # Calculated as the subtraction of average optical flow from the optical flow of the current frame
    personalForce = opticFlows[-1] - avgOpt

    # Calculating the interaction force:
    interactionForce = socialForce - personalForce / tau

    # We only need to determine the magnitude of the vectors, which we can
    # do by converting the vectors to the polar coordinate system
    # (Hence, notice that we only need 'mag' here)
    mag, ang = cv2.cartToPolar(interactionForce[..., 0], interactionForce[..., 1])

    return mag


# Get the interaction force for the positions of given particles
# particles -- particle positions
# interactionForce -- a 2D array containing all of the magnitudes of all of the interaction forces
def getInteractionForcesForParticles(particles, interactionForce):

    retForces = []
    for p in particles:
        retForces.append(interactionForce[p[0], p[1]])

    return retForces


# Particle advection
# Moves particles every L frames in the direction of the average optical flow corresponding to their current position
# particles -- an array of particle positions
# opticFlows -- a list of dense optical flows from previous frames
# L -- number of previous frames to look at when determining the average optical flow
def particleAdvection(particles, opticFlows, L):

    # To speed up the process, we can use the dense optical flow to calculate the interaction force for any position on
    # the current frame, and then simply look at the positions as we need to during the algorithm
    interactionForces = calcInteractionForces(opticFlows, L)

    if len(opticFlows) % L == 0:

        avgOpt = np.sum(opticFlows[-L:], axis=0) / L

        helpM = []

        for p in particles:
            helpM.append(avgOpt[p[0], p[1], 0])
            helpM.append(avgOpt[p[0], p[1], 1])

        helpM = np.array(helpM).reshape(np.size(helpM)//2, 2)
        particles = particles.astype(float)
        particles += helpM
        particles = particles.astype(int)

        particles = np.clip(particles, [0, 0], [np.size(opticFlows[0], axis=0)-1, np.size(opticFlows[0], axis=1)-1])

    return particles, getInteractionForcesForParticles(particles, interactionForces)


# Gauss approximation
# Approximates a gaussian curve based on force magnitudes
# Returns all the 3-sigma outliers for such a curve
# particles -- particles
# forces -- interaction forces
def gaussApprox(particles, forces):

    #return particles, forces

    helpForce = np.copy(forces)
    samplSize = len(helpForce)

    alsoHelp = helpForce[:samplSize]
    mean = np.sum(alsoHelp) / samplSize
    stddev = np.sqrt(np.sum((alsoHelp - mean)**2) / (samplSize-1))
    tobeat = mean + 3*stddev

    ret = [(p, f) for (p, f) in zip(particles, forces) if f >= tobeat]

    return zip(*ret)


# Method used to get the average sum of interaction force outliers per frame using a video clip without anomalies
# More or less the same skeleton as the actual anomaly detection algorithm in terms of frame analysis and such
# folderName -- path to the folder containing the .tif frames
# extension -- extension frame images have
# L -- number of frames used for determining the average optical flow
def getAvgInteractionForceSum(folderName, extension, L):

    # INITIAL PREPARATION

    # Used to keep track of all the sums of outlier forces on frames
    totalForces = []

    # A list of all previous optic flow vectors calculated in all the frames
    opticFlows = []

    # Prepare full file path
    foldPath = folderName + '/'
    i = 1
    prefix = '00'

    # Complete path to the very first frame
    file = foldPath + prefix + str(i) + extension

    # Load the first frame
    # The images themselves are already in greyscale, but because OpenCV can't figure
    # it out on its own, we need to manually set every frame to greyscale to make
    # sure the optical flow will be calculated correctly later down the line
    prev = cv2.imread(file)
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Prepare to start at the next frame
    i += 1
    file = foldPath + prefix + str(i) + extension

    # Create particles
    particles = createParticles(prev.shape)

    # BEGINNING OF THE ALGORITHM

    print("Learning average interaction force for a frame without anomalies...")

    # Repeat as long as there are frames:
    while os.path.isfile(file):

        # Load next frame
        frame = cv2.imread(file)
        framegray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optic flow (dense), using the previous and current frame
        # Returns an M x N array of 2D vectors (where M x N are dimensions of the image)
        # In other words, we've determined the optical flow for every pixel on the image
        flow = cv2.calcOpticalFlowFarneback(prev, framegray, None, 0.5, 4, 15, 3, 5, 1.2, 0)

        # Add the calculated optic flow to the list of optic flows
        opticFlows.append(flow)

        if i > L:
            # Calculate the interaction forces for all the particles
            particles, forces = particleAdvection(particles, opticFlows, L)

            # Determine the particles that have interaction force values over 3 sigma by making a Gauss approximation
            outlierparticles, outlierforces = gaussApprox(particles, forces)

            # Remember the sum of outlier forces for that frame
            totalForces.append(np.sum(list(outlierforces)))

        # Show frame
        cv2.imshow('Frame', frame)

        # Stop the algorithm with the 'q' key
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        # Prepare the prefix for the next file (in this dataset, no sample has over 1000 frames)
        i += 1
        if 10 <= i < 100:
            prefix = '0'
        elif i >= 100:
            prefix = ''

        # Prepare path for next frame, set the current frame to be the previous one
        file = foldPath + prefix + str(i) + extension
        prev = framegray

    print("Done!")
    # Return the average outlier sum
    return np.sum(totalForces) / len(totalForces)


# Anomaly detection function
# Returns an array containing flags for each frame -- 0 if no anomaly was found, 1 if it was
# folderName -- path to the folder containing the .tif frames
# extension -- extension frame images have
# L -- number of frames used for determining the average optical flow
# useExistingRef -- flag to determine whether or not the user will be using a pre-set outlier sum as reference
# refF -- the value of the interaction force outlier sum
# refScale -- how much the interaction force outlier sum of a frame needs to go over the reference sum to be considered
# an anomaly
def anomalyDetect(folderName, extension, L, useExistingRef=False, refF=None, refScale=1.1):

    # INITIAL PREPARATION

    print()
    print("Preparing for " + str(folderName))

    # List of detected anomalies
    # Each field represents a frame -- 0 means no anomaly, 1 means anomaly
    anomalies = []

    # A list of all previous optic flow vectors calculated in all the frames
    opticFlows = []

    # Prepare full file path
    foldPath = folderName + '/'
    i = 1
    prefix = '00'

    # Complete path to the very first frame
    file = foldPath + prefix + str(i) + extension

    # Load the first frame
    # The images themselves are already in greyscale, but because OpenCV can't figure
    # it out on its own, we need to manually set every frame to greyscale to make
    # sure the optical flow will be calculated correctly later down the line
    prev = cv2.imread(file)
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Prepare to start at the next frame
    i += 1
    file = foldPath + prefix + str(i) + extension

    # Create particles
    particles = createParticles(prev.shape)

    # BEGINNING OF THE ALGORITHM

    # Repeat as long as there are frames:
    while os.path.isfile(file):

        # Load next frame
        frame = cv2.imread(file)
        framegray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optic flow (dense), using the previous and current frame
        # Returns an M x N array of 2D vectors (where M x N are dimensions of the image)
        # In other words, we've determined the optical flow for every pixel on the image
        flow = cv2.calcOpticalFlowFarneback(prev, framegray, None, 0.5, 4, 15, 3, 5, 1.2, 0)

        # Add the calculated optic flow to the list of optic flows
        opticFlows.append(flow)

        if i > L:

            # Calculate the interaction forces for all particles
            particles, forces = particleAdvection(particles, opticFlows, L)

            # Determine the particles that have interaction force values over 3 sigma by making a Gauss approximation
            outlierparticles, outlierforces = gaussApprox(particles, forces)
            outlierparticles = list(outlierparticles)

            # If the user has not given an input on what the reference sum should be, take the L+1st frame as
            # the reference point
            # YOU SHOULD DEFINITELY NOT DO THIS
            if i == L+1 and useExistingRef==False:
                refF = np.sum(list(outlierforces))
                print(refF)

            # Either way, determine if a frame has an anomaly by summing up the outlier interaction force values and
            # seeing if they go over the reference sum
            else:
                if np.abs(np.sum(list(outlierforces))) > refScale*refF:
                    anomalies.append(1)
                    print("Frame " + str(i) + ": Anomaly detected! Interaction force sum: " + str(np.sum(list(outlierforces))))
                else:
                    anomalies.append(0)

            # Draw the outlier particles
            for k in range(np.size(outlierparticles, 0)):
                frame[outlierparticles[k][0], outlierparticles[k][1]] = [0,0,255]

        # Show frame
        cv2.imshow('Frame', frame)

        # Stop the algorithm with the 'q' key
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        # Prepare the prefix for the next file (in this dataset, no sample has over 1000 frames)
        i += 1
        if 10 <= i < 100:
            prefix = '0'
        elif i >= 100:
            prefix = ''

        # Prepare path for next frame, set the current frame to be the previous one
        file = foldPath + prefix + str(i) + extension
        prev = framegray

    return np.array(anomalies)


if __name__ == '__main__':
    # Tests
    # UCSD Ped 2
    L = 12
    refForce = getAvgInteractionForceSum('UCSDped2/Train/Train002', '.tif', L)
    anomalyDetect('UCSDped2/Test/Test006', '.tif',  L, True, refForce, 1.1)
    anomalyDetect('UCSDped2/Test/Test002', '.tif',  L, True, refForce, 1.1)
    anomalyDetect('UCSDped2/Test/Test008', '.tif',  L, True, refForce, 1.1)

    # UCSD Ped 1
    L = 10
    refForce = getAvgInteractionForceSum('UCSDped1/Train/Train001', '.tif', L)
    anomalyDetect('UCSDped1/Test/Test032', '.tif',  L, True, refForce)
    anomalyDetect('UCSDped1/Test/Test002', '.tif',  L, True, refForce)
