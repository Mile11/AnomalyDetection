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
def particleAdvection(particles, opticFlows, L, tau=0.5):

    # To speed up the process, we can use the dense optical flow to calculate the interaction force for any position on
    # the current frame, and then simply look at the positions as we need to during the algorithm
    interactionForces = calcInteractionForces(opticFlows, L, tau=tau)

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

    helpForce = np.copy(forces)
    samplSize = len(helpForce)

    alsoHelp = helpForce[:samplSize]
    mean = np.sum(alsoHelp) / samplSize
    stddev = np.sqrt(np.sum((alsoHelp - mean)**2) / (samplSize-1))
    tobeat = mean + 3*stddev

    ret = [(p, f) for (p, f) in zip(particles, forces) if f >= tobeat]

    return zip(*ret)


# Method used to get the average sum of interaction force outliers per frame using a video clip without anomalies
# More or less the same skeleton as the actual anomaly detection algorithm (in terms of frame analysis and such)
# foldPath -- if the user is inputting the frames as image files, this should be the path to the folder containing the files.
# if the user is loading a video file, this should be the path to the video file (extension must be included)
# extension -- extension frame images have; not used in the case of a video file
# L -- number of frames used for determining the average optical flow
# frameDigits -- the algorithm assumes that, if the frames are stored as seperate files, that they all end with
# a N digit number (where the number of digits remains constant. for example, if N = 4, the first frame would be
# labeled as '0001' and so on)
# vidFile -- flag to check if the file given is a video file
# tau -- relaxation constant used for social force model; advised to be in an interval of [0.5, 0.85]
def getAvgInteractionForceSum(foldPath, extension='', L=10, frameDigits=3, vidFile=False, tau=0.5):

    # INITIAL PREPARATION

    # Used to keep track of all the sums of outlier forces on frames
    totalForces = []

    # A list of all previous optic flow vectors calculated in all the frames
    opticFlows = []

    # Prepare full file path
    i = 1
    j = 1
    prefix = '0' * (frameDigits-1)

    # Complete path to the very first frame (this step will do nothing in case of a video file)
    file = foldPath + prefix + str(i) + extension

    cap = None

    # Load the first frame
    if not vidFile:
        prev = cv2.imread(file)
    else:
        cap = cv2.VideoCapture(foldPath)
        if not cap.isOpened():
            print("Error opening")
            exit(1)
        ret, prev = cap.read()

    # We need to manually set every frame to greyscale to calculate the optical flow
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Prepare to start at the next frame
    i += 1
    file = foldPath + prefix + str(i) + extension

    # Create particles
    particles = createParticles(prev.shape)

    # BEGINNING OF THE ALGORITHM

    print("Learning average interaction force for a frame without anomalies...")

    # Repeat as long as there are frames:
    while (not vidFile and os.path.isfile(file)) or (vidFile and cap.isOpened()):

        # Load next frame
        if not vidFile:
            frame = cv2.imread(file)
        else:
            ret, frame = cap.read()
            if not ret:
                break

        framegray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optic flow (dense), using the previous and current frame
        # Returns an M x N array of 2D vectors (where M x N are dimensions of the image)
        # In other words, we've determined the optical flow for every pixel on the image
        flow = cv2.calcOpticalFlowFarneback(prev, framegray, None, 0.5, 4, 15, 3, 5, 1.2, 0)

        # Add the calculated optic flow to the list of optic flows
        opticFlows.append(flow)

        if i > L:
            # Calculate the interaction forces for all the particles
            particles, forces = particleAdvection(particles, opticFlows, L, tau=tau)

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
        if i >= 10 ** j:
            j += 1
            prefix = '0' * (frameDigits-j)

        # Prepare path for next frame, set the current frame to be the previous one
        file = foldPath + prefix + str(i) + extension
        prev = framegray

    print("Done!")
    # Return the average outlier sum
    return np.sum(totalForces) / len(totalForces)


# Anomaly detection function
# Returns an array containing flags for each frame -- 0 if no anomaly was found, 1 if it was
# foldPath -- if the user is inputting the frames as image files, this should be the path to the folder containing the files.
# if the user is loading a video file, this should be the path to the video file (extension must be included)
# extension -- extension frame images have; not used in the case of a video file
# L -- number of frames used for determining the average optical flow
# useExistingRef -- flag to determine whether or not the user will be using a pre-set outlier sum as reference
# refF -- the value of the interaction force outlier sum
# refScale -- how much the interaction force outlier sum of a frame needs to go over the reference sum to be considered
# an anomaly
# frameDigits -- the algorithm assumes that, if the frames are stored as seperate files, that they all end with
# a N digit number (where the number of digits remains constant. for example, if N = 4, the first frame would be
# labeled as '0001' and so on)
# vidFile -- flag to check if the file given is a video file
# tau -- relaxation constant used for social force model; advised to be in an interval of [0.5, 0.85]
def anomalyDetect(foldPath, extension='', L=10, useExistingRef=False, refF=None, refScale=1.1, frameDigits=3, vidFile=False, tau=0.5):

    # INITIAL PREPARATION

    print()
    print("Preparing for " + str(foldPath))

    # List of detected anomalies
    # Each field represents a frame -- 0 means no anomaly, 1 means anomaly
    anomalies = []

    # A list of all previous optic flow vectors calculated in all the frames
    opticFlows = []

    # Prepare full file path
    i = 1
    j = 1
    prefix = '0' * (frameDigits-1)

    # Complete path to the very first frame
    file = foldPath + prefix + str(i) + extension

    cap = None

    # Load the first frame
    if not vidFile:
        prev = cv2.imread(file)
    else:
        cap = cv2.VideoCapture(foldPath)
        if not cap.isOpened():
            print("Error opening")
            exit(1)
        ret, prev = cap.read()

    # We need to manually set every frame to greyscale to calculate the optical flow
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Prepare to start at the next frame
    i += 1
    file = foldPath + prefix + str(i) + extension

    # Create particles
    particles = createParticles(prev.shape)

    # BEGINNING OF THE ALGORITHM

    # Repeat as long as there are frames:
    while (not vidFile and os.path.isfile(file)) or (vidFile and cap.isOpened()):

        # Load next frame
        if not vidFile:
            frame = cv2.imread(file)
        else:
            ret, frame = cap.read()
            if not ret:
                break

        framegray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optic flow (dense), using the previous and current frame
        # Returns an M x N array of 2D vectors (where M x N are dimensions of the image)
        # In other words, we've determined the optical flow for every pixel on the image
        flow = cv2.calcOpticalFlowFarneback(prev, framegray, None, 0.5, 4, 15, 3, 5, 1.2, 0)

        # Add the calculated optic flow to the list of optic flows
        opticFlows.append(flow)

        if i > L:

            # Calculate the interaction forces for all particles
            particles, forces = particleAdvection(particles, opticFlows, L, tau=tau)

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
        if i >= 10 ** j:
            j += 1
            prefix = '0' * (frameDigits-j)

        # Prepare path for next frame, set the current frame to be the previous one
        file = foldPath + prefix + str(i) + extension
        prev = framegray

    return np.array(anomalies)


if __name__ == '__main__':
    # Main method used for general testing

    extension = ''
    L = 10
    useExistingRef = False
    vidFile = False
    refF = 0
    tau = 0.5
    refScale = 1.1
    frameDigits = 3
    filePath = ''

    vidFile2 = False
    extension1 = ''
    frameDigits1 = 3
    filePath1 = ''

    print("Welcome to this (relatively simplistic) anomaly detection algorithm!")
    print("Please specify if you will be loading the video as a set of frames (images) or as an actual video file!")

    inp = ''
    while inp != 1 and inp != 2:
        inp = int(input("Enter 1 for images, 2 for video > "))

    if inp == 1:
        print("\nYou have chosen to use images as frames!")
        print("Please specify the path to the folder which contains the desired images.")
        print("WARNING: Image frame file names MUST be properly ennumered and represented with an N-digit counter. For example '001' or 'frame_001'.")
        print("In the case of the former (where it's a simple counter), please make sure to add a '\\' at the end of the folder path.")
        print("For example, if the frames were labeled as 'XXX': 'UCSD/Test/Test1/'")
        print("In the case of the latter, where there is a string of some kind preceding the counter, you must ALSO enter the part of the string.")
        print("For example, if the frames are labeled as 'seq_XXX', you would enter the path: 'mall_dataset/frames/seq_'")
        filePath = input("So, with all that said, please enter the folder path! > ")
        frameDigits = int(input("Please enter the number of digits your frame counter has. (For example, in the case of '001' it would be 3!) > "))
        extension = input("Please enter the file extension! (Don't forget the '.') > ")
    else:
        vidFile = True
        print("\nYou have chosen to use a video file!")
        filePath = input("Please enter the path to the file! > ")

    tau = float(input("\nPlease set the social relaxation constant! (Tau) > "))
    L = int(input("Please enter the amount of frames used for determining the average optical flow in calculating the interaction force! > "))
    refScale = float(input("Please enter the scale by with the sum of interaction forces of a frame needs to be larger than the treshold! > "))

    inp2 = ''
    while inp2 != 'y' and inp2 != 'n':
        inp2 = input("\nWould you like to submit a clip or set of image frames to determine the treshold sum of interaction forces for anomaly detection? [y/n] > ")

    if inp2 == 'y':
        useExistingRef = True
        print("Is the example a clip or a set of images?")
        inp = ''
        while inp != 1 and inp != 2:
            inp = int(input("Enter 1 for images, 2 for video > "))

        if inp == 1:
            print("\nYou have chosen to use images as frames!")
            if vidFile:
                print("Please specify the path to the folder which contains the desired images.")
                print("WARNING: Image frame file names MUST be properly ennumered and represented with an N-digit counter. For example '001' or 'frame_001'.")
                print("In the case of the former (where it's a simple counter), please make sure to add a '\\' at the end of the folder path.")
                print("For example, if the frames were labeled as 'XXX': 'UCSD/Test/Test1/'")
                print("In the case of the latter, where there is a string of some kind preceding the counter, you must ALSO enter the part of the string.")
                print("For example, if the frames are labeled as 'seq_XXX', you would enter the path: 'mall_dataset/frames/seq_'")
            else:
                print("The same rules as with entering the path to the clip you want to analyze apply.")
            filePath1 = input("Please enter the folder path! > ")
            frameDigits1 = int(input("Please enter the number of digits your frame counter has. (For example, in the case of '001' it would be 3!) > "))
            extension1 = input("Please enter the file extension! (Don't forget the '.') > ")
        else:
            vidFile2 = True
            print("\nYou have chosen to use a video file!")
            filePath1 = input("Please enter the path to the file! > ")

        refF = getAvgInteractionForceSum(filePath1, extension=extension1, L=L, frameDigits=frameDigits1, vidFile=vidFile2, tau=tau)
    else:
        inp2 = ''
        while inp2 != 'y' and inp2 != 'n':
            inp2 = input("\nWould you like to manually set a treshold? [y/n] > ")

        if inp2 == 'y':
            useExistingRef = True
            refF = float(input("Please enter the treshold! > "))
        else:
            useExistingRef = False

    print("\nAll set!")
    anomalyDetect(filePath, extension=extension, L=L, useExistingRef=useExistingRef, refF=refF, refScale=refScale, frameDigits=frameDigits, vidFile=vidFile, tau=tau)