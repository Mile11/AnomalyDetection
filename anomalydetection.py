import cv2
import numpy as np
import os.path
import random


# Creates a set of particles, which contain a set of coordinates to the frame
# particleNum -- number of particles to create
# imageDims -- dimensions of the image, to know the interval of the random initial positions
def createParticles(particleNum, imageDims):
    xCoords = np.atleast_2d(np.random.randint(imageDims[0]-1, size=particleNum))
    yCoords = np.atleast_2d(np.random.randint(imageDims[1]-1, size=particleNum))

    particles = np.concatenate((xCoords, yCoords), axis=0)
    return particles.T


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


# Determine the optimal position of the particles

# NOT ACTUALLY IMPLEMENTED!! ONLY CALCULATES THE INTERACTION FORCES OF A FRAME AND RETURNS THE VALUES OF
# THOSE INTERACTION FORCES FOR THE GIVEN PARTICLES!!

# I DON'T KNOW HOW TO GET THIS THING TO WORK PROPERLY!!

# particles -- an array of particle positions
# opticFlows -- a list of dense optical flows from previous frames
# iterNum -- number of iterations for the hive algorithm
# L -- number of previous frames to look at when determining the average optical flow
# inertia -- parameter W, used for determining the balance of previous particles and new ones
# C1, C2 -- acceleration constants
def particleHiveAlgorithm(particles, opticFlows, L, iterNum=100, inertia=0.2, C1=0.6, C2=0.002):

    # To speed up the process, we can use the dense optical flow to calculate the interaction force for any position on
    # the current frame, and then simply look at the positions as we need to during the algorithm
    interactionForces = calcInteractionForces(opticFlows, L)

    return particles, getInteractionForcesForParticles(particles, interactionForces)

    # print(interactionForces)
    # personalBests = np.copy(particles)
    # bestForces = getInteractionForcesForParticles(particles, interactionForces)
    #
    # # In case, for some reason, the iteration number is set to 0
    # newForces = bestForces
    # newParticles = particles
    #
    # # Parameters related to the hive algorithm
    # globalSmallestForce = np.amax(bestForces)
    # globalBest = particles[np.argmax(bestForces)]
    # particleNum = np.size(particles, 0)
    #
    # # Preparation to begin
    # oldParticles = particles
    # oldVelocity = 0
    #
    # for i in range(iterNum):
    #     newVelocity = inertia*oldVelocity + C1*np.random.rand(particleNum, 2)*(personalBests - oldParticles) + C2*np.random.rand(particleNum, 2)*(globalBest - oldParticles)  # Calculate new velocity
    #     newParticles = oldParticles + newVelocity  # Move all particles for their respective new velocities
    #     newParticles = newParticles.astype(int)  # Make all of the new particles into ints (actual usable positions)
    #     newForces = getInteractionForcesForParticles(newParticles, interactionForces)  # Find the interaction forces on those positions
    #     smallestForce = np.amin(newForces)  # Find the smallest force among the found forces
    #
    #     # Update global force if new smallest interaction force is found
    #     if smallestForce > globalSmallestForce:
    #         globalSmallestForce = smallestForce
    #         globalBest = newParticles[np.argmax(newForces)]
    #
    #     for j in range(np.size(newParticles, 0)):
    #         if newForces[j] > bestForces[j]:
    #             bestForces[j] = newForces[j]
    #             personalBests[j] = newParticles[j]
    #
    #     # Prepare for next iteration
    #     oldParticles = newParticles
    #     oldVelocity = newVelocity

    # return newParticles, newForces


# RANSAC algorithm
# Approximates a gaussian curve based on force magnitudes by taking different samples, intending to find the one
# with the least outliers
# Returns all the outliers for such a curve
# particles -- particles
# forces -- interaction forces
# R -- number of iterations (the paper has it at 1000, but the algorithm runs too slow)
def RANSAC(particles, forces, R=5):

    helpForce = np.copy(forces)
    samplSize = len(helpForce)//3
    outlierNum = len(helpForce)
    ret = []

    for i in range(R):
        random.shuffle(helpForce)

        alsoHelp = helpForce[:samplSize]
        mean = np.sum(alsoHelp) / samplSize
        stddev = np.sqrt(np.sum((alsoHelp - mean)**2) / (samplSize-1))
        tobeat = mean + 3*stddev

        outliers = [(p, f) for (p, f) in zip(particles, forces) if f >= tobeat]
        if len(outliers) <= outlierNum:
            ret = outliers
            outlierNum = len(outliers)

    return zip(*ret)


# Anomaly detection function
# folderName -- path to the folder containing the .tif frames
# extension -- extension frame images have
# particleNum -- number of particles to use
# L -- number of frames used for determining the average optical flow
def anomalyDetect(folderName, extension, particleNum, L):

    # INITIAL PREPARATION

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
    particles = createParticles(particleNum, prev.shape)

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

        if i > 2:
            particles, forces = particleHiveAlgorithm(particles, opticFlows, L)
            outlierparticles, outlierforces = RANSAC(particles, forces)
            outlierparticles = list(outlierparticles)

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


anomalyDetect('Test001', '.tif',  20000, 10)