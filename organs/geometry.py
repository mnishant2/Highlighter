'''
This module is going to contain the common functions of geometry to be used in the ckyc project
'''

import math
from operator import itemgetter
import organs.arithmetic

# this function calculates distance between two points.
def distance(point1, point2):
    return math.hypot(point1[0]- point2[0], point1[1]- point2[1])
    pass

# Below function calculates area of a polygon defined as below:
# [[x1, y1], [x2, y2], [x3, y3]]

def polygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

# This function check if a given 4 edged polygon is equivalent to a approximate square
def checkSquare(points):
    distances = [distance(points[0], points[1]), distance(points[1], points[2]), distance(points[2], points[3]), distance(points[3], points[0])]
    # print distances
    for distance1 in distances:
        for distance2 in distances:
            if arithmetic.equivalent(distance1, distance2) == False:
                return False
                pass
            pass
        pass
    return True
    pass

# The below function tells if a set of points is near to (0,0) which is the origin
# This is a funny function not sure why I created that earlier - Ankur
def anyPointAboutZero(points):
    for point in points:
        if point[0] < 2 or point[1] < 2:
            # print "Haan 0 mila"
            # print point[0]
            # print point[1]
            return True
            break
            pass
        pass
    pass
    return False

# This takes a triangle as input and tell what its orientation is

def getTriangleOrientation(points):
    orientationPoints = sorted(points,key=itemgetter(1))
    d1 = orientationPoints[0][1] - orientationPoints[1][1]
    d2 = orientationPoints[1][1] - orientationPoints[2][1]
    if d1 > d2:
        return 'down'
        pass
    else:
        return 'up'
        pass
    pass


def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]
def ang(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    # Get cosine value
    cos_ = dot_prod/magA/magB
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod/magB/magA)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle)%360

    if ang_deg-180>=0:
        # As in if statement
        return 360 - ang_deg
    else:

        return ang_deg
