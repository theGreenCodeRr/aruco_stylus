#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
from cv2 import aruco

# 4x4 markers (using 50th's ID)
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

def main():
    # Create 10 markers
    for i in range(12):

        ar_image = aruco.generateImageMarker(dictionary, i, 150)  # ID=i ，markerSize=150x150px．

        fileName = "ar" + str(i).zfill(2) + ".png"      # filename

        cv2.imwrite(fileName, ar_image)        # Save marker images

if __name__ == "__main__":
    main()