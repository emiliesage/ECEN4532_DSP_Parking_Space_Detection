# Parking Lot Detection

## Prefface

The goal of this project is to develop a methodology to detect open parking spaces from a camera in real time based on methods that do not utalize machine learning. The significance of this project is twofold. Parking space detection is useful in poplulated areas, especially university campuses, and allowing people to find available parking prior to ariving at the lot is important to saving time and planning a day. This development for time and memory efficent adaptive algorithms allows for projects such as this to be run on any platform. The begining of this project, the development to the point of writing this report, is focused on developing a prototype of the algorithm in Matlab. This prototyping phase allows for quick changes and itterative development to refine the algorithm, and develop a working method.

## Instructions for Using the Program

Included in this project is the matlab file for the developed method. To implement this with different data the program requires a .mat file that contains video data in grayscale format. From this the program will prompt to define parking spaces, and these defined spaces are saved to be used later for future runs of the program. The output of the program is a video with a timelapse of the changes in the occupacy of the different detection methods, plots comparing the two methods, and statistical data from the images.

## Methodology
