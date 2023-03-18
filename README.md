# CSE455 Final
A Rubiks cube color extractor

### Problem Description
Entering face colors for a given Rubiks cube state is generally manual and labor intensive, as each of the 9 * 6 facelets on the cube must be keyed in manually. The goal of the project is to create a program that can identify the facelets and get their colors automatically, requiring the user only to show each of the 6 sides to the camera in order to read in the full state of the cube.

### Previous Work
This project was largely inspired by Andrej Karpathy's cube extractor project, which he made in 2011. Unfortunately, the link is now dead, which is the main motivation for creating this project on my own in Python.

Since this relies on classical computer vision techniques, there aren't a lot of previous work this project is based on, but I do take inspiration from Andrej's project, as well as some other projects on Github which try to do something similar (https://github.com/kkoomen/qbr).

### My Approach
The main additional goal I wanted to strive for was to make the program automatically invariant to cube color, which some other projects on Github I noticed were unable to achieve without explicit user input identifying the cube as either white or black (white or black meaning the base color of the cube when the stickers are not attached). 

Overall, the program reads in frames using the webcam and OpenCV2, and does the following:

1. The image is resized to speed up the runtime
2. A bilateral filter is applied to the image in order to prevent flickering and false edges
3. A canny edge detector is applied to the image
4. The edges detected by the canny edge detector are dilated so that they join together a bit better, filling in any possible gaps
5. Contours are found, preserving their hierarchy, as well as keeping all points in the contour, instead of a reduced representation.
6. For each contour, we check if it is "square-like", which uses a custom heuristic that finds the rotated rectangular bounding box of the contour, and checks if the area of the bounding box is within some percentage of the area of the contour. It also checks that the width and height of the rectangular bounding box are sufficiently similar. These "square-like" contours are likely stickers on the facelets of the cube.
7. For each of these square like contours, the center of mass is calculated, and a point is associated with that square.

At this point, we can have up to 9 points on the cube, but we may have less if not all facelets were discovered on a given side. The remainder of the algorithm attempts to extrapolate the positions of the remaining un-detected points, using whatever `n` points it has so far. The intuition is to find a bounding box for the entire cube, which we can then sample colors at locations 1/3 or 2/3 across the bounding box to get the facelet colors. In order to find the cube bounding box, we search for a grid within the points we have so far. The intuition is that lines aligned with the cube grid will intersect more points than those that aren't. More concretely:

1. For each point, we draw a line to every other point. This line has a direction, which is just the normalized vector pointing from the point to another point. We search through a list of other possible grid axes. If the current line is roughly parellel to an existing candidate grid axis, that candidate grid axis gains a "vote". The same happens if the  current line is roughly perpendicular to an existing candidate grid axis. If this current line was only parallel to another candidate, or it was neither parallel nor perpendicular to any candidate, it is added to the list of candidate grid axes.
2. We sort the list of candidates by votes and pick the candidate with highest votes. We then go down the list and pick the first candidate that is roughly perpendicular the highest vote candidate. These two candidates are the x and y axes of our imaginary coordinate system.
3. We need to anchor the coordinate system at one of the points, such that a box drawn using the x and y axes as two of its side lengths would bound the Rubiks cube. To do this, we anchor the coordinate system at every point that we detect. If the point is a valid anchor point, all the other detected points should lie in the first quadrant of the coordinate system. We return the first such valid anchor point.
4. We find the furthest detected point from the anchor point in both the x and y axis directions, and draw a bounding box around the cube.
5. We partition a grid so that a point hopefully lies on each facelet.
6. We draw a small circle with radius of 10 pixels and use that circle to find the mean color. 
7. The mean color is the color of each facelet. 

At this point, we have recovered the colors of the cube, and can record them elsewhere.

### Datasets
No datasets were used in this project.

### Results
The program is able to automatically get the colors of both black and white cubes, in mostly controlled lighting conditions. 

### Discussion
- What problems did I encounter?
Finding an algorithm that could consistently detect facelets was actually much harder than I was expecting. The black and white cubes both presented their own challenges: while the black cube had excellent contrast on yellow and white stickers, the darker blue and red colors made it hard to detect good edges. Similarly, for the white cube, the white and yellow stickers were hard to detect edges for.

Once I had some facelet points, the issue of actually reconstructing a grid was also quite difficult, and I struggled with that for a long time. Being able to reconstruct a grid was really important as it made the algorithm resilient to situations when it could not detect all 9 facelets.

- Next steps?
I initially intended to create this in Rust, but had to abandon that both because of issues with getting the OpenCV2 bindings to work properly, and because Python was much faster to prototype in. With more time I would eventually like to port this over to Rust.

Additionally, this program is mostly a proof of concept, as it can only detect colors, but doesn't gather any additional useful metadata, like which side was presented, or actually recosntructing the full cube state from the images. However, these should be relatively simple add-ons, and I hope to add those in soon.

- How does approach differ from others?
This project differs from others mostly in trying to find a coordinate system based on a series of points, which allows the program to operate with only 4-5 detected facelets.
