# CSC477-Final-Project

This is Jimmy, Morgan, and Yuvraaj's final project for CSC477H5 at UTM

## Overview
TODO: Fill out this readme with some more information

## File Structure
The repo is structured as:

-   `src` contains the files of implemented algorithms used to generate, simulate, and produce our findings.
-   `requirements.txt` contains the necessary python packages required to execute this project.

## How To Use
To execute any given planner run the following command in your terminal:

Linux/MacOS
```
python3 src/[your chosen planner] src/worlds/[your chosen pkl file]
```

Windows
```
python src/[your chosen planner] src/worlds/[your chosen pkl file]
```

For example, to run the RRT* planner on the provided world `map.pkl` on Linux, do:
```
python3 src/rrt_star_planner.py src/worlds/map.pkl
```

To specify the start and goal destination coordinates on your map, edit the `start_state` and `dest_state` variables in the main function of your chosen planner.

To exit the visualization, press any key.
