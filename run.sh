#!/bin/bash

source ../devel/setup.bash

echo "run..."
gnome-terminal --tab --title "trail_detect" -e 'bash -c "sleep 2; roslaunch trail_detect trail_detect.launch; exec bash;"' \
--tab --title "trail_control" -e 'bash -c "sleep 10; roslaunch trail_control trail_control.launch; exec bash;"' \