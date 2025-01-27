#!/usr/bin/bash

v4l2-ctl -d /dev/video0 --set-ctrl=auto_exposure=1
v4l2-ctl -d /dev/video0 --set-ctrl=exposure_dynamic_framerate=0
v4l2-ctl -d /dev/video0 --set-ctrl=exposure_time_absolute=250
v4l2-ctl -d /dev/video0 --set-ctrl=white_balance_automatic=0
v4l2-ctl -d /dev/video0 --set-ctrl=white_balance_temperature=4600
v4l2-ctl -d /dev/video0 --set-ctrl=focus_automatic_continuous=0
v4l2-ctl -d /dev/video0 --set-ctrl=gain=50
