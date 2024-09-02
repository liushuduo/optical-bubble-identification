ffmpeg -skip_frame nokey -i test.mp4 -vsync vfr -frame_pts false out-%02d.png
::-vsync vfr: discard the unused frames
::-frame_pts true: use the frame index for image names, otherwise, the index starts from 1 and increments 1 each time