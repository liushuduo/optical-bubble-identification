:: 提取inVideo.mp4中的某时间(-ss 后加时间格式为hh:mm:ss)的连续帧(-vframes 后加需要提取的帧数)，
ffmpeg -ss 00:00 -i inVideo.mp4 -vframes 15 out-%03d.png
