:: 将一文件夹内的所有wav文件合并为一个，使用时将该脚本复制到wav文件的文件夹内
:: Create File List
for %%i in (*.wav) do echo file '%%i'>> mylist.txt

:: Concatenate Files
ffmpeg -f concat -safe 0 -i mylist.txt -c copy output.wav