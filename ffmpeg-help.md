ffmpeg -r 1/24 -i image0000-step%05d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p finding-nathan.mp4


ffmpeg -framerate 24 -pattern_type glob -i '*step*.png' -c:v libx264 -r 24 -pix_fmt yuv420p finding-nathan.mp4



