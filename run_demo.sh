# 1. Run Masked Droid SLAM (also detect+track humans in this step)
python scripts/estimate_camera.py --video "./example_video.mov" 
# -- You can indicate if the camera is static. The algorithm will try to catch it as well.
python scripts/estimate_camera.py --video "./another_video.mov" --static_camera

# 2. Run 4D human capture with VIMO.
python scripts/estimate_humans.py --video "./example_video.mov"

# 3. Put everything together. Render the output video.
python scripts/visualize_tram.py --video "./example_video.mov"
