from ultralytics import FastSAM

# Create a FastSAM model
model = FastSAM("FastSAM-s.pt")  # or FastSAM-x.pt

# Track with a FastSAM model on a video
results = model.track(source="Crawling/Video Crawling/NBA Videos/precious-achiuwa/precious-achiuwa_video_0.mp4", imgsz=640)