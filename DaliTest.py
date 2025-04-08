import nvidia.dali.pipeline as pipeline
import nvidia.dali.fn as fn
import numpy as np

# Parameters
BATCH_SIZE = 1
NUM_THREADS = 2
DEVICE_ID = 0
SEQUENCE_LENGTH = 16  # frames

class VideoPipeline(pipeline.Pipeline):
    def __init__(self, batch_size, num_threads, device_id, sequence_length, filenames):
        super(VideoPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        # Create the video reader operator; do not call it yet.
        self.reader = fn.readers.video(
            device="gpu",
            filenames=filenames,
            sequence_length=sequence_length,
            random_shuffle=False
        )

    def define_graph(self):
        # Simply call the stored operator without extra keyword parameters.
        output = self.reader()
        return output

# Replace "test_video.asf" with the path to your video file.
video_file = "test_video.asf"

# Create and build the pipeline.
pipe = VideoPipeline(
    batch_size=BATCH_SIZE,
    num_threads=NUM_THREADS,
    device_id=DEVICE_ID,
    sequence_length=SEQUENCE_LENGTH,
    filenames=[video_file]
)
pipe.build()

# Run the pipeline.
output = pipe.run()
frames = output[0].as_array()  # Expected shape: [BATCH_SIZE, SEQUENCE_LENGTH, H, W, C]

print("DALI pipeline output shape:", frames.shape)
