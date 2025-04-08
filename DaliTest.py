import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.pipeline as pipeline
import numpy as np

# Parameters
batch_size = 1
num_threads = 2
device_id = 0
sequence_length = 16  # Number of frames per sequence

class VideoPipeline(pipeline.Pipeline):
    def __init__(self, batch_size, num_threads, device_id, sequence_length, filenames):
        super(VideoPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.input = fn.readers.video(
            device="gpu",
            filenames=filenames,
            sequence_length=sequence_length,
            random_shuffle=False,
            shard_id=0,
            num_shards=1,
            initial_fill=16,     # number of frames to prefill the buffer
            file_list=None,
        )

    def define_graph(self):
        output = self.input(name="Reader")
        return output

# Replace "test_video.asf" with your video file path.
video_file = "test_video.asf"
pipe = VideoPipeline(batch_size=batch_size,
                     num_threads=num_threads,
                     device_id=device_id,
                     sequence_length=sequence_length,
                     filenames=[video_file])
pipe.build()
output = pipe.run()
# Convert the output of the first batch to numpy array
frames = output[0].as_array()  # shape: [batch_size, sequence_length, H, W, C]
print("DALI pipeline output shape:", frames.shape)
