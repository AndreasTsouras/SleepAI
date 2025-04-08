import nvidia.dali.fn as fn
import nvidia.dali.pipeline as pipeline

# Parameters
BATCH_SIZE = 1
NUM_THREADS = 2
DEVICE_ID = 0
SEQUENCE_LENGTH = 16  # Number of frames in the sequence

class VideoPipeline(pipeline.Pipeline):
    def __init__(self, batch_size, num_threads, device_id, sequence_length, filenames):
        super(VideoPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        # Store the video reader operator.
        self.video_reader = fn.readers.video(
            device="gpu",
            filenames=filenames,
            sequence_length=sequence_length,
            random_shuffle=False
        )

    def define_graph(self):
        # Call the stored operator to get its output DataNode.
        # DO NOT add extra parameters here.
        output = self.video_reader()
        return output

if __name__ == '__main__':
    # Change "test_video.asf" to your test video path if necessary.
    pipe = VideoPipeline(
        batch_size=BATCH_SIZE,
        num_threads=NUM_THREADS,
        device_id=DEVICE_ID,
        sequence_length=SEQUENCE_LENGTH,
        filenames=["test_video.asf"]
    )
    pipe.build()
    outputs = pipe.run()
    # Convert the output to a numpy array.
    # Expected shape: [BATCH_SIZE, SEQUENCE_LENGTH, H, W, C]
    frames = outputs[0].as_array()
    print("DALI pipeline output shape:", frames.shape)
