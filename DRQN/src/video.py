import subprocess as sp
import cv2
# import cv2.cv as cv
import numpy as np

class VideoWriter:
    # class Error(Exception):
    #     pass

    # def __init__(self, w, h, fps=25, output_file="video.mp4"):
    #     self.command = [
    #         'ffmpeg',
    #         '-y',                    # overwrite output file if it exists
    #         '-f', 'rawvideo',
    #         '-vcodec', 'rawvideo',
    #         '-s', '%dx%d' % (w, h),  # size of one frame
    #         '-pix_fmt', 'bgr24',
    #         '-r', str(fps),          # frames per second
    #         '-i', '-',               # The imput comes from a pipe
    #         '-an',                   # Tells FFMPEG not to expect any audio
    #         '-vcodec', 'mpeg4',
    #         output_file,
    #     ]
    #     self.ffmpeg = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)

    # def add_frame(self, frame):
    #     try:
    #         self.ffmpeg.stdin.write(frame.tostring())
    #     except IOError:
    #         reason = self.ffmpeg.stderr.read()
    #         raise self.Error(reason)

    # def close(self):
    #     self.ffmpeg.terminate()


    def __init__(self, w, h, fps=25, output_file="video.avi"):
        self.writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MJPG'),fps,(w,h))


    def add_frame(self, frame):
        self.writer.write(frame)

    def close(self):
        self.writer.release()

# Usage example
if __name__ == "__main__":
    # import numpy as np

    # w, h = 640, 480
    # video = VideoWriter(w, h)
    # for i in range(100):
    #     video.add_frame(np.random.randint(0, 255, (h, w, 3)))

    writer = cv2.VideoWriter('test1.mp4',cv2.VideoWriter_fourcc(*'MJPG'),25,(640,480))
    for i in range(1000):
        x = np.random.randint(255,size=(480,640)).astype('uint8')
        x = np.repeat(x,3,axis=1)
        x = x.reshape(480, 640, 3)
        writer.write(x)
    writer.release()

