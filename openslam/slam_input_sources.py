import cv2


class video_source(object):

    def __init__(self, video_path, skip_frames=0):
        self.video_stream = cv2.VideoCapture(video_path)
        if not self.video_stream.isOpened():
            if skip_frames > 0:
                cv2.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
        else:
            print("Couldn't open video!", video_path)

    def get_next_frame(self, frame):
        """Return the next frame in the video stream"""
        ret, frame = self.video_stream.read()
        return ret


class image_source(object):

    def __init__(self, dataset):
        #self.folder_path = dataset.data_path
        self.dataset = dataset
        self.sorted_image_list = sorted(dataset.image_list)
        """Creates a list of imfile names to read from"""
        self.curr_frame_n = 0

    def get_next_frame(self):
        """Return the next frame in the file list"""
        if (self.curr_frame_n < len(self.sorted_image_list)):
            print("Reading frame: ", self.sorted_image_list[self.curr_frame_n])
            print("data path: ", self.dataset.data_path)
            frame = self.dataset.load_image(
                        self.sorted_image_list[self.curr_frame_n])
        else:
            frame = None
        self.curr_frame_n += 1
        return frame #frame is not None
