import __load_modules  # noqa
import time
import cv2
import tflite_runtime.interpreter as tflite
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
from src.category import read_label_pbtxt
from src.utils import VideoStream, draw_boxes, preprocess_input

flags.DEFINE_string("model", None, "path to model inference graph")
flags.DEFINE_string("output", "data/outputs/ipcam_output.avi", "path to output video")
flags.DEFINE_string("ip", None, "IP address of camera")
flags.DEFINE_integer("port", 8080, "Port in which camera is running")
flags.DEFINE_string("username", None, "Username to access camera stream")
flags.DEFINE_string("password", None, "Password to access camera stream")
flags.DEFINE_string("labels", None, "path to label.txt file")


def main(_argv):
    flags.mark_flag_as_required("model")
    flags.mark_flag_as_required("ip")
    flags.mark_flag_as_required("labels")

    stream_url = f"rtsp://{FLAGS.ip}:{FLAGS.port}/h264_ulaw.sdp"
    if FLAGS.username and FLAGS.password:
        stream_url = f"rtsp://{FLAGS.username}:{FLAGS.password}@{FLAGS.ip}:{FLAGS.port}/h264_ulaw.sdp"

    labels = read_label_pbtxt(FLAGS.labels)

    start_time = time.time()
    interpreter = tflite.Interpreter(FLAGS.model)
    end_time = time.time()
    logging.info("model loaded")
    logging.info(f"Elapsed time: {str(end_time - start_time)}sec")

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    h = input_details[0]["shape"][1]
    w = input_details[0]["shape"][2]

    interpreter.resize_tensor_input(input_details[0]["index"], [1, 320, 320, 3])
    interpreter.allocate_tensors()

    start_time = time.time()
    cap = VideoStream(cam=stream_url).start()

    width = int(cap.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = None
    if FLAGS.output:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    while True:
        img = cap.read()
        img = preprocess_input(img)
        image_tensor = np.expand_dims(img, axis=0)
        interpreter.set_tensor(input_details[0]["index"], image_tensor)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[3]["index"])[0]
        classes = interpreter.get_tensor(output_details[4]["index"])[0]
        scores = interpreter.get_tensor(output_details[0]["index"])[0]

        output_image = draw_boxes(
            img.copy(),
            boxes,
            classes,
            scores,
            labels,
            h,
            w,
            min_threshold=FLAGS.threshold,
        )
        output_image = cv2.resize(output_image, (width, height))
        cv2.imshow("Object Detection", output_image)
        if out:
            out.write(output_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.stop()
    end_time = time.time()
    logging.info(f"Elapsed time: {str(end_time - start_time)}sec")


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
