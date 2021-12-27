import __load_modules  # noqa
import time
import os
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from absl import app, flags, logging
from absl.flags import FLAGS
from src.utils import draw_boxes
from src.category import read_label_pbtxt
from src.utils import load_image, preprocess_input

flags.DEFINE_string("model", None, "path to tflite model")
flags.DEFINE_string("image", None, "path to input image")
flags.DEFINE_string(
    "output", "data/outputs/detection_output.jpg", "path to output image"
)
flags.DEFINE_float("threshold", 0.5, "detection threshold")


def main(_argv):
    flags.mark_flag_as_required("model")
    flags.mark_flag_as_required("image")

    labels = read_label_pbtxt(os.path.join(FLAGS.model, "labelmap.txt"))

    interpreter = tflite.Interpreter(os.path.join(FLAGS.model, "detect.tflite"))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]["shape"][1]
    width = input_details[0]["shape"][2]

    image = load_image(FLAGS.image, (width, height))
    image = preprocess_input(image)

    image_np = np.expand_dims(image, axis=0)

    start_time = time.time()
    interpreter.set_tensor(input_details[0]["index"], image_np)
    interpreter.invoke()
    end_time = time.time()

    boxes = interpreter.get_tensor(output_details[0]["index"])[0]
    classes = interpreter.get_tensor(output_details[1]["index"])[0]
    scores = interpreter.get_tensor(output_details[2]["index"])[0]

    output_image = draw_boxes(
        image.copy(),
        boxes,
        classes,
        scores,
        labels,
        height,
        width,
        min_threshold=FLAGS.threshold,
    )

    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(FLAGS.output, output_image)
    cv2.imshow("Detection Output", output_image)
    cv2.waitKey(0)
    logging.info(f"Elapsed time: {str(end_time - start_time)}sec")


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
