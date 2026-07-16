#----------------------------------------------------#
#   Single image prediction, camera detection, FPS testing,
#   directory prediction, and ONNX export combined in one script.
#   Switch between modes using the 'mode' variable below.
#----------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from deeplab import DeeplabV3

if __name__ == "__main__":
    #-------------------------------------------------------------------------#
    #   To modify class colors, edit self.colors in the __init__ function.
    #-------------------------------------------------------------------------#
    deeplab = DeeplabV3()
    #----------------------------------------------------------------------------------------------------------#
    #   mode specifies the prediction mode:
    #   'predict'       Single image prediction. For saving images or extracting objects, see comments below.
    #   'video'         Video detection using camera or video file. See comments below.
    #   'fps'           FPS benchmark test. Uses img/street.jpg by default. See comments below.
    #   'dir_predict'   Traverse a directory and save results. Default: img/ -> img_out/. See comments below.
    #   'export_onnx'   Export model to ONNX format. Requires PyTorch >= 1.7.1.
    #----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    #-------------------------------------------------------------------------#
    #   count           Whether to count pixels per class (area) and ratio.
    #   name_classes    Class names for printing category and count.
    #
    #   count and name_classes only take effect when mode='predict'.
    #-------------------------------------------------------------------------#
    count           = False
    # name_classes    = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    name_classes    = ["_background_","Black_stripe_dwarfism", "lodging"]
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          Video file path. Set to 0 for camera.
    #   video_save_path     Output video path. Leave "" to skip saving.
    #   video_fps           FPS for saved video.
    #
    #   These only take effect when mode='video'.
    #   Press Ctrl+C or let the video finish to complete the save.
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       Number of test frames for FPS measurement. Larger = more accurate.
    #   fps_image_path      Image used for FPS testing.
    #
    #   These only take effect when mode='fps'.
    #----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     Directory containing images to predict.
    #   dir_save_path       Directory to save prediction results.
    #
    #   These only take effect when mode='dir_predict'.
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    #-------------------------------------------------------------------------#
    #   simplify            Whether to simplify the ONNX model.
    #   onnx_save_path      Path to save the ONNX model.
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    if mode == "predict":
        '''
        Notes for predict mode:
        1. This script does not directly support batch prediction. For batch prediction, use
           os.listdir() to traverse a folder and Image.open() to load images.
           See get_miou.py for an example of traversal implementation.
        2. To save the result, use r_image.save("img.jpg").
        3. To show only the segmentation (no blending), set mix_type=1 in deeplab.py.
        4. To extract a region based on the mask, see detect_image() in deeplab.py:
           loop through each pixel's class and extract the corresponding region.
        seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
            seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
            seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = deeplab.detect_image(image, count=count, name_classes=name_classes)
                r_image.show()

    elif mode == "video":
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("Failed to read from camera/video. Check camera connection or video path.")

        fps = 0.0
        while(True):
            t1 = time.time()
            # Read one frame
            ref, frame = capture.read()
            if not ref:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            frame = Image.fromarray(np.uint8(frame))
            # Run detection
            frame = np.array(deeplab.detect_image(frame))
            # Convert RGB back to BGR for OpenCV display
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = deeplab.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = deeplab.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
    elif mode == "export_onnx":
        deeplab.convert_to_onnx(simplify, onnx_save_path)

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
