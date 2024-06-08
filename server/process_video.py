import sys
if 'yolov9' not in sys.path:
    sys.path.append('yolov9')

import cv2

# ML/DL
import numpy as np
import torch

# YOLOv9
from models.common import DetectMultiBackend, AutoShape
from utils.general import set_logging

from supervision import Detections as BaseDetections
from supervision.config import CLASS_NAME_DATA_FIELD
import supervision as sv

import util.line_zone_custom as lc
from util.enum_direction_movement import DirectionMovement as enum_dm


class ExtendedDetections(BaseDetections):
    @classmethod
    def from_yolov9(cls, yolov9_results) -> 'ExtendedDetections':
        xyxy, confidences, class_ids = [], [], []

        for det in yolov9_results.pred:
            for *xyxy_coords, conf, cls_id in reversed(det):
                xyxy.append(torch.stack(xyxy_coords).cpu().numpy())
                confidences.append(float(conf))
                class_ids.append(int(cls_id))

        class_names = np.array([yolov9_results.names[i] for i in class_ids])

        if not xyxy:
            return cls.empty()

        return cls(
            xyxy=np.vstack(xyxy),
            confidence=np.array(confidences),
            class_id=np.array(class_ids),
            data={CLASS_NAME_DATA_FIELD: class_names},
        )


def initialize_model():

    weights = 'yolov9/weights/gelan-e.pt'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend(weights=weights, device=device, fuse=True)
    model = AutoShape(model)
    model.conf = 0.4#0.5 #2.5
    model.iou = 0.7 # Valores mais baixos resultam em menos detecções, eliminando caixas sobrepostas, úteis para reduzir duplicatas
    model.classes = [19]
    model.agnostic_nms = False
    model.max_det = 1000
    return model



class ProcessVideo():

    def __init__(self, video, direction_movement):

        self.model = initialize_model()
        self.size_frame = 320
        self.cattle_weight = 0
        self.cattle_counting = 0
        self.video = video
        self.direction_movement = direction_movement
        self.create_video_info()
        self.create_byte_tracker()
        self.create_line()
        self.create_line_annotator()
        self.create_label_annotator()

    def prepare_yolov9(self):

        self.model.conf = self.conf
        self.model.iou = self.iou
        self.model.classes = self.classes
        self.model.agnostic_m = self.agnostic_nms
        self.model.max_det = self.max_det


    def create_video_info(self):
        
        self.video_info = sv.VideoInfo.from_video_path(self.video)


    def create_byte_tracker(self):

        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=0.25, 
            lost_track_buffer=100, #250
            minimum_matching_threshold=0.95, 
            frame_rate=self.video_info.fps
        )


    def get_cattle_weighing(self, frame):
        
        print('PESAGEM')
        return "DESATIVADO"


    def create_line(self):

        '''
        line_start = sv.Point(650, 680)
        line_end = sv.Point(650, 0) 
        '''

        # video 1000073760.mp4
        #line_start = sv.Point(0, 600)
        #line_end = sv.Point(480, 600) 


        # video 1000073781.mp4
        line_start = sv.Point(0, 350)
        line_end = sv.Point(355, 350) 

        self.line = lc.LineZoneCustom(
            start=line_start, 
            end=line_end,
            direction_movement = self.direction_movement,
            interest_object = 0.5
        )


    def create_line_annotator(self):

        if self.direction_movement == enum_dm.TOP_TO_DOWN or self.direction_movement == enum_dm.LEFT_TO_RIGHT:
           
            self.line_annotator = sv.LineZoneAnnotator(
                thickness = 2,
                text_thickness = 1,
                text_scale = 0.4,
                custom_out_text = 'Quantidade',
                display_in_count = False
            )
        else:

           self.line_annotator = sv.LineZoneAnnotator(
                thickness = 2,
                text_thickness = 1,
                text_scale = 0.4,
                custom_in_text = 'Quantidade',
                display_out_count = False
            )

    def create_label_annotator(self):

        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.5, 
            color=sv.ColorPalette.ROBOFLOW,
            #color_lookup=sv.ColorLookup.CLASS, 
            text_position=sv.Position.CENTER
        )


    def create_annotated_frame(self, annotated_frame, detections):
      
        labels = [
            f"{self.model.model.names[class_id]} {confidence:0.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        return self.label_annotator.annotate(
            scene=annotated_frame.copy(), 
            detections=detections, 
            labels=labels
        )


    def process_frame(self, frame, show_analyze=False):

        frame_rgb = frame[..., ::-1]
        results = self.model(frame_rgb, size=self.size_frame, augment=False)
        detections = ExtendedDetections.from_yolov9(results)
        return detections

    '''
    def resized(self, frame):

        width = int(self.video_info.width // 1.25)
        height = int(self.video_info.height // 1.25)

        return cv2.resize(frame, ( width, height))
    '''

    def process(self, frame, analyze=True, show_analyze=False):

        if analyze:
            detections = self.process_frame( frame, show_analyze)
            detections = self.byte_tracker.update_with_detections(detections)

            self.line_annotator.annotate(frame, self.line )
            self.line.trigger(detections=detections)

            if self.direction_movement == enum_dm.TOP_TO_DOWN or self.direction_movement == enum_dm.LEFT_TO_RIGHT:
                
                if( self.cattle_counting != self.line.out_count):
                    self.cattle_counting = self.line.out_count
                    self.cattle_weight = self.get_cattle_weighing(frame)
            else:
                if( self.cattle_counting != self.line.in_count):
                    self.cattle_counting = self.line.in_count
                    self.cattle_weight = self.get_cattle_weighing(frame)

               

            if show_analyze:
                return self.create_annotated_frame(frame, detections)
            else:
                return frame
        else: 
            self.line_annotator.annotate(frame, self.line )
            return frame


    def start(self, frame, show_analyze=False):
        return self.process(frame, show_analyze=show_analyze), self.cattle_counting, self.cattle_weight


    def show(self, frame):
        return self.process(frame, analyze=False), self.cattle_counting, self.cattle_weight


# Otimizações específicas do OpenCV
cv2.setUseOptimized(True)
cv2.setNumThreads(4)



def test_video(video_path:str):

    process_video = ProcessVideo(video, 
        direction_movement=enum_dm.TOP_TO_DOWN)

    cap = cv2.VideoCapture(video)

    # Configurações do vídeo de saída
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter('video2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width,frame_height))

    while cap.isOpened():

        success, frame = cap.read()

        if success:

            annotated_frame, cattle_counting, cattle_weight = process_video.start(frame, show_analyze=True)
            #cv2.imshow("Detections", annotated_frame)
            out.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    print('FIM')
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    video = "./videos/1000073760.mp4"
    test_video(video)

   
