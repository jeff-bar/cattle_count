import supervision as sv
from typing import Tuple
import numpy as np

from supervision.detection.core import Detections
from supervision.geometry.core import Point, Position

from util.enum_direction_movement import DirectionMovement 

class LineZoneCustom(sv.LineZone):

    def __init__(
        self,
        start: Point,
        end: Point,
        direction_movement: DirectionMovement,
        interest_object: float = 0.5
    ):

        triggering_anchors =  (
            Position.TOP_LEFT,
            Position.TOP_RIGHT,
            Position.BOTTOM_LEFT,
            Position.BOTTOM_RIGHT,
        )

        if( interest_object < 0.1 or interest_object > 0.9):
            raise ValueError("The value of {} is invalid, it must be between 0.1 and 0.9.".format(interest_object))

        super().__init__(start, end, triggering_anchors)
        self.direction_movement = direction_movement
        self.interest_object = interest_object


    def trigger(self, detections: Detections) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update the `in_count` and `out_count` based on the objects that cross the line.

        Args:
            detections (Detections): A list of detections for which to update the
                counts.

        Returns:
            A tuple of two boolean NumPy arrays. The first array indicates which
                detections have crossed the line from outside to inside. The second
                array indicates which detections have crossed the line from inside to
                outside.
        """
        crossed_in = np.full(len(detections), False)
        crossed_out = np.full(len(detections), False)

        if len(detections) == 0:
            return crossed_in, crossed_out


        ### modificação do código original 
        half_bboxes = []
        tracker_ids = []

        for i, bbox in enumerate(detections.xyxy):
            x1, y1, x2, y2 = bbox
            if( self.direction_movement == DirectionMovement.TOP_TO_DOWN ):
                y1 = int((y1 + y2) * ( 1 - self.interest_object))  # Considera a porcentagem da altura. Usa a metade inferior
            elif( self.direction_movement == DirectionMovement.DOWN_TO_TOP):
                y1 = int((y1 + y2) * self.interest_object) # Considera a porcentagem da altura. Usa a metade superior
            elif( self.direction_movement == DirectionMovement.RIGHT_TO_LEFT):
                x2 = x1 + ((x2 - x1) * ( self.interest_object))   # Considera a porcentagem da largura. Usa a metade esquerda
            elif( self.direction_movement == DirectionMovement.LEFT_TO_LEFT):
               x2 = x1 + ((x2 - x1) * ( 1 - self.interest_object))   # Considera a porcentagem da largura. Usa a metade direita
            else:
                raise ValueError("The value of {} is invalid.".format(self.direction_movement))
                
            half_bboxes.append([x1, y1, x2, y2])

            if i < len(detections.tracker_id) and detections.tracker_id[i] is not None:
                tracker_ids.append(detections.tracker_id[i])


        half_bboxes = np.array(half_bboxes)
        half_detections = Detections(
            xyxy=half_bboxes,
            confidence=detections.confidence,
            class_id=detections.class_id,
            tracker_id=np.array(tracker_ids) if tracker_ids else np.array([-1]) 
        )

        ### fim da modificação do código original 

        all_anchors = np.array(
            [
                half_detections.get_anchors_coordinates(anchor)
                for anchor in self.triggering_anchors
            ]
        )

        for i, tracker_id in enumerate(half_detections.tracker_id):
            if tracker_id is None:
                continue

            box_anchors = [Point(x=x, y=y) for x, y in all_anchors[:, i, :]]

            in_limits = all(
                [
                    self.is_point_in_limits(point=anchor, limits=self.limits)
                    for anchor in box_anchors
                ]
            )

            if not in_limits:
                continue

            triggers = [
                self.vector.cross_product(point=anchor) < 0 for anchor in box_anchors
            ]

            if len(set(triggers)) == 2:
                continue

            tracker_state = triggers[0]

            if tracker_id not in self.tracker_state:
                self.tracker_state[tracker_id] = tracker_state
                continue

            if self.tracker_state.get(tracker_id) == tracker_state:
                continue

            self.tracker_state[tracker_id] = tracker_state
            if tracker_state:
                self.in_count += 1
                crossed_in[i] = True
            else:
                self.out_count += 1
                crossed_out[i] = True

        return crossed_in, crossed_out