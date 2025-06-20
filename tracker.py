from collections import defaultdict
import math

class Tracker:
    def __init__(self, max_distance=35, max_history=30):
        self.center_points = defaultdict(list)  # {id: [(x, y), (x, y), ...]}
        self.id_count = 0
        self.max_distance = max_distance
        self.max_history = max_history

    def update(self, objects_rect):
        objects_bbs_ids = []

        for rect in objects_rect:
            x1, y1, x2, y2 = rect
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            matched_id = None
            for obj_id, history in self.center_points.items():
                prev_cx, prev_cy = history[-1]  # last point
                dist = math.hypot(cx - prev_cx, cy - prev_cy)
                if dist < self.max_distance:
                    matched_id = obj_id
                    break

            if matched_id is not None:
                self.center_points[matched_id].append((cx, cy))
                if len(self.center_points[matched_id]) > self.max_history:
                    self.center_points[matched_id].pop(0)
                objects_bbs_ids.append([x1, y1, x2, y2, matched_id])
            else:
                self.center_points[self.id_count].append((cx, cy))
                objects_bbs_ids.append([x1, y1, x2, y2, self.id_count])
                self.id_count += 1

        # Prune unused IDs
        new_center_points = defaultdict(list)
        for _, _, _, _, obj_id in objects_bbs_ids:
            new_center_points[obj_id] = self.center_points[obj_id]

        self.center_points = new_center_points.copy()
        return objects_bbs_ids
