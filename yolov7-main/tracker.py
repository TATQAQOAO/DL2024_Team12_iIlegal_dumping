import numpy as np
from collections import OrderedDict, Counter
from typing import Tuple, Optional
from scipy.spatial.distance import cdist
from utils.general import xyxy2xywh


def split_bag_persons(centers, labels):
    # 将检测结果分成垃圾袋和人
    #print("進來split_bag_person\n")
    bag_centers = centers[labels == 'garbage']
    person_centers = centers[labels == 'person']
    return bag_centers, person_centers

class SimpleTracker:
    """
    Location based tracker for person and garbage.
    Also does location based bag-person association.
    """

    def __init__(self, self_association_thres, bag_person_thres):
        # type: (float, float) -> None
        """
        Initialize tracker.

        Args:
            self_association_thres: threshold value for person-person and bag-bag association between frames in pixel
                units.
            bag_person_thres: threshold value for initial bag-person association.
        """
        #print("run into SimpleTracke\n ")
        self.all_centers = {'garbage': OrderedDict(),
                            'person': OrderedDict()}  # Stores seen bounding box centers by object id
        self.prev_frame_ids = {'garbage': [],
                               'person': []}  # Stores ids of objects observed in the last frame

        self.bag_person_association = dict()  # Maps bag id to bag owner id
        self.bag_person_dist = dict()  # Maps bag id to distance to the bag owner
        self.instance_count = {'garbage': 0,
                               'person': 0}  # Counts how many garbage and person have been seen
        self.bag_person_thres = bag_person_thres
        self.self_association_thres = self_association_thres
        self.prev_frame_kept = {'garbage': False,
                                'person': False}  # Tracks if last frame's bounding boxes were kept or ignored
        self.keep_frame_counter = {'garbage': 0,
                                   'person': 0}  # Counts how many frames back object centers have been stored

    def update(self, boxes, labels):
        # type: (np.ndarray, np.ndarray) -> None
        """
        Updates the trackers state given bounding box and class detections.
        Args:
            boxes: numpy array containing two diagonal corner coordinates of the bounding boxes,
                shape [num_bounding_boxes, 4]
            labels: Array with bounding box labes, shape [num_centers]
        """
        #centers = compute_center(boxes)
        #print("\n進來update\n")
        centers = xyxy2xywh(boxes)
        #centers = converted_boxes[:, :2]  # 獲取中心點
        labels = np.array([self.map_label(label) for label in labels])  # 將標籤轉換為對應的名稱

        bag_bounding_centers, person_bounding_centers = split_bag_persons(centers, labels)

        self.frame2frame_association(person_bounding_centers, 'person')
        self.frame2frame_association(bag_bounding_centers, 'garbage')
        self.update_bag_person_association()
        # Initialize a dictionary to store unattended status
        self.unattended_status = {}

        # Process each garbage and determine if it's unattended
        for garbage_id in self.prev_frame_ids['garbage']:
            unattended = self.is_unattended(garbage_id)
            self.unattended_status[garbage_id] = unattended
        print(self.prev_frame_ids)
    def map_label(self, label):
        """
        Map YOLOv7 integer label to string label for tracking.
        """
        label_map = {6: 'person', 2: 'garbage'}  # 更新此映射以匹配您的標籤編號 @@
        return label_map.get(label, 'unknown')

    def is_unattended(self, garbage_id):
        # type: (int) -> bool
        """
        Checks if a given bag misses an owner or  has it's owner at a distance larger that the bag_person_thres
            threshold.
        Args:
            garbage_id:
        Returns:
            True if the bag does not have an owner or is too far away from its owner.
            False otherwise.
        """
        print("進來is_unattended\n")

        person_id = self.bag_person_association[garbage_id]
        print(person_id,"\n")
        if person_id is None:
            return True
        person_center = self.all_centers['person'][person_id]
        garbage_center = self.all_centers['garbage'][garbage_id]
        # print(person_center,"\n")
        # print(garbage_center,"\n")

        if np.sqrt(((person_center - garbage_center) ** 2).sum()) > self.bag_person_thres:
            return True

        return False

    def frame2frame_association(self, new_centers, tag):
        # type: (np.ndarray, str) -> None
        """
        Associates centers of 'person' and 'garbage' observed in the last frame with centers observed in the current
        frame.
        The association is done forward in time, i.e. we find the closest center in the new frame for each center
        observed in the previous frame.
        If two centers in the previous frame map to the same center in the new frame the closest center gets the
        association.
        In case some center in the new frame can't find a match in the old frame it is added as a new object with a
        new id.
        In case there were no observed objects in the previous frame or no observed objects in the current frame, the
        state is not updated.

        Args:
            new_centers: Array of bounding box centers detected in the current frame.
            tag: Either 'person' or 'garbage'.
        """
        print("frame2frame_association\n")
        frame_ids = []
        frame_centers = []
        new_frame_unused_centers = list(range(new_centers.shape[0]))

        # 新增部分：检查并扩展 new_centers 为二维数组
        if new_centers.ndim == 1:
            new_centers = np.expand_dims(new_centers, axis=0)

        if len(self.prev_frame_ids[tag]) > 0 and len(new_centers) > 0:
            prev_frame_centers = np.stack([self.all_centers[tag][id] for id in self.prev_frame_ids[tag]], axis=0)
            distances = cdist(prev_frame_centers, new_centers)

            cc_in_new_frame_index = distances.argmin(axis=1)
            new_frame_unused_centers = list(set(new_frame_unused_centers) - set(cc_in_new_frame_index.tolist()))

            min_dist = distances[range(len(self.prev_frame_ids[tag])), cc_in_new_frame_index]
            index_counter = Counter(cc_in_new_frame_index)

            for dist, prev_frame_id, new_center, index in zip(min_dist,
                                                            self.prev_frame_ids[tag],
                                                            new_centers[cc_in_new_frame_index],
                                                            cc_in_new_frame_index):

                if dist < self.self_association_thres and index_counter[index] <= 1:
                    # case where there is a unique closest center
                    self.all_centers[tag][prev_frame_id] = new_center
                    frame_ids.append(prev_frame_id)
                    frame_centers.append(new_center)
                elif dist > self.self_association_thres and index_counter[index] <= 1:
                    # case where the closest frame is too far away
                    self.all_centers[tag][self.instance_count[tag]] = new_center
                    frame_ids.append(self.instance_count[tag])
                    frame_centers.append(new_center)
                    self.instance_count[tag] += 1
                else:
                    # case where one new center is closest to several centers
                    other_dists = min_dist[cc_in_new_frame_index == index]
                    if dist <= other_dists.min():
                        self.all_centers[tag][prev_frame_id] = new_center
                        frame_ids.append(prev_frame_id)
                        frame_centers.append(new_center)

        # add the new centers which were not closest to any old center
        for new_center in new_centers[new_frame_unused_centers, :]:
            self.all_centers[tag][self.instance_count[tag]] = new_center
            frame_ids.append(self.instance_count[tag])
            frame_centers.append(new_center)
            self.instance_count[tag] += 1

        if frame_ids:
            self.prev_frame_ids[tag] = frame_ids
            self.prev_frame_kept[tag] = False
            self.keep_frame_counter[tag] = 0
        else:
            self.keep_frame_counter[tag] += 1
            if self.keep_frame_counter[tag] > 8:
                for id in self.prev_frame_ids[tag]:
                    self.all_centers[tag][id] = np.array([np.Inf, np.Inf, np.Inf])
                self.prev_frame_ids[tag] = []
                self.prev_frame_kept[tag] = False
            else:
                self.prev_frame_kept[tag] = True

        # print("1:",frame_ids, self.prev_frame_ids[tag],"\n")
        # print("2",self.all_centers[tag],"\n")

    def update_bag_person_association(self):
        # type: () -> None
        """
        Iterates over all detected garbage in the last frame (current frame) and updates the bag-person association and
        the bag person distance.
        """
        print("run into update_bag_person_association\n")
        for garbage_id in self.prev_frame_ids['garbage']:
            print("1:",garbage_id,"/n")
            if garbage_id not in self.bag_person_association or self.bag_person_association[garbage_id] is None:
                # Case were the bag has not previous owner
                person_id, dist = self.find_closest_person_to_bag(garbage_id)
                self.bag_person_association[garbage_id] = person_id
                self.bag_person_dist[garbage_id] = dist
                print(f"新包: garbage_id={garbage_id}, person_id={person_id}, dist={dist}\n")
            elif self.bag_person_association[garbage_id] not in self.prev_frame_ids['person']:
                # Case were the garbage owner as not observed in the current frame
                self.bag_person_dist[garbage_id] = float('inf')
                print(f"未找到擁有者: garbage_id={garbage_id}, dist=inf\n")
            else:
                # Case were both bag and owner were observed in the current frame
                bag_person_vector = (self.all_centers['person'][self.bag_person_association[garbage_id]] -
                                     self.all_centers['garbage'][garbage_id])
                self.bag_person_dist[garbage_id] = np.sqrt(np.power(bag_person_vector, 2).sum())
                print(f"已關聯: garbage_id={garbage_id}, person_id={self.bag_person_association[garbage_id]}, dist={self.bag_person_dist[garbage_id]}\n")
        print("bag_person_association:", self.bag_person_association)
        print("bag_person_dist:", self.bag_person_dist)
    def find_closest_person_to_bag(self, garbage_id):
        # type: (int) -> Tuple[Optional[int], float]
        """
        Checks for closest person in the current frame given an id of a detected bag.
        Returns the id of the person and the distance given that a person could be found with a distance below the
        bag_person_thres threshold.

        Args:
            garbage_id: Id of a bag observed in the current frame.
        Returns:
            person_id: Id of the closest person or None if no person could be found with a distance smaller than
                bag_person_thres
            distance: Distance in pixels between the person and the bag. Inf if not person could be found.
        """
        print("run into find_closest_person_to_bag\n")
        garbage_center = self.all_centers['garbage'][garbage_id]
        dists = []
        for person_id in self.prev_frame_ids['person']:
            person_center = self.all_centers['person'][person_id]
            dist = np.sqrt(np.power(person_center - garbage_center, 2).sum())
            dists.append(dist)
            print(f"包ID {garbage_id} 與人ID {person_id} 的距離: {dist}")
        if not self.prev_frame_ids['person']:
            return None, float('inf')
        closest_person_ind = int(np.array(dists).argmin())
        if dists[closest_person_ind] < self.bag_person_thres:
            return self.prev_frame_ids['person'][closest_person_ind], dists[closest_person_ind]
        else:
            return None, float('inf')
