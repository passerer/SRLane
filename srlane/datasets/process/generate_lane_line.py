import math

import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.lines import LineString, LineStringsOnImage
from scipy.interpolate import InterpolatedUnivariateSpline

from ..registry import PROCESS


@PROCESS.register_module
class GenerateLaneLine(object):
    def __init__(self, transforms=None, cfg=None, training=True):
        self.transforms = transforms
        self.img_w, self.img_h = cfg.img_w, cfg.img_h
        self.num_points = cfg.num_points
        self.n_offsets = cfg.num_points
        self.n_strips = cfg.num_points - 1
        self.strip_size = self.img_h / self.n_strips
        self.max_lanes = cfg.max_lanes
        self.feat_ds_strides = cfg.feat_ds_strides
        self.offsets_ys = np.arange(self.img_h, -1, -self.strip_size)
        self.training = training

        if transforms is None:
            raise NotImplementedError("transforms is None")

        if transforms is not None:
            img_transforms = []
            for aug in transforms:
                p = aug["p"]
                if aug["name"] != "OneOf":
                    img_transforms.append(
                        iaa.Sometimes(p=p,
                                      then_list=getattr(
                                          iaa,
                                          aug["name"])(**aug["parameters"])))
                else:
                    img_transforms.append(
                        iaa.Sometimes(
                            p=p,
                            then_list=iaa.OneOf([
                                getattr(iaa,
                                        aug_["name"])(**aug_["parameters"])
                                for aug_ in aug["transforms"]
                            ])))
        else:
            img_transforms = []
        self.transform = iaa.Sequential(img_transforms)

    def lane_to_linestrings(self, lanes):
        lines = []
        for lane in lanes:
            lines.append(LineString(lane))

        return lines

    @staticmethod
    def sample_lane(points, sample_ys):
        """Interpolates the x-coordinates of a sorted set of points
        based on the given sample_ys.

        Args:
            points: Sorted points representing a lane.
            sample_ys:  Y-coordinates.

        Returns:
            ndarray: X-coordinates.
        """
        # this function expects the points to be sorted
        points = np.array(points)
        if not np.all(points[1:, 1] < points[:-1, 1]):
            raise ValueError("Annotaion points have to be sorted")
        x, y = points[:, 0], points[:, 1]

        # interpolate points inside domain
        assert len(points) > 1
        interp = InterpolatedUnivariateSpline(y[::-1],
                                              x[::-1],
                                              k=min(3, len(points) - 1))
        domain_min_y = y.min()
        domain_max_y = y.max()
        sample_ys_inside_domain = sample_ys[(sample_ys >= domain_min_y)
                                            & (sample_ys <= domain_max_y)]
        assert len(sample_ys_inside_domain) > 0
        interp_xs = interp(
            sample_ys_inside_domain)  # Since it is interpolation, the interp_xs are guaranteed to be within the range of the image. # noqa: E501

        # extrapolate lane to the bottom of the image with a straight line
        # using the 2 points closest to the bottom
        two_closest_points = points[:2]
        extrap = np.polyfit(two_closest_points[:, 1],
                            two_closest_points[:, 0],
                            deg=1)
        extrap_ys = sample_ys[sample_ys > domain_max_y]
        extrap_xs = np.polyval(extrap, extrap_ys)  # It is possible to exceed the range. # noqa: E501
        all_xs = np.hstack((extrap_xs, interp_xs))

        return all_xs

    @staticmethod
    def filter_duplicate_points(points):
        """Filters out duplicate points from a given list of points.

        Args:
            points: Sorted points representing a lane.

        Returns:
            List: Filtered points.
        """
        if points[-1][1] > points[0][1]:
            raise ValueError("Annotaion points have to be sorted")
        filtered_points = []
        used = set()
        for p in points:
            if p[1] not in used:
                filtered_points.append(p)
                used.add(p[1])

        return filtered_points

    @staticmethod
    def check_horizontal_lane(points, angle_threshold=5):
        """Check whether a lane is nearly horizontal.

        Args:
            points: Sorted points representing a lane.
            angle_threshold: angle threshold.

        Returns:
            bool: True if the lane angle is greater than the threshold,
             indicating a non-horizontal lane. False otherwise.
        """
        if len(points) < 2:
            return False
        rad = math.atan(
            math.fabs((points[-1][1] - points[0][1]) /
                      (points[0][0] - points[-1][0] + 1e-6)))
        angle = math.degrees(rad)

        return angle > angle_threshold

    def generate_angle_map(self, lanes):
        """Genrate ground-truth angle map for multi resolution features.

        Args:
            lanes: Annotatedd lanes.

        Returns:
            List: Angle maps.
        """
        gt_angle_list = []
        gt_seg_list = []
        for stride in self.feat_ds_strides:
            offsets_ys = np.arange(self.img_h, -1, -stride)
            gt_angle = np.zeros((self.img_h // stride, self.img_w // stride))
            gt_seg = np.zeros((self.img_h // stride, self.img_w // stride))
            for lane_idx, lane in enumerate(lanes, 1):
                try:
                    all_xs = self.sample_lane(
                        lane, offsets_ys)
                except AssertionError:
                    continue
                all_xs = all_xs / stride
                offsets_ys_down = offsets_ys / stride
                for i, (x, y) in enumerate(zip(all_xs[1:],
                                               offsets_ys_down[1:]), 1):
                    int_x, int_y = int(x), int(y)
                    if (int_x < 0 or int_x >= gt_angle.shape[1] or
                            int_y < 0 or int_y >= gt_angle.shape[0]):
                        continue
                    theta = math.atan(1 / (x - all_xs[i - 1] + 1e-6)) / math.pi
                    theta = theta if theta > 0 else 1 - abs(theta)
                    gt_angle[int_y][int_x] = theta
                    gt_seg[int_y][int_x] = 1  # lane_idx
            gt_angle_list.append(gt_angle)
            gt_seg_list.append(gt_seg)
        return gt_angle_list, gt_seg_list

    def transform_annotation(self, old_lanes):
        """Transforms the annotations.

        Args:
            old_lanes: Multi lanes represented by points.

        Returns:
            dict:
            - "gt_lane": Filtered, aligned, ground-truth lanes.
            - "gt_angle": Ground-truth lane angle map.
        """
        img_w, img_h = self.img_w, self.img_h
        # removing lanes with less than 2 points
        old_lanes = filter(lambda x: len(x) > 1, old_lanes)  # The y-coordinate increases from the top of the image downwards. # noqa: E501
        # sort lane points by Y (bottom to top of the image)
        old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]
        # remove points with same Y (keep first occurrence)
        old_lanes = [self.filter_duplicate_points(lane) for lane in old_lanes]
        old_lanes = list(filter(self.check_horizontal_lane, old_lanes))
        # normalize the annotation coordinates
        old_lanes = [[[
            x * self.img_w / float(img_w), y * self.img_h / float(img_h)
        ] for x, y in lane] for lane in old_lanes]

        angle_map_list, seg_map_list = self.generate_angle_map(old_lanes)

        lanes = np.ones(
            (self.max_lanes, 2 + 2 + self.n_offsets), dtype=np.float32
        ) * -1e5  # 2 scores, 1 start_y, 1 length, n_offsets coordinates
        # lanes are invalid by default
        lanes[:, 0] = 1
        lanes[:, 1] = 0
        for lane_idx, lane in enumerate(old_lanes):
            if lane_idx >= self.max_lanes:
                break
            try:
                all_xs = self.sample_lane(
                    lane, self.offsets_ys)
            except AssertionError:
                continue
            # separate between inside and outside points
            inside_mask = (all_xs >= 0) & (all_xs < self.img_w)
            xs_inside_image = all_xs[inside_mask]
            xs_outside_image = all_xs[~inside_mask]
            if len(xs_inside_image) <= 1:
                continue
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            lanes[lane_idx, 2] = len(xs_outside_image)

            lanes[lane_idx, 3] = len(xs_inside_image)
            lanes[lane_idx, 4:4 + len(all_xs)] = all_xs

        new_anno = {
            "gt_lane": lanes,
            "gt_angle": angle_map_list,
            "gt_seg": seg_map_list,
        }
        return new_anno

    def linestrings_to_lanes(self, lines):
        lanes = []
        for line in lines:
            lanes.append(line.coords)

        return lanes

    def __call__(self, sample):
        """Applies the lane transformation to a sample.

        Args:
            sample: The input sample containing "img" and "lanes" information.

        Returns:
            dict:
            - "img": Normalized image.
            - "lanes": Original lanes.
            - "gt_lane": Transformed, Filtered, aligned, ground-truth lanes.
            - "gt_angle": Ground-truth lane angle map.
        """
        img_org = sample["img"]
        line_strings_org = self.lane_to_linestrings(sample["lanes"])
        line_strings_org = LineStringsOnImage(line_strings_org,
                                              shape=img_org.shape)

        for i in range(10):
            if self.training:
                img, line_strings = self.transform(
                    image=img_org.copy().astype(np.uint8),
                    line_strings=line_strings_org)
            else:
                img, line_strings = self.transform(
                    image=img_org.copy().astype(np.uint8),
                    line_strings=line_strings_org)
            line_strings.clip_out_of_image_()
            try:
                annos = self.transform_annotation(
                    self.linestrings_to_lanes(line_strings))
                break
            except Exception as e:
                if (i + 1) == 10:
                    raise Exception(e)

        sample["img"] = img.astype(np.float32) / 255.
        sample.update(annos)

        return sample
