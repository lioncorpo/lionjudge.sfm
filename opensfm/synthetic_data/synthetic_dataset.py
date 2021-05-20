import logging
import os
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from opensfm import tracking, features as oft, types, pysfm, pymap, pygeometry, io
from opensfm.dataset import DataSet


logger = logging.getLogger(__name__)


class SyntheticDataSet(DataSet):
    reconstruction: types.Reconstruction
    exifs: Dict[str, Any]
    features: Optional[Dict[str, oft.FeaturesData]]
    reference_lla: Dict[str, float]

    def __init__(
        self,
        reconstruction: types.Reconstruction,
        exifs: Dict[str, Any],
        features: Optional[Dict[str, oft.FeaturesData]] = None,
        tracks_manager: Optional[pysfm.TracksManager] = None,
        output_path: Optional[str] = None,
    ):
        data_path = "" if not output_path else output_path
        if data_path:
            io.mkdir_p(data_path)
            io.mkdir_p(os.path.join(data_path, "images"))
        super(SyntheticDataSet, self).__init__(data_path)
        self.reconstruction = reconstruction
        self.exifs = exifs
        self.features = features
        self.tracks_manager = tracks_manager
        self.image_list = list(reconstruction.shots.keys())
        self.reference_lla = {"latitude": 0.0, "longitude": 0.0, "altitude": 0.0}
        self.matches = None
        self.config["use_altitude_tag"] = True
        self.config["align_method"] = "naive"

    def images(self) -> List[str]:
        return self.image_list

    def load_camera_models(self) -> Dict[str, pygeometry.Camera]:
        return self.reconstruction.cameras

    def load_rig_cameras(self) -> Dict[str, pymap.RigCamera]:
        return self.reconstruction.rig_cameras

    def load_rig_assignments(self) -> List[List[Tuple[str, str]]]:
        rig_assignments = []
        for instance in self.reconstruction.rig_instances.values():
            rig_assignments.append([(k, v.id) for k, v in instance.rig_cameras.items()])
        return rig_assignments

    def load_exif(self, image: str) -> Dict[str, Any]:
        return self.exifs[image]

    def exif_exists(self, image: str) -> bool:
        return True

    def features_exist(self, image: str) -> bool:
        if self.features is None:
            return False
        feat = self.features
        if feat is None:
            return False
        return image in feat

    def load_words(self, image: str):
        n_closest = 50
        return [image] * n_closest

    def load_features(self, image: str) -> Optional[oft.FeaturesData]:
        if not self.features:
            return None
        feat = self.features
        if feat is None:
            return None
        return feat[image]

    def save_features(self, image: str, features_data: oft.FeaturesData):
        pass

    def matches_exists(self, image: str) -> bool:
        self._check_and_create_matches()
        if self.matches is None:
            return False
        return True

    def load_matches(self, image: str) -> Dict[str, np.ndarray]:
        self._check_and_create_matches()
        if self.matches is not None:
            return self.matches[image]
        else:
            return {}

    def _check_and_create_matches(self):
        if self.matches is None:
            self.matches = self._construct_matches()

    def _construct_matches(self):
        matches = {}
        for im1 in self.images():
            for im2 in self.images():
                if im1 == im2:
                    continue
                image_matches = matches.setdefault(im1, {})
                tracks = tracking.common_tracks(self.tracks_manager, im1, im2)[0]
                if len(tracks) > 10:
                    pair_matches = []
                    for t in tracks:
                        observations = self.tracks_manager.get_track_observations(t)
                        pair_matches.append(
                            np.array([observations[im1].id, observations[im2].id])
                        )
                    image_matches[im2] = np.array(pair_matches)
        return matches

    def load_tracks_manager(
        self, filename: Optional[str] = None
    ) -> pysfm.TracksManager:
        tracks_mgr = self.tracks_manager
        if not tracks_mgr:
            raise RuntimeError("No tracks manager for the synthetic dataset")
        return tracks_mgr

    def invent_reference_lla(
        self, images: Optional[List[str]] = None
    ) -> Dict[str, float]:
        return self.reference_lla

    def load_reference_lla(self) -> Dict[str, float]:
        return self.reference_lla

    def reference_lla_exists(self) -> bool:
        return True
