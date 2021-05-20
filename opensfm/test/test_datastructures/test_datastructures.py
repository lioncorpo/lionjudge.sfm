import numpy as np
from opensfm import pymap
from opensfm import types
from opensfm import reconstruction
from opensfm import pysfm

def test_shot_cameras():
    """Test that shot cameras are created and deleted correctly"""
    map_mgn = pymap.Map()
    cam_model = pymap.CameraModel(640, 480, "")
    n_cams = 10
    cams = []
    for i in range(0, n_cams):
        cam = map_mgn.create_shot_camera(i, cam_model, "cam" + str(i))
        cams.append(cam)
    assert len(cams) == map_mgn.number_of_cameras()

    # Test double create
    for i in range(0, n_cams):
        cam = map_mgn.create_shot_camera(i, cam_model, "cam" + str(i))
        assert cams[i] == cam

    # Delete 2 out of the ten cameras
    map_mgn.remove_shot_camera(0)
    assert n_cams - 1 == map_mgn.number_of_cameras()
    map_mgn.remove_shot_camera(3)
    assert n_cams - 2 == map_mgn.number_of_cameras()

    # Double deletion
    map_mgn.remove_shot_camera(3)
    assert n_cams - 2 == map_mgn.number_of_cameras()
    shots = map_mgn.get_all_cameras()
    shot_idx = sorted([shot_id for shot_id in shots.keys()])
    remaining_shots = [1, 2, 4, 5, 6, 7, 8, 9]
    # shot_idx.sort()
    assert shot_idx == remaining_shots
    # remove the rest
    for shot_id in remaining_shots:
        map_mgn.remove_shot_camera(shot_id)
    assert map_mgn.number_of_shots() == 0


def test_pose():
    pose = pymap.Pose()
    # Test default
    assert np.allclose(pose.get_cam_to_world(), np.eye(4), 1e-10)
    assert np.allclose(pose.get_world_to_cam(), np.eye(4), 1e-10)

    # Test with other matrix
    T = np.array([[1, 2, 2, 30],
                  [3, 1, 3, 20],
                  [4, 3, 1, 10],
                  [8, 7, 5, 1]], dtype=np.float)

    # TODO: Set with actual transformation matrix!
    pose.set_from_world_to_cam(T)
    assert np.allclose(pose.get_world_to_cam(), T)
    assert np.allclose(pose.get_cam_to_world(), np.linalg.inv(T))

    pose.set_from_cam_to_world(T)
    assert np.allclose(pose.get_cam_to_world(), T)
    assert np.allclose(pose.get_world_to_cam(), np.linalg.inv(T)) 


def test_shots():
    shots = []
    map_mgn = pymap.Map()
    cam_model = pymap.CameraModel(640, 480, "")
    cam = map_mgn.create_shot_camera(0, cam_model, "cam" + str(0))
    pose = pymap.Pose()
    n_shots = 10
    for shot_id in range(n_shots):
        # Test next_id function
        assert shot_id == map_mgn.next_unique_shot_id()
        #Create with cam object
        if shot_id < 5:
            shot = map_mgn.create_shot(shot_id, cam, "shot" + str(shot_id), pose)
        else: #and id
            shot = map_mgn.create_shot(shot_id, 0, "shot" + str(shot_id), pose)
        shots.append(shot)

    # Create again and test if the already created pointer is returned
    for shot_id in range(n_shots):
        shot = map_mgn.create_shot(shot_id, cam, "shot" + str(shot_id), pose)
        assert shot == shots[shot_id]
    assert map_mgn.number_of_shots() == n_shots
    # Test default parameters
    shots.append(map_mgn.create_shot(n_shots, cam, "shot" + str(shot_id)))
    shots.append(map_mgn.create_shot(n_shots + 1, cam))
    assert map_mgn.number_of_shots() == n_shots + 2
    delete_ids = [0, 1, 2, 3, 7, 7, 7, 7]
    remaining_ids = [4, 5, 6, 8, 9, 10, 11]
    n_deleted = len(np.unique(delete_ids))
    for shot_id in delete_ids:
        map_mgn.remove_shot(shot_id)

    assert map_mgn.number_of_shots() == n_shots + 2 - n_deleted
    shot_ids = sorted([shot_id for shot_id in map_mgn.get_all_shots()])
    assert remaining_ids == shot_ids

    # Try to create only the deleted ones again
    for shot_id in range(n_shots):
        shot = map_mgn.create_shot(shot_id, cam, "shot" + str(shot_id), pose)
    assert map_mgn.number_of_shots() == n_shots + 2
    shot_ids = sorted([shot_id for shot_id in map_mgn.get_all_shots()])
    all_ids = [idx for idx in range(n_shots + 2)]
    assert shot_ids == all_ids

    # Delete all shots
    shot_ids = sorted([shot_id for shot_id in map_mgn.get_all_shots()])
    for shot_id in shot_ids:
        map_mgn.remove_shot(shot_id)
    assert map_mgn.number_of_shots() == 0
    assert map_mgn.next_unique_shot_id() == n_shots + 2
    map_mgn.create_shot(0, cam, "shot" + str(shot_id), pose)
    assert map_mgn.next_unique_shot_id() == n_shots + 2
    map_mgn.create_shot(1, cam, "shot" + str(shot_id), pose)
    assert map_mgn.next_unique_shot_id() == n_shots + 2
    map_mgn.create_shot(1000, cam, "shot" + str(shot_id), pose)
    assert map_mgn.next_unique_shot_id() == 1001


def test_landmarks():
    map_mgn = pymap.Map()
    n_lms = 100
    lms = []
    for lm_id in range(n_lms):
        pos = np.random.rand(3)
        assert map_mgn.next_unique_landmark_id() == lm_id
        lm = map_mgn.create_landmark(lm_id, pos, "lm" + str(lm_id))
        lms.append(lm)
        assert np.allclose(pos.flatten(), lm.get_global_pos())
    assert n_lms == map_mgn.number_of_landmarks()
    # Test double creation
    for lm_id in range(n_lms):
        pos = np.random.rand(3)
        lm = map_mgn.create_landmark(lm_id, pos, "lm" + str(lm_id))
        assert lm == lms[lm_id]
    assert n_lms == map_mgn.number_of_landmarks()
    # Test pos update
    for lm_id in range(n_lms):
        pos = np.random.rand(3)
        lm = lms[lm_id]
        lm.set_global_pos(pos)
        assert np.allclose(lm.get_global_pos(), pos)

    # Remove every second landmark
    remaining_ids = [idx for idx in range(1, 100, 2)]
    for lm_id in range(0, 100, 2):
        map_mgn.remove_landmark(lm_id)
    lm_ids = sorted([lm_id for lm_id in map_mgn.get_all_landmarks()])
    # check if remaining lms in the map are equal to the expected
    assert len(lm_ids) == len(remaining_ids)
    assert lm_ids == remaining_ids
    assert n_lms - len(remaining_ids) == map_mgn.number_of_landmarks()

    map_mgn.create_landmark(1000, pos, "lm" + str(lm_id))
    assert map_mgn.next_unique_landmark_id() == 1001
    assert len(remaining_ids) + 1 == map_mgn.number_of_landmarks()


def test_larger_problem():
    map_mgn = pymap.Map()
    # Create 2 cameras
    cam1 = pymap.CameraModel(640, 480, "perspective")
    cam2 = pymap.CameraModel(640, 480, "perspective")
    # Create 2 shot cameras
    shot_cam1 = map_mgn.create_shot_camera(0, cam1, "shot_cam1")
    shot_cam2 = map_mgn.create_shot_camera(1, cam2, "shot_cam2")
    # Create 10 shots, 5 with each shot camera
    shots = []
    for shot_id in range(0,5):
        shots.append(map_mgn.create_shot(shot_id, shot_cam1))
    for shot_id in range(5, 10):
        shots.append(map_mgn.create_shot(shot_id, shot_cam2))
    assert len(shots) == map_mgn.number_of_shots()

    n_kpts = 100
    # init shot with keypts and desc
    for shot in shots:
        assert shot.number_of_keypoints() == 0
        shot.init_keypts_and_descriptors(n_kpts)
        assert shot.number_of_keypoints() == n_kpts

    n_lms = 200
    # Create 200 landmarks
    landmarks = []
    for lm_id in range(0, n_lms):
        pos = np.random.rand(3)
        lm = map_mgn.create_landmark(lm_id, pos, "lm" + str(lm_id))
        landmarks.append(lm)
    assert map_mgn.number_of_landmarks() == n_lms

    # assign 100 to each shot (observations)
    for shot in shots:
        lm_obs = np.asarray(np.random.rand(n_kpts) * 1000 % n_lms,
                            dtype=np.int)
        feat_obs = np.asarray(np.random.rand(n_kpts) * 1000 % n_kpts,
                              dtype=np.int)
        # Add observations between a shot and a landmark
        # with duplicates!
        for f_idx, lm_id in zip(feat_obs, lm_obs):
            lm = landmarks[lm_id]
            map_mgn.add_observation(shot, lm, f_idx)
            assert lm.is_observed_in_shot(shot)
        assert len(np.unique(feat_obs)) == shot.compute_num_valid_pts(1)


def test_map_wrapper():
    rec = types.Reconstruction()
    # reconstruction.reference = data.load_reference()
    # reconstruction.cameras = copy.deepcopy(camera_priors)

    camera = types.PerspectiveCamera()
    camera.id = "cam1"
    camera.width = 640
    camera.height = 480
    camera.focal = 1
    camera.k1 = 0
    camera.k2 = 0
    metadata = types.ShotMetadata()
    metadata.gps_dop = 0.0
    metadata.gps_position = [0.0, 0.0, 0.0]
    camera2 = types.PerspectiveCamera()
    camera2.id = "cam2"
    camera2.width = 640
    camera2.height = 480
    camera2.focal = 1
    camera2.k1 = 0
    camera2.k2 = 0
    cameras = {}
    cameras[camera.id] = camera
    cameras[camera2.id] = camera2
    rec.cameras = cameras
    assert(len(rec.cameras) == 2)
    shot1 = types.Shot()
    shot1.id = "im1"
    shot1.camera = "cam1" 
    shot1.pose = types.Pose()
    shot1.metadata = metadata
    rec.add_shot(shot1)

    shot2 = types.Shot()
    shot2.id = "im2"
    shot2.camera = "cam2"
    shot2.pose = types.Pose()
    shot2.metadata = metadata
    rec.add_shot(shot2)
    assert(len(rec.shots) == 2)

    for shot in rec.shots:
        print("shot:", shot.id, shot.name)
    for shot in rec.shots.values():
        print("shot val:", shot.id, shot.name)
    p = types.Point()
    p.id = 0
    p.coordinates = np.array([0, 1, 2])
    rec.add_point(p)
    p.id = 1
    p.coordinates = np.array([0, 1, 3])
    rec.add_point(p)
    p.id = 2
    p.coordinates = np.array([2, 1, 3])
    rec.add_point(p)

    assert(len(rec.points) == 3)
    assert(rec.map.number_of_landmarks() == 3)

    my_point = rec.get_point(2)
    assert(my_point.id == 2)
    shot_1 = rec.get_shot("im1")
    shot_2 = rec.get_shot("im2")
    point = rec.get_point(0)
    point2 = rec.get_point(1)
    observation_1 = pysfm.Observation(1,2,3,255,255,0,100)
    observation_2 = pysfm.Observation(1,2,3,255,255,0,51)
    rec.map.add_observation(shot_1, point, observation_1)
    rec.map.add_observation(shot_2, point, observation_2)
    observation_3 = pysfm.Observation(1,2,3,255,255,0,3110)
    observation_4 = pysfm.Observation(1,2,3,255,255,0,510)
    rec.map.add_observation(shot_1, point2, observation_3)
    rec.map.add_observation(shot_2, point2, observation_4)
    print(len(rec.points))
    rec.points = {}
    print(len(rec.points))
    print(point.id)
    print(point.get_observations())
    p = types.Point()
    p.id = 0
    p.coordinates = np.array([0, 1, 2])
    rec.add_point(p)
    p.id = 1
    p.coordinates = np.array([0, 1, 3])
    rec.add_point(p)
    p.id = 2
    p.coordinates = np.array([2, 1, 3])
    rec.add_point(p)
    print(len(rec.points))

    

test_map_wrapper()
# test_larger_problem()
# test_landmarks()
# test_shot_cameras()
# test_pose()
# test_shots()

# new unit tests
# test name/id retrieval
# test get shot
# test clear

