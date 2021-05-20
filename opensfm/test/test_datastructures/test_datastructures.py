import numpy as np
from opensfm import pymap


def test_shot_cameras():
    """Test that shot cameras are created and deleted correctly"""
    map_mgn = pymap.Map()
    cam_model = pymap.Camera(640, 480, "")
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
    cam_model = pymap.Camera(640, 480, "")
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
    cam1 = pymap.Camera(640, 480, "perspective")
    cam2 = pymap.Camera(640, 480, "perspective")
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


test_larger_problem()
test_landmarks()
test_shot_cameras()
test_pose()
test_shots()



"""
def test_create_shot_cameras(n_cams, map_mgn, cam_model):
  cams = []
  for i in range(0, n_cams):
    cam = map_mgn.create_shot_camera(i, cam_model, "cam"+str(i))
    print(cam)
    cams.append(cam)
  return cams

def test_delete_shot_cameras(cams, map_mgn):
  print("Deleting ", len(cams), " cameras") 
  for cam in cams:
    print("Delete cam: ", cam.camera_name_, " with id: ", cam.id_)
    map_mgn.remove_shot_camera(cam.id_)

  print("Number of cameras after delete: ", map_mgn.number_of_cameras())


def test_pose():
  pose = pymap.Pose()
  print("Default pose: \n cam_to_world \n{}, world_to_cam \n{}".\
    format(pose.get_cam_to_world(), pose.get_world_to_cam()))
  
  # TODO: Set with actual transformation matrix!
  pose.set_from_world_to_cam(np.array([[1,0,0,30],[0,1,0,20],[0,0,1,10],[0,0,0,1]]))
  print("Modify pose with set_from_world_to_cam: \n cam_to_world: \n{} \n world_to_cam: \n{}".\
    format(pose.get_cam_to_world(), pose.get_world_to_cam()))
  pose.set_from_cam_to_world(np.array([[1,0,0,30],[0,1,0,20],[0,0,1,10],[0,0,0,1]]))
  print("Modify pose with set_from_cam_to_world: \n cam_to_world: \n{} \n world_to_cam: \n{}".\
    format(pose.get_cam_to_world(), pose.get_world_to_cam()))

def test_create_shots(map_mgn, n_shots, cam):
  shots = []
  pose = pymap.Pose()
  for shot_id in range(n_shots):
    shots.append(map_mgn.create_shot(shot_id, cam, pose, "shot"+str(shot_id)))
    print("created: ", shots[-1].id_, ": ", shots[-1].name_, "obj: ", shots[-1])
  
  #-----------Try to create everything again ------------
  for shot_id in range(n_shots):
    shots.append(map_mgn.create_shot(shot_id, cam, pose, "shot"+str(shot_id)))
    print("created: ", shots[-1].id_, ": ", shots[-1].name_, "obj: ", shots[-1])
  return shots


def test_remove_shots(map_mgn):
  print("Number of Shots ", map_mgn.number_of_shots())
  shot_ids = [0, 5, 6]
  for shot_id in shot_ids:
      map_mgn.remove_shot(shot_id)
  print("Number of Shots {} after deleting {}".
        format(map_mgn.number_of_shots(), shot_ids))

  #------------DELETE the same multiple times--------------
  shot_ids = [1,1,1]
  for shot_id in shot_ids:
    map_mgn.remove_shot(shot_id)
  print("Number of Shots {} after deleting {}".
    format(map_mgn.number_of_shots(), shot_ids))

  #------------DELETE ALREADY DELETED--------------
  shot_ids = [5,6,8]  
  for shot_id in shot_ids:
    map_mgn.remove_shot(shot_id)
  print("Number of Shots {} after deleting {}".
    format(map_mgn.number_of_shots(), shot_ids))

  #------------DELETE ALL--------------
  all_shots = map_mgn.get_all_shots()
  for shot_id in all_shots.keys():
    map_mgn.remove_shot(shot_id)
  print("Number of Shots {} after deleting all".
    format(map_mgn.number_of_shots()))

  #------------DELETE ALL AGAIN--------------
  all_shots = map_mgn.get_all_shots()
  for shot_id in all_shots.keys():
    map_mgn.remove_shot(shot_id)
  print("Number of Shots {} after deleting {}".
    format(map_mgn.number_of_shots(), shot_ids))

def print_all_shots(map_mgn):
  all_shots = map_mgn.get_all_shots()
  print("Number of Shots ", len(all_shots))
  for shot in all_shots.values():
    print("Shots in map mgn: ", shot.id_)

def test_create_landmarks(map_mgn, n_lms):
  lms = []
  
  for lm_id in range(n_lms):
    lms.append(map_mgn.create_landmark(lm_id, np.random.rand(3,1), "lm"+str(lm_id)))
    lm = map_mgn.create_landmark(lm_id, np.random.rand(3,1), "lm"+str(lm_id))
    if (lm != lms[-1]):
      print("Double creation!")
      exit()
  print("Test create landmarks passed")

def test_remove_landmarks(map_mgn):
  lm_to_delete = np.array(np.random.rand(200,1)*1000, dtype=np.int)
  print("Landmarks before: ", map_mgn.number_of_landmarks(), len(np.unique(lm_to_delete)), np.max(lm_to_delete))
  for lm in lm_to_delete:
    map_mgn.remove_landmark(lm)
  print("Landmarks after: ", map_mgn.number_of_landmarks())

def test_larger_problem(map_mgn):
  # Create 2 cameras
  cam1 = pymap.Camera()
  cam2 = pymap.Camera()
  # Create 2 shot cameras
  shot_cam1 = map_mgn.create_shot_camera(0, cam1, "shot_cam1")
  shot_cam2 = map_mgn.create_shot_camera(1, cam2, "shot_cam2")
  # Create 10 shots, 5 with each shot camera
  shots = []
  for shot_id in range(0,5):
    shots.append(map_mgn.create_shot(shot_id, shot_cam1))
  for shot_id in range(5, 10):
    shots.append(map_mgn.create_shot(shot_id, 1))
    # shots[-1].init_keypts_and_descriptors(100)
  for shot in shots:
    shot.init_keypts_and_descriptors(100)
  # Create 200 landmarks
  landmarks = []
  for lm_id in range(0, 200):
    lm = map_mgn.create_landmark(lm_id,np.random.rand(3,1),"lm"+str(lm_id))
    print("Create lm: ", lm_id, lm, len(landmarks))
    landmarks.append(lm)
  
  # assign 100 to each shot (observations)
  for shot in shots: 
    lm_obs = np.asarray(np.random.rand(100)*1000%200, dtype=np.int)
    feat_obs = np.asarray(np.random.rand(100)*1000%100, dtype=np.int)
    # print("Before {} observations to shot {}".format(shot.compute_num_valid_pts(), shot.id_))
    for f_idx, lm_id in zip(feat_obs, lm_obs):
      # print(f_idx, lm_id, shot.number_of_keypoints())
      map_mgn.add_observation(shot, landmarks[lm_id],f_idx)
    print("Added {} observations to shot {}".format(shot.compute_num_valid_pts(), shot.id_))
    if (len(np.unique(feat_obs)) != shot.compute_num_valid_pts()):
      print("Error double add!" )
      return

  for lm in landmarks:
    print("lm: ", lm.id_, " has ", lm.number_of_observations())
    if (lm.has_observations()):
      shots = lm.get_observations()
      shot = list(shots.keys())[0]
      #TODO: print all the observations of shot
      map_mgn.remove_shot(shot.id_)
    print("lm: ", lm.id_, " has ", lm.number_of_observations(), "after delete!")

  # Remove 50 landmarks
  # lm_obs = np.unique(np.asarray(np.random.rand(50)*1000%200, dtype=np.int))
  # for lm_id in lm_obs:
    # map_mgn.remove_landmark(lm_id)
  map_mgn.remove_landmark(0)
  


  for shot in shots:
      print("{} observations in shot {}".format(shot.compute_num_valid_pts(), shot.id_))

  # Just try to delete it from all the shots
  # To test what happens if we delete it from a shot without the observation


  map_mgn.add_observation(shots[-1], landmarks[-1], 0)



  


# Map Manager
map_mgn = pymap.Manager()
cam_model = pymap.Camera()

# Test the cameras
cams = test_create_shot_cameras(10, map_mgn, cam_model)
print("Created {} cameras ".format( len(cams)))
print(cam_model)
test_delete_shot_cameras(cams, map_mgn)

# Test the pose
test_pose()

cam = map_mgn.create_shot_camera(0, cam_model, "cam"+str(0))

# Test the shots
shots = test_create_shots(map_mgn,10,cam)
test_remove_shots(map_mgn)

#Test the landmarks
landmarks = test_create_landmarks(map_mgn, 1000)
test_remove_landmarks(map_mgn)


#Now create a large problem
test_larger_problem(map_mgn)

"""
