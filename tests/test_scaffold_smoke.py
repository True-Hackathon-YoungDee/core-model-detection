from fall_detection.pose import PersonPose

from conftest import make_person, standing_pose


def test_standing_pose_returns_33_points_each_list():
    landmarks, world_landmarks = standing_pose()
    assert len(landmarks) == 33
    assert len(world_landmarks) == 33


def test_make_person_round_trips_through_person_from_landmarks():
    person = make_person(person_id=7)
    assert isinstance(person, PersonPose)
    assert person.person_id == 7
    assert len(person.landmarks) == 33
    assert len(person.world_landmarks) == 33
    assert person.score > 0.0
