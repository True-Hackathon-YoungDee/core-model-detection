from fall_detection.runner import PosePipeline

from conftest import make_person


def test_pose_pipeline_on_person_lost_is_called_when_tracker_forgets_person():
    lost_ids = []
    pipeline = PosePipeline(smoothing=True, on_person_lost=lost_ids.append)

    persons = pipeline.process([make_person(person_id=0)], t_seconds=0.0)
    assigned_id = persons[0].person_id

    for i in range(1, 20):
        pipeline.process([], t_seconds=float(i))

    assert assigned_id in lost_ids


def test_pose_pipeline_on_person_lost_is_optional():
    pipeline = PosePipeline(smoothing=True)  # no on_person_lost passed
    pipeline.process([make_person(person_id=0)], t_seconds=0.0)
    for i in range(1, 20):
        pipeline.process([], t_seconds=float(i))  # must not raise
