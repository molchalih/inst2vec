from core.config import LabelsSettings


def test_labels_settings_has_clip_engine_knobs():
    s = LabelsSettings()
    assert s.clip_gpu_memory_utilization == 0.93
    assert s.clip_max_model_len == 6144
    assert s.clip_enforce_eager is True
