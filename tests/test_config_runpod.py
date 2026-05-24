from core.config import EmbeddingsSettings, RunpodSettings


def test_runpod_settings_defaults():
    rp = RunpodSettings()
    assert rp.volume_mount_path == "/runpod-volume"
    assert rp.reconcile_path == ".runpod_fleet.json"
    assert rp.pod_video_root == "/runpod-volume/videos"
    assert rp.pod_model_path == "/runpod-volume/models/Qwen3-VL-Embedding-8B"
    assert rp.template_id == ""  # empty -> deploy from [runpod].image instead
    assert rp.gpu_type_id == ""  # empty -> auto-fetch GPUs in the volume's DC
    assert rp.gpu_max_price_hr == 0.80
    assert rp.gpu_min_vram_gb == 24
    assert rp.gpu_min_ram_gb == 30


def test_pod_idle_ttl_present_and_positive():
    emb = EmbeddingsSettings(
        exclude_disqualified_users=True,
        embed_max_length=1,
        adaptive_max_frames=1,
        adaptive_default_fps=1.0,
    )
    assert emb.pod_idle_ttl_s == 300
