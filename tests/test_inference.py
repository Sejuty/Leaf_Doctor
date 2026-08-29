import cv2
import numpy as np
import pytest

from leafdoctor.config import CLASS_NAMES, IMG_SIZE, MODEL_PATH, pretty_label
from leafdoctor.inference import InvalidImageError, decode_image, load_model, predict, preprocess


def encode(rgb):
    """Encode an RGB array to PNG bytes the way a browser upload would arrive."""
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()


def test_class_names_are_15_and_sorted():
    # flow_from_directory assigns indices alphabetically, so the persisted order
    # must be sorted or labels silently shift against the model's output index.
    assert len(CLASS_NAMES) == 15
    assert CLASS_NAMES == sorted(CLASS_NAMES)


def test_decode_preserves_channel_order():
    """The regression test for the BGR/RGB bug.

    A pure-red RGB image must come back pure red, not blue. The old app.py fed
    cv2's BGR output straight into a model trained on RGB.
    """
    red = np.zeros((10, 10, 3), dtype=np.uint8)
    red[:, :, 0] = 255  # red channel in RGB

    decoded = decode_image(encode(red))

    np.testing.assert_array_equal(decoded, red)
    assert decoded[0, 0].tolist() == [255, 0, 0]


def test_preprocess_shape_and_range():
    rgb = np.random.randint(0, 256, (300, 200, 3), dtype=np.uint8)
    batch = preprocess(rgb)

    assert batch.shape == (1, *IMG_SIZE, 3)
    assert batch.dtype == np.float32
    assert 0.0 <= batch.min() and batch.max() <= 1.0


def test_preprocess_normalisation_matches_training():
    white = np.full((50, 50, 3), 255, dtype=np.uint8)
    assert preprocess(white).max() == pytest.approx(1.0)


def test_decode_rejects_garbage():
    with pytest.raises(InvalidImageError):
        decode_image(b"this is not an image")


def test_grayscale_and_rgba_become_three_channels():
    gray = np.full((20, 20), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", gray)
    assert ok
    assert decode_image(buf.tobytes()).shape[2] == 3


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato — Yellow Leaf Curl Virus"),
        ("Pepper__bell___Bacterial_spot", "Bell Pepper — Bacterial spot"),
        ("Potato___healthy", "Potato — Healthy"),
        ("Tomato_healthy", "Tomato — Healthy"),
    ],
)
def test_pretty_label(raw, expected):
    assert pretty_label(raw) == expected


def test_every_class_name_renders():
    for name in CLASS_NAMES:
        assert " — " in pretty_label(name)


@pytest.mark.slow
def test_model_output_matches_label_count():
    model = load_model()
    assert model.output_shape[-1] == len(CLASS_NAMES)


@pytest.mark.slow
def test_predict_returns_sorted_probabilities():
    assert MODEL_PATH.exists()
    model = load_model()
    rgb = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    results = predict(model, rgb, top_k=3)

    assert len(results) == 3
    probs = [p for _, p in results]
    assert probs == sorted(probs, reverse=True)
    assert all(label in CLASS_NAMES for label, _ in results)
