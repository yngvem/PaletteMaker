import numpy as np
from pytest import fixture, raises
from color_array import ColorArray
from palette import ColorPalette


# ---------------- Fixtures ---------------#
@fixture
def random_2_3_array():
    return np.random.random((2, 3))


@fixture
def random_rgb_pair(random_2_3_array):
    return ColorArray(random_2_3_array, color_space="rgb")


@fixture
def random_lab_pair(random_2_3_array):
    return ColorArray(random_2_3_array, color_space="lab")


@fixture
def random_rgb_color_palette(random_rgb_pair):
    return ColorPalette(random_rgb_pair)


@fixture
def random_lab_color_palette(random_lab_pair):
    return ColorPalette(random_lab_pair)


# ----------------- Tests ------------------#
def test_interpolation_preserves_color_space(
    random_rgb_color_palette, random_lab_color_palette
):
    assert random_rgb_color_palette._interpolate(2).color_space == "rgb"
    assert random_rgb_color_palette._interpolate(5).color_space == "rgb"
    assert random_rgb_color_palette._interpolate(10).color_space == "rgb"

    assert random_lab_color_palette._interpolate(2).color_space == "lab"
    assert random_lab_color_palette._interpolate(5).color_space == "lab"
    assert random_lab_color_palette._interpolate(10).color_space == "lab"


def test_interpolation_preserves_endpoints(random_rgb_color_palette):
    endpoints = random_rgb_color_palette.reference_colors
    assert np.allclose(random_rgb_color_palette._interpolate(2)[0], endpoints[0])
    assert np.allclose(random_rgb_color_palette._interpolate(5)[0], endpoints[0])
    assert np.allclose(random_rgb_color_palette._interpolate(10)[0], endpoints[0])

    assert np.allclose(random_rgb_color_palette._interpolate(2)[-1], endpoints[-1])
    assert np.allclose(random_rgb_color_palette._interpolate(5)[-1], endpoints[-1])
    assert np.allclose(random_rgb_color_palette._interpolate(10)[-1], endpoints[-1])


def test_interpolation_fails_with_less_than_one_node(random_rgb_color_palette):
    with raises(ValueError):
        random_rgb_color_palette._interpolate(0)
    with raises(ValueError):
        random_rgb_color_palette._interpolate(-1)


def test_odd_interpolation_midpoint_is_mean(random_rgb_color_palette):
    mean_color = random_rgb_color_palette.reference_colors.mean(axis=0)
    assert np.allclose(random_rgb_color_palette._interpolate(3)[1], mean_color)
    assert np.allclose(random_rgb_color_palette._interpolate(5)[2], mean_color)


def test_reference_color_must_be_color_array(random_2_3_array):
    with raises(ValueError):
        ColorPalette(random_2_3_array)


def test_reference_color_must_be_rank_2():
    with raises(ValueError):
        ColorPalette(ColorArray(np.random.random((2, 2, 3))))
