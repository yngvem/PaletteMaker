__version__ = '0.1'
__author__ = 'Marie Roald & Yngve Moe'

import numpy as np
from pytest import fixture, approx
from color_array import ColorArray
import skimage.color as skcolor


np.random.seed(1)


#---------------- Fixtures ----------------#
@fixture
def random_array():
    return np.random.random((1, 1, 3))

@fixture
def random_rgb(random_array):
    return ColorArray(random_array, color_space='rgb')

@fixture
def random_lab(random_array):
    return ColorArray(random_array, color_space='lab')

@fixture
def random_xyz(random_array):
    return ColorArray(random_array, color_space='xyz')


#----------------- Tests ------------------#
def test_rgb_to_xyz_inverted_by_xyz_to_rgb(random_rgb):
    assert np.allclose(random_rgb, random_rgb.to_xyz().to_rgb())

def test_rgb_to_lab_inverted_by_lab_to_rgb(random_rgb):
    assert np.allclose(random_rgb, random_rgb.to_lab().to_rgb())

def test_rgb_to_rgb(random_rgb):
    assert np.allclose(random_rgb, random_rgb.to_rgb())

def test_xyz_to_rgb_inverted_by_rgb_to_xyz(random_xyz):
    assert np.allclose(random_xyz, random_xyz.to_rgb().to_xyz())

def test_xyz_to_lab_inverted_by_lab_to_xyz(random_xyz):
    assert np.allclose(random_xyz, random_xyz.to_lab().to_xyz())

def test_xyz_to_xyz_(random_xyz):
    assert np.allclose(random_xyz, random_xyz.to_xyz())

def test_lab_to_xyz_inverted_by_xyz_to_lab(random_lab):
    assert np.allclose(random_lab, random_lab.to_xyz().to_lab())

def test_lab_to_rgb_inverted_by_rgb_to_lab(random_lab):
    assert np.allclose(random_lab, random_lab.to_rgb().to_lab())

def test_lab_to_lab(random_lab):
    assert np.allclose(random_lab, random_lab.to_lab())








def test_rgb_to_lab_against_skimage(random_rgb):
    assert np.allclose(random_rgb.to_lab(), skcolor.rgb2lab(random_rgb), rtol=1e-2)
    
def test_lab_to_rgb_against_skimage(random_lab):
    assert np.allclose(random_lab.to_rgb('project'), skcolor.lab2rgb(random_lab), rtol=1e-2)

def test_xyz_to_rgb_against_skimage(random_xyz):
    assert np.allclose(random_xyz.to_rgb('project'), skcolor.xyz2rgb(random_xyz), rtol=1e-2)

def test_rgb_to_xyz_against_skimage(random_rgb):
    assert np.allclose(random_rgb.to_xyz(), skcolor.rgb2xyz(random_rgb), rtol=1e-2)

def test_lab_to_xyz_against_skimage(random_lab):
    assert np.allclose(random_lab.to_xyz(), skcolor.lab2xyz(random_lab), rtol=1e-2)

def test_xyz_to_lab_against_skimage(random_xyz):
    assert np.allclose(random_xyz.to_lab(), skcolor.xyz2lab(random_xyz), rtol=1e-2)


