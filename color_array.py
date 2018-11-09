__version__ = "0.1"
__author__ = "Marie Roald & Yngve Moe"

import numpy as np
from numba import jit, prange


class ColorArray(np.ndarray):
    def __new__(cls, input_array, color_space="rgb"):
        obj = np.asarray(input_array).view(cls)
        obj.color_space = color_space
        if obj.shape[-1] != 3:
            raise ValueError("Last dimension of color arrays must be three")
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.color_space = getattr(obj, "color_space", None)

    @staticmethod
    @jit(nopython=True, nogil=True)
    def _set_illegal_vals_nan(rgb):
        rgb_ = rgb.reshape(-1, 3)
        for i in range(rgb_.shape[0]):
            if np.any(np.logical_or(rgb_[i] < 0, rgb_[i] > 1)):
                rgb_[i, :] = np.nan

        return rgb_.reshape(rgb.shape)

    def to_rgb(self, illegal_rgb_behaviour=None):
        """Translate the colour representation to (s)RGB.
        """
        if self.color_space == "rgb":
            return self.copy()
        elif self.color_space == "xyz":
            rgb = self._xyz_to_rgb(self)

            if illegal_rgb_behaviour is None or illegal_rgb_behaviour.lower() == "none":
                return rgb
            elif illegal_rgb_behaviour.lower() == "project":
                return np.clip(rgb, 0, 1)
            elif illegal_rgb_behaviour.lower() == "nan":
                rgb = self._set_illegal_vals_nan(rgb)
                return rgb
            else:
                raise ValueError(
                    'Illegal rgb behaviour must be "none", "project" or "nan"'
                )

        else:
            return self.to_xyz().to_rgb(illegal_rgb_behaviour)

    def to_lab(self):
        """Translate the colour representation to CIELAB.
        """
        if self.color_space == "lab":
            return self.copy()
        elif self.color_space == "xyz":
            return self._xyz_to_lab(self)
        else:
            return self.to_xyz().to_lab()

    def to_hcl(self):
        """Translate the colour representation to HCL.
        """
        if self.color_space == "hcl":
            return self.copy()
        elif self.color_space == "xyz":
            return self._xyz_to_hcl(self)
        else:
            return self.to_xyz().to_hcl()

    def to_xyz(self):
        """Translate the colour representation to XYZ.
        """
        if self.color_space == "xyz":
            return self.copy()
        elif self.color_space == "rgb":
            return self._rgb_to_xyz(self)
        elif self.color_space == "lab":
            return self._lab_to_xyz(self)
        elif self.color_space == "hcl":
            return self._hcl_to_xyz(self)
        elif self.color_space == "yuv":
            return self._yuv_to_xyz(self)
        else:
            raise ValueError(f"{self.color_space} is not a known color space")

    # ------------------- To XYZ ------------------- #
    def _rgb_to_xyz(self, rgb):
        linear_rgb = self._linearize_rgb(rgb)
        transformation_matrix = np.array(
            [
                [0.4124, 0.3576, 0.1805],
                [0.2126, 0.7152, 0.0722],
                [0.0193, 0.1192, 0.9505],
            ]
        )

        reshaped_rgb = linear_rgb.reshape(-1, 3)
        reshaped_xyz = reshaped_rgb @ transformation_matrix.T
        reshaped_xyz.color_space = "xyz"
        return reshaped_xyz.reshape(rgb.shape)

    @staticmethod
    @jit(nopython=True, nogil=True, fastmath=True, parallel=True)
    def _linearize_rgb_fast(c):
        c_ = c.flatten()
        for i in prange(c_.shape[0]):
            if c_[i] > 0.04045:
                c_[i] = ((c_[i] + 0.055) / 1.055) ** 2.4
            else:
                c_[i] = c_[i] / 12.92

        return c_.reshape(c.shape)

    def _linearize_rgb(self, c):
        c_linearized = self._linearize_rgb_fast(c)
        return ColorArray(c_linearized, color_space=c.color_space)

    def _lab_to_xyz(self, lab):
        X_n = 95.047 / 100
        Y_n = 100 / 100
        Z_n = 108.883 / 100

        L = lab[..., 0]
        a = lab[..., 1]
        b = lab[..., 2]

        xyz = np.empty_like(lab)

        L_ = (L + 16) / 116
        xyz[..., 0] = X_n * self._lab_f_inverse(L_ + a / 500)
        xyz[..., 1] = Y_n * self._lab_f_inverse(L_)
        xyz[..., 2] = Z_n * self._lab_f_inverse(L_ - b / 200)

        xyz.color_space = "xyz"
        return xyz

    @staticmethod
    @jit(nopython=True, nogil=True, fastmath=True, parallel=True)
    def _lab_f_inverse(c):
        delta = 6 / 29

        c_ = c.flatten()
        for i in prange(c_.shape[0]):
            if c_[i] > delta:
                c_[i] = c_[i] ** 3
            else:
                c_[i] = 3 * delta ** 2 * (c_[i] - 4 / 29)

        return c_.reshape(c.shape)

    def _hcl_to_xyz(self, hcl):
        lab = self._hcl_to_lab(hcl)
        return lab.to_xyz()

    def _hcl_to_lab(self, hcl):
        lab = hcl.copy()
        lab[..., 0] = lab[..., 2]
        lab[..., 1], lab[..., 2] = self._pol_to_cart(hcl[..., 1], hcl[..., 0])
        lab.color_space = "lab"
        return lab

    def _pol_to_cart(self, rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)

    # ------------------- From XYZ ------------------- #
    def _xyz_to_rgb(self, xyz):
        transformation_matrix = np.linalg.inv(
            np.array(
                [
                    [0.4124, 0.3576, 0.1805],
                    [0.2126, 0.7152, 0.0722],
                    [0.0193, 0.1192, 0.9505],
                ]
            )
        )
        reshaped_xyz = xyz.reshape(-1, 3)
        reshaped_rgb = reshaped_xyz @ transformation_matrix.T
        linear_rgb = reshaped_rgb.reshape(xyz.shape)
        rgb = self._gamma_correct_rgb(linear_rgb)
        rgb.color_space = "rgb"
        return rgb

    @staticmethod
    @jit(nopython=True, nogil=True, fastmath=True, parallel=True)
    def _gamma_correct_rgb_fast(c):
        c_ = c.flatten()
        for i in prange(c_.shape[0]):
            if c_[i] > 0.003_130_8:
                c_[i] = 1.055 * (c_[i] ** (1 / 2.4)) - 0.055
            else:
                c_[i] = 12.92 * c_[i]

        return c_.reshape(c.shape)

    def _gamma_correct_rgb(self, c):
        c_gamma_corrected = self._gamma_correct_rgb_fast(c)
        return ColorArray(c_gamma_corrected, color_space=c.color_space)

    def _xyz_to_lab(self, xyz):
        X_n = 95.047 / 100
        Y_n = 100 / 100
        Z_n = 108.883 / 100

        x = xyz[..., 0]
        y = xyz[..., 1]
        z = xyz[..., 2]

        lab = np.empty_like(xyz)
        lab[..., 0] = 116 * self._lab_f(y / Y_n) - 16
        lab[..., 1] = 500 * (self._lab_f(x / X_n) - self._lab_f(y / Y_n))
        lab[..., 2] = 200 * (self._lab_f(y / Y_n) - self._lab_f(z / Z_n))
        lab.color_space = "lab"

        return lab

    @staticmethod
    @jit(nopython=True, nogil=True, fastmath=True, parallel=True)
    def _lab_f(c):
        delta = 6 / 29

        c_ = c.flatten()
        for i in prange(c_.shape[0]):
            if c_[i] > delta ** 3:
                c_[i] = c_[i] ** (1 / 3)
            else:
                c_[i] = c_[i] / (3 * delta ** 2) + 4 / 29

        return c_.reshape(c.shape)

    def _xyz_to_hcl(self, hcl):
        return self._lab_to_hcl(self.to_lab())

    def _lab_to_hcl(self, lab):
        hcl = lab.copy()
        hcl[..., 2] = hcl[..., 0]
        hcl[..., 1], hcl[..., 0] = self._cart_to_pol(lab[..., 1], lab[..., 2])
        hcl.color_space = "hcl"
        return hcl

    def _cart_to_pol(self, x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return (rho, phi)
