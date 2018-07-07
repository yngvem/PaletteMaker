__version__ = '0.1'
__author__ = 'Marie Roald & Yngve Moe'

import numpy as np


class ColorArray(np.ndarray):
    def __new__(cls, input_array, color_space='rgb'):
        if input_array.shape[-1] != 3:
            raise ValueError('Last dimension of color arrays must be three')
        obj = np.asarray(input_array).view(cls)
        obj.color_space = color_space
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.color_space = getattr(obj, 'color_space', None)

    def to_rgb(self, illegal_rgb_behaviour=None):
        """Translate the colour representation to (s)RGB.
        """
        if self.color_space == 'rgb':
            return self.copy()
        elif self.color_space == 'xyz':
            rgb = self._xyz_to_rgb(self)

            if illegal_rgb_behaviour is None or illegal_rgb_behaviour.lower() == 'none': 
                return rgb
            elif illegal_rgb_behaviour.lower() == 'project':
                return np.clip(rgb, 0, 1)
            elif illegal_rgb_behaviour.lower() == 'nan':
                rgb[rgb<0] = np.nan
                rgb[rgb>1] = np.nan
                return rgb
            else:
                raise ValueError('Illegal rgb behaviour must be "none", "project" or "nan"')

        else:
            return self.to_xyz().to_rgb(illegal_rgb_behaviour)

    def to_lab(self):
        """Translate the colour representation to CIELAB.
        """
        if self.color_space == 'lab':
            return self.copy()
        elif self.color_space == 'xyz':
            return self._xyz_to_lab(self)
        else:
            return self.to_xyz().to_lab()

    def to_hcl(self):
        """Translate the colour representation to HCL.
        """
        if self.color_space == 'hcl':
            return self.copy()
        elif self.color_space == 'xyz':
            return self._xyz_to_hcl(self)
        else:
            return self.to_xyz().to_hcl()

    def to_xyz(self):
        """Translate the colour representation to XYZ.
        """
        if self.color_space == 'xyz':
            return self.copy()
        elif self.color_space == 'rgb':
            return self._rgb_to_xyz(self)
        elif self.color_space == 'lab':
            return self._lab_to_xyz(self)
        elif self.color_space == 'hcl':
            return self._hcl_to_xyz(self)
        elif self.color_space == 'yuv':
            return self._yuv_to_xyz(self)
        else:
            raise ValueError(f'{self.color_space} is not a known color space')

    # ------------------- To XYZ ------------------- #
    def _rgb_to_xyz(self, rgb):
        linear_rgb = self._linearize_rgb(rgb)
        transformation_matrix = np.array([[0.4124, 0.3576, 0.1805],
                                          [0.2126, 0.7152, 0.0722],
                                          [0.0193, 0.1192, 0.9505]])

        reshaped_rgb = linear_rgb.reshape(-1, 3)
        reshaped_xyz = reshaped_rgb @ transformation_matrix.T
        reshaped_xyz.color_space = 'xyz'
        return reshaped_xyz.reshape(rgb.shape)

    def _linearize_rgb(self, c):
        mask = c > 0.04045
        c_linear = c/12.92
        c_linear[mask] = ((c[mask] + 0.055)/1.055)**2.4
        return c_linear

    def _lab_to_xyz(self, lab):
        X_n = 95.047/100
        Y_n = 100/100
        Z_n = 108.883/100

        L = lab[..., 0]
        a = lab[..., 1]
        b = lab[..., 2]

        xyz = np.empty_like(lab)

        L_ = (L + 16)/116
        xyz[..., 0] = X_n*self._lab_f_inverse(L_ + a/500)
        xyz[..., 1] = Y_n*self._lab_f_inverse(L_)
        xyz[..., 2] = Z_n*self._lab_f_inverse(L_ - b/200)

        xyz.color_space = 'xyz'
        return xyz

    def _lab_f_inverse(self, c):
        delta = 6/29
        mask = c>delta

        c_ = 3*delta**2*(c-4/29)
        c_[mask] = c[mask]**3
        return c_

    def _hcl_to_xyz(self, hcl):
        lab = self._hcl_to_lab(hcl)
        return lab.to_xyz()
    
    def _hcl_to_lab(self, hcl):
        lab = np.copy(hcl)
        lab[..., 0] = lab[..., 2]
        lab[..., 1], lab[..., 2] = self._pol_to_cart(hcl[..., 1], hcl[..., 0])
        lab.color_space = 'lab'
        return lab

    def _pol_to_cart(self, rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)

    # ------------------- From XYZ ------------------- #
    def _xyz_to_rgb(self, xyz):
        transformation_matrix = np.linalg.inv(np.array([[0.4124, 0.3576, 0.1805],
                                                        [0.2126, 0.7152, 0.0722],
                                                        [0.0193, 0.1192, 0.9505]]))
        reshaped_xyz = xyz.reshape(-1, 3)
        reshaped_rgb = reshaped_xyz @ transformation_matrix.T
        linear_rgb = reshaped_rgb.reshape(xyz.shape)
        rgb = self._gamma_correct_rgb(linear_rgb)
        rgb.color_space = 'rgb'
        return rgb

    def _gamma_correct_rgb(self, c):
        mask = c > 0.0031308
        c_gamma_corrected = 12.92*c
        c_gamma_corrected[mask] = 1.055 * (c[mask]**(1/2.4)) - 0.055
        return c_gamma_corrected

    def _xyz_to_lab(self, xyz):
        X_n = 95.047/100
        Y_n = 100/100
        Z_n = 108.883/100

        x = xyz[..., 0]
        y = xyz[..., 1]
        z = xyz[..., 2]

        lab = np.empty_like(xyz)
        lab[..., 0] = 116*self._lab_f(y/Y_n) - 16
        lab[..., 1] = 500*(self._lab_f(x/X_n) - self._lab_f(y/Y_n))
        lab[..., 2] = 200*(self._lab_f(y/Y_n) - self._lab_f(z/Z_n))
        lab.color_space = 'lab'

        return lab

    def _lab_f(self, c):
        delta = 6/29

        mask = c > delta**3
        c_ = c/(3 * delta**2) + 4/29
        c_[mask] = c[mask]**(1/3)
        return c_

    def _xyz_to_hcl(self, hcl):
        return self._lab_to_hcl(self.to_lab())

    def _lab_to_hcl(self, lab):
        hcl = np.copy(lab)
        hcl[..., 2] = hcl[..., 0]
        hcl[..., 1], hcl[..., 0] = self._cart_to_pol(lab[..., 1], lab[..., 2])
        hcl.color_space = 'hcl'
        return hcl

    def _cart_to_pol(self, x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)
