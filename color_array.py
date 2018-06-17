__version__ = '0.1'
__author__ = 'Marie Roald & Yngve Moe'

import numpy as np


class ColorArray(np.dnarray):
    def __new__(cls, input_array, color_space='rgb'):
        obj = np.asarray(input_array).view(cls)
        obj.color_space = color_space
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.color_space = gettatr(obj, 'color_space', None)

	def to_srgb(self):
		"""Translate the colour representation to sRGB.
		"""
		if self.color_space == 'rgb':
			return self.copy()
		elif self.color_space == 'xyz':
			return self._xyz_to_rgb()
		else:
			return self.to_xyz().to_srgb(self)

	def to_lab(self):
		"""Translate the colour representation to CIELAB.
		"""
		if self.color_space == 'lab':
			return self.copy()
		elif self.color_space == 'xyz':
			return self._xyz_to_lab()
		else:
			return self.to_xyz().to_lab(self)

	def to_hcl(self):
		"""Translate the colour representation to HCL.
		"""
		if self.color_space == 'hcl':
			return self.copy()
		elif self.color_space == 'xyz':
			return self._xyz_to_hcl()
		else:
			return self.to_xyz().to_hcl(self)

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
	@staticmethod
	def _rgb_to_xyz(rgb):
        linear_rgb = self._linearize_srgb(rgb)
        transformation_matrix = np.array([[0.4124, 0.3576, 0.1805],
                                          [0.2126, 0.7152, 0.0722],
                                          [0.0193, 0.1192, 0.9505]])

        reshaped_rgb = linear_rgb.reshape(-1, 3)
        reshaped_xyz = reshaped_rgb @ transformation_matrix.T
        return reshaped_xyz.reshape(rgb.shape)

    @staticmethod
    def _linearize_srgb(c):
        mask = c > 0.04045
        c_linear = c/12.92
        c_linear[mask] = ((c[mask] + 0.055)/1.055)**2.4
        return c_linear

	@staticmethod
	def _lab_to_xyz(lab):
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

        return xyz

    @staticmethod
    def _lab_f_inverse(c):
        delta = 6/29
        mask = c>delta

        c_ = 3*delta**2*(c-4/29)
        c_[mask] = c[mask]**3
        return c_

	@staticmethod
	def _hcl_to_xyz(hcl):
		lab = self._hcl_to_lab(hcl)
		return lab.to_xyz()
	
	@staticmethod
	def _hcl_to_lab(hcl):
		lab = np.copy(hcl)
		lab[..., 0] = lab[..., 2]
		lab[..., 1], lab[..., 2] = self._pol_to_cart(hcl[..., 1], hcl[..., 0])
		return lab

	@staticmethod
	def _pol_to_cart(rho, phi):
		x = rho * np.cos(phi)
		y = rho * np.sin(phi)
		return(x, y)

	# ------------------- From XYZ ------------------- #
	@staticmethod
	def _xyz_to_rgb(xyz):
        transformation_matrix = np.array([[ 3.2406, -1.5372, -0.4986],
                                          [-0.9689,  1.8758,  0.0415],
                                          [ 0.0557, -0.2040,  1.0570]])
        reshaped_xyz = xyz.reshape(-1, 3)
        reshaped_rgb = reshaped_xyz @ transformation_matrix.T
        linear_rgb = reshaped_rgb.reshape(xyz.shape)
        rgb = self._gamma_correct_srgb(linear_rgb)
        return rgb

    @staticmethod
    def _gamma_correct_srgb(c):
        mask = c > 0.0031308
        c_gamma_corrected = 12.9292*c
        c_gamma_corrected[mask] = 1.055 * (c[mask]**(1/2.4)) - 0.055
        return c_gamma_corrected

	@staticmethod
	def _xyz_to_lab(c):
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

        return lab

    @staticmethod
    def _lab_f(c):
        delta = 6/29

        mask = c > delta**3
        c_ = c/(3 * delta**2) + 4/29
        c_[mask] = c**(1/3)
        return c_

	@staticmethod
	def _xyz_to_hcl(hcl):
		return self._lab_to_hcl(self.to_lab())

	@staticmethod
	def _lab_to_hcl(lab):
		hcl = np.copy(lab)
    	hcl[..., 2] = hcl[..., 0]
		hcl[..., 1], hcl[..., 0] = self._cart_to_pol(lab[..., 1], lab[..., 2])
		return hcl

	def _cart_to_pol(x, y):
		rho = np.sqrt(x**2 + y**2)
		phi = np.arctan2(y, x)
		return(rho, phi)
