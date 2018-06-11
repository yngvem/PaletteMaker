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
		
	def to_yuv(self):
		"""Translate the colour representation to YUV.
		"""
		if self.color_space == 'yuv':
			return self.copy()
		elif self.color_space == 'xyz':
			return self._xyz_to_yuv(self)
		else:
			return self.to_xyz().to_yuv()
	
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
	def _rgb_to_xyz(c):
		pass
	
	@staticmethod
	def _lab_to_xyz(c):
		pass

	@staticmethod
	def _hcl_to_xyz(c):
		pass
	
	@staticmethod
	def _yuv_to_xyz(c):
		pass

	# ------------------- From XYZ ------------------- #
	@staticmethod
	def _xyz_to_rgb(c):
		pass
	
	@staticmethod
	def _xyz_to_lab(c):
		pass

	@staticmethod
	def _xyz_to_hcl(c):
		pass
	
	@staticmethod
	def _xyz_to_yuv(c):
		pass