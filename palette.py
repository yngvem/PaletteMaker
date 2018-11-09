from color_array import ColorArray
import matplotlib.colors as mpl_colors
import numpy as np


class ColorPalette:
    def __init__(self, reference_colors, name="pallete_maker_cmap"):
        """Initiates a color palette instance

        Parameters:
        -----------
        reference_colors : ColorArray
            Color array of length two that specifies the colors to interpolate
            between.
        """
        self.reference_colors = reference_colors
        self.name = name

    @property
    def reference_colors(self):
        return self._reference_colors

    @reference_colors.setter
    def reference_colors(self, reference_colors):
        if not isinstance(reference_colors, ColorArray):
            raise ValueError("`reference_colors` must be a ColorArray instance")
        if len(reference_colors.shape) != 2:
            raise ValueError(
                "`reference_colors` must be two-dimensional "
                "(one color dimension and one channel dimension)."
            )

        self._reference_colors = reference_colors

    def generate_cmap(self, n_interpolants=256):
        """Create a matplotlib colormap from the reference array

        Generates a color map which ranges from the first color in 
        ´reference_colors´ to the last. ´n_interpolants´ colors are 
        linearly interpolated in the color space of the reference colors.
        These colors are used to create a LinearSegmentedColormap (from 
        matplotlib).

        Parameters:
        -----------
        n_interpolants : int
            Number of colors sent to the matplotlib LinearSegmentedColormap 
            class.

        Returns:
        --------
        LinearSegmentedColormap 
        """
        colors = self._interpolate(n_interpolants).to_rgb()
        return mpl_colors.LinearSegmentedColormap.from_list(self.name, colors)

    def generate_swatches(self, n_swatches):
        return self._interpolate(n_swatches).to_rgb()

    def _interpolate(self, n_interpolants):
        """Generate `n_interpolants` colours from the reference colors.

        The specified colours are linearly interpolated.
        """

        if n_interpolants < 1:
            raise ValueError("Must be at least one interpolant")
        weights = np.tile(np.arange(n_interpolants) / (n_interpolants - 1), (3, 1)).T

        colors = (
            self.reference_colors[np.newaxis, 0] * np.flip(weights, 0)
            + self.reference_colors[np.newaxis, 1] * weights
        )

        return colors


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    reference_colors = ColorArray([[0.4, 0.26, 0.35], [0.53, 0.7, 0.87]]).to_hcl()
    name = "test_pallete"
    pallete = ColorPalette(reference_colors, name)

    # Some random data for line plots
    num_lines = 5
    x = np.linspace(-1, 1, 10)
    ys = [np.random.randn(10) for _ in range(num_lines)]

    # Some random data for heatmap
    y = np.linspace(-5, 5, 100)
    xx, yy = np.meshgrid(y, y)

    def r(x, y):
        return np.sqrt(x ** 2 + y ** 2)

    zz = np.sinc(r(xx, yy))

    # Plot the stuff
    fig, subs = plt.subplots(1, 2)
    colors = pallete.generate_swatches(num_lines)
    print(colors)
    for yi, color in zip(ys, colors):
        subs[0].plot(x, yi, color=color)

    colormap = pallete.generate_cmap()
    subs[1].imshow(zz, cmap=colormap)
    plt.show()
