class ColorPalette
    def __init__(self, reference_colors):
        """Initiates a color palette instance

        Parameters:
        -----------
        reference_colors : ColorArray
            Color array of length two that specifies the colors to interpolate
            between.
        """
        self.reference_colors = reference_colors

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
        pass

