# Copyright 2018 Moritz Hilscher
#
# This file is part of Blockcrafter.
#
# Blockcrafter is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Blockcrafter is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Blockcrafter.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from PIL import Image

def extract_colormap_colors(colormap, flipped=False):
    if flipped:
        colormap = colormap.transpose(Image.ROTATE_180)

    data = np.array(colormap)
    assert data.dtype == np.uint8
    h, w, _ = data.shape
    return [
        data[254, 0],
        data[255, 253],
        data[3, 0]
    ]

def encode_colormap_colors(colors):
    def encode_color(color):
        assert len(color) == 4
        return "#%02.x%02.x%02.x%02.x" % tuple(reversed(color))
    return "|".join(map(encode_color, colors))

