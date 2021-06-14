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

import os
import sys
import numpy as np
import math
import argparse
import itertools
import fnmatch
from PIL import Image
from vispy import app, gloo, io, geometry

from blockcrafter import mcmodel
from blockcrafter import render

COLUMNS = 128

class BlockImages:
    def __init__(self):
        self.blocks = []

    def append(self, image):
        self.blocks.append(image)
        return len(self.blocks) - 1

    def export(self, columns):
        w, h = self.blocks[0].size
        rows = (len(self.blocks) + columns) // columns
        image = Image.new("RGBA", (columns * w, rows * h))
        for i, block in enumerate(self.blocks):
            assert block.size == (w, h)
            x = i % columns
            y = (i - x) // columns
            image.paste(block, (x * w, y * h))
        return image

class Canvas(app.Canvas):
    def __init__(self, args):
        super().__init__()

        self.draw_attempt = False
        self.show()
        self.update()

        self.args = args
        self.texture_sizes = args.texture_size
        if self.texture_sizes is None:
            self.texture_sizes = [16, 12]
        self.views = args.view
        if self.views is None:
            self.views = ["isometric", "topdown", "side"]
        self.rotations = args.rotation
        if self.rotations is None:
            self.rotations = [0, 1, 2, 3]

        self.assets = mcmodel.Assets.create(self.args.assets)

    def render_blocks(self, blockstates, texture_size, render_view, rotation, info_path, image_path):
        block_size = None
        if render_view == "isometric":
            block_size = texture_size * 2, texture_size * 2
        elif render_view == "topdown":
            block_size = texture_size, texture_size
        elif render_view == "side":
            w = texture_size
            h = math.sqrt(2) * texture_size
            # make h integer and even:
            h = int(math.floor(h / 2) * 2)
            block_size = (h, w)
        else:
            assert False, "Invalid view '%s'" % view

        model, view, projection = render.create_transform_ortho(aspect=1.0, view=render_view, fake_ortho=True)

        texture = gloo.Texture2D(shape=block_size + (4,))
        depth = gloo.RenderBuffer(shape=block_size)
        fbo = gloo.FrameBuffer(color=texture, depth=depth)
        fbo.activate()

        gloo.set_viewport(0, 0, block_size[1], block_size[0])
        gloo.set_clear_color((0.0, 0.0, 0.0, 0.0))

        render.set_blending("premultiplied")

        os.makedirs(self.args.output_dir, exist_ok=True)
        finfo = open(info_path, "w")
        print("%d %d %d" % (block_size[1], block_size[0], COLUMNS), file=finfo)

        def is_blockstate_included(name):
            patterns = self.args.blocks
            if patterns is None:
                return True
            for pattern in patterns:
                if fnmatch.fnmatch(name, pattern):
                    return True
            return False

        def write_block_info(blockstate, variant, indices):
            name = blockstate.prefix + ":" + blockstate.name
            properties = dict(blockstate.extra_properties)
            properties["color"] = str(indices[0])
            properties["uv"] = str(indices[1])
            print("%s %s %s" % (name, mcmodel.encode_variant(variant), mcmodel.encode_variant(properties)), file=finfo)

        images = BlockImages()
        for blockstate in blockstates:
            name = blockstate.prefix + ":" + blockstate.name
            if not is_blockstate_included(name):
                continue
            glblock = render.Block(blockstate)
            for index, variant in enumerate(blockstate.variants):
                modes = ["color", "uv"]
                indices = []
                for mode in modes:
                    if not self.args.no_render:
                        gloo.clear(color=True, depth=True)
                        actual_rotation = rotation
                        if name == "minecraft:full_water":
                            actual_rotation = 0
                        if blockstate.disable_blending or mode=="uv":
                            render.set_blending("opaque")
                        else:
                            render.set_blending("premultiplied")
                        if blockstate.disable_culling:
                            render.apply_face_culling(on=False)
                        else:
                            render.apply_face_culling(on=True)
                        actual_model = render.apply_model_rotation(model, rotation=0)
                        glblock.render(variant, actual_model, view, projection, rotation=actual_rotation, mode=mode)

                    array = np.array(fbo.read("color"))
                    if blockstate.disable_blending:
                        # make image opaque
                        if mode == "color":
                            array[:, :, 3] = (array[:, :, 3] > 0) * 255
                    image = Image.fromarray(array)
                    index = images.append(image)
                    indices.append(index)

                # waterlogged blocks need several variants written out
                if blockstate.waterloggable:
                    variant = dict(variant)
                    # 1. waterlogged version (will get water on top)
                    # (some blocks are always waterlogged, they don't have the waterlogged property)
                    if not blockstate.inherently_waterlogged:
                        variant["waterlogged"] = "true"
                    write_block_info(blockstate, variant, indices)

                    # 2. non-waterlogged version
                    # (only blocks that are not always waterlogged need this)
                    if not blockstate.inherently_waterlogged:
                        variant["waterlogged"] = "false"
                        write_block_info(blockstate, variant, indices)

                    # 3. blocks that are waterlogged in Minecraft but don't get the water on top in Mapcrafter
                    # (because there is water on top already, Mapcrafter needs this extra state internally)
                    variant["was_waterlogged"] = "true"
                    variant["waterlogged"] = "false"
                    write_block_info(blockstate, variant, indices)
                else:
                    # normal blocks just get their info written out
                    write_block_info(blockstate, variant, indices)

        if not self.args.no_render:
            images.export(columns=COLUMNS).save(image_path)

        finfo.close()
        fbo.deactivate()

    def on_draw(self, event):
        if self.draw_attempt:
            self.close()
            return
        self.draw_attempt = True

        path_template = "{view}_{rotation}_{texture_size}"
        blockstates = self.assets.blockstates
        for texture_size, view, rotation in itertools.product(self.texture_sizes, self.views, self.rotations):
            name = path_template.format(texture_size=texture_size, view=view, rotation=rotation)

            info_path = os.path.join(self.args.output_dir, name + ".txt")
            image_path = os.path.join(self.args.output_dir, name + ".png")
            print(image_path, flush=True)
            self.render_blocks(blockstates, texture_size, view, rotation, info_path, image_path)

        self.close()

def main():
    parser = argparse.ArgumentParser(description="Generate block images for Mapcrafter.")
    parser.add_argument("--osmesa", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--texture-size", "-t", type=int, action="append")
    parser.add_argument("--view", "-v", type=str, action="append")
    parser.add_argument("--rotation", "-r", type=int, action="append")
    parser.add_argument("--assets", "-a", type=str, action="append", required=True)
    parser.add_argument("--blocks", "-b", type=str, action="append")
    parser.add_argument("--output-dir", "-o", type=str, required=True)

    args = parser.parse_args()
    if args.osmesa:
        import vispy
        vispy.use("osmesa")
        assert vispy.app.use_app().backend_name == "osmesa"

    c = Canvas(args)
    app.run()
