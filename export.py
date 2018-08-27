#!/usr/bin/env python

import os
import sys
import numpy as np
import math
import argparse
import itertools
from PIL import Image
from vispy import app, gloo, io, geometry

import mcmodel
import render

COLUMNS = 32

def load_blockstate_properties():
    properties = {}

    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "blockstates.properties")
    f = open(path, "r")
    for line in f.readlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ")
        assert len(parts) == 2, "Invalid line '%s'" % line

        name = parts[0]
        p = mcmodel.parse_variant(parts[1])
        properties[name] = p
    f.close()
    return properties

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
            image.paste(block, (x * h, y * h))
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
            self.texture_sizes = [16]
        self.views = args.view
        if self.views is None:
            self.views = ["isometric"]
        self.rotations = args.rotation
        if self.rotations is None:
            self.rotations = [0, 1, 2, 3]

        self.assets = mcmodel.Assets(args.assets)
        self.extra_properties = load_blockstate_properties()

    def render_blocks(self, blockstates, texture_size, render_view, rotation, info_path, image_path):
        block_size = None
        if render_view == "isometric":
            block_size = texture_size * 2
        elif render_view == "topdown":
            block_size = texture_size
        else:
            assert False, "Invalid view '%s'" % view

        model, view, projection = render.create_transform_ortho(aspect=1.0, view=render_view, fake_ortho=True)

        texture = gloo.Texture2D(shape=(block_size, block_size, 4))
        depth = gloo.RenderBuffer(shape=(block_size, block_size))
        fbo = gloo.FrameBuffer(color=texture, depth=depth)
        fbo.activate()

        gloo.set_viewport(0, 0, block_size, block_size)
        gloo.set_clear_color((0.0, 0.0, 0.0, 0.0))
        gloo.set_state(depth_test=True, blend=True)
        total_variants = sum([ len(b.variants) for b in blockstates ])

        os.makedirs(self.args.output_dir, exist_ok=True)
        finfo = open(info_path, "w")
        print("%d %d" % (total_variants, COLUMNS), file=finfo)

        images = BlockImages()
        for blockstate in blockstates:
            glblock = render.Block(blockstate)
            for index, variant in enumerate(blockstate.variants):
                modes = ["color", "uv"]
                indices = []
                for mode in modes:
                    gloo.clear(color=True, depth=True)
                    
                    if not self.args.no_render:
                        actual_model = render.apply_model_rotation(model, rotation=rotation)
                        glblock.render(variant, model, view, projection, mode=mode)

                    image = Image.fromarray(fbo.read("color"))
                    index = images.append(image)
                    indices.append(index)

                name = blockstate.prefix + ":" + blockstate.name
                properties = {"color" : str(indices[0]), "uv" : str(indices[1])}
                extra_properties = self.extra_properties.get(name, {})
                properties.update(extra_properties)
                print("%s %s %s" % (name, mcmodel.encode_variant(variant), mcmodel.encode_variant(properties)), file=finfo)

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
            print(image_path)
            self.render_blocks(blockstates, texture_size, view, rotation, info_path, image_path)

        self.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate block images for Mapcrafter.")
    parser.add_argument("--osmesa", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--texture-size", "-t", type=int, action="append")
    parser.add_argument("--view", "-v", type=str, action="append")
    parser.add_argument("--rotation", "-r", type=int, action="append")
    parser.add_argument("--assets", "-a", type=str, required=True)
    parser.add_argument("--output-dir", "-o", type=str, required=True)

    args = parser.parse_args()
    if args.osmesa:
        import vispy
        vispy.use("osmesa")
        assert vispy.app.use_app().backend_name == "osmesa"

    c = Canvas(args)
    app.run()
