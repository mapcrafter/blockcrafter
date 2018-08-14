#!/usr/bin/env python

import os
import sys
import numpy as np
import math
from PIL import Image
from vispy import app, gloo, io, geometry

import mcmodel
import render

assetdir = "assets"
#assetdir = "/home/moritz/.minecraft/versions/1.13/1.13.jar"
w, h = 32, 32
columns = 32
rotation = 0
outdir = "out"

class BlockImages:
    def __init__(self):
        self.blocks = []

    def append(self, image):
        self.blocks.append(image)
        return len(self.blocks) - 1

    def export(self, columns=32):
        rows = (len(self.blocks) + columns) // 16
        image = Image.new("RGBA", (columns * w, rows * h))
        for i, block in enumerate(self.blocks):
            x = i % columns
            y = (i - x) // columns
            image.paste(block, (x * h, y * h))
        return image

class Canvas(app.Canvas):
    def __init__(self):
        super().__init__()

        self.show()
        self.update()

        self.draw_attempt = False

    def on_draw(self, event):
        if self.draw_attempt:
            self.close()
        self.draw_attempt = True

        model, view, projection = render.create_transform_ortho(aspect=1.0, fake_ortho=True)

        texture = gloo.Texture2D(shape=(w, h, 4))
        depth = gloo.RenderBuffer(shape=(w, h))
        fbo = gloo.FrameBuffer(color=texture, depth=depth)
        fbo.activate()

        gloo.set_viewport(0, 0, w, h)
        gloo.set_clear_color((0.0, 0.0, 0.0, 0.0))
        gloo.set_state(depth_test=True, blend=True)

        assets = mcmodel.Assets(assetdir)
        blockstates = assets.blockstates
        total_variants = sum([ len(b.variants) for b in blockstates ])
        print("Got %d blockstates, %d variants in total" % (len(blockstates), total_variants))

        os.makedirs(outdir, exist_ok=True)
        finfo = open(os.path.join(outdir, "blocks.txt"), "w")
        print("%d %d" % (total_variants, columns), file=finfo)

        images = BlockImages()
        solid_uv_index = None

        for blockstate in blockstates:
            print("Taking block %s" % blockstate.name)

            glblock = render.Block(blockstate)
            for index, variant in enumerate(blockstate.variants):
                print("Rendering variant %d/%d" % (index+1, len(blockstate.variants)))

                modes = ["color", "uv"]
                indices = []
                for mode in modes:
                    gloo.clear(color=True, depth=True)

                    actual_model = np.dot(render.create_model_transform(rotation=rotation), model)

                    glblock.render(variant, actual_model, view, projection, mode=mode)

                    image = Image.fromarray(fbo.read("color"))
                    index = images.append(image)
                    indices.append(index)

                variant_name = mcmodel.encode_variant(variant)
                if variant_name == "":
                    variant_name = "-"
                block_filename = "%s_%d" % (blockstate.name, index)

                print("%s %s color=%d,uv=%d" % (blockstate.name, variant_name, indices[0], indices[1]), file=finfo)

        images.export(columns).save(os.path.join(outdir, "blocks.png"))

        finfo.close()
        fbo.deactivate()
        self.close()

if __name__ == "__main__":
    import vispy
    if "--osmesa" in sys.argv[1:]:
        vispy.use("osmesa")
        assert vispy.app.use_app().backend_name == "osmesa"

    c = Canvas()
    app.run()
