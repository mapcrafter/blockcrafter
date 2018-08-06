#!/usr/bin/env python

import os
import sys
import numpy as np
import math
from PIL import Image
from vispy import app, gloo, io, geometry
from glumpy import glm

import mcmodel
import render

w, h = 32, 32
rotation = 0
outdir = "out"

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
        
        # flippying y does not seem to be required with vispy's fbo read-method
        model, view, projection = render.create_transform_ortho(aspect=1.0, fake_ortho=True, offscreen=False)

        texture = gloo.Texture2D(shape=(w, h, 4))
        depth = gloo.RenderBuffer(shape=(w, h))
        fbo = gloo.FrameBuffer(color=texture, depth=depth)
        fbo.activate()

        gloo.set_viewport(0, 0, w, h)
        gloo.set_clear_color((0.0, 0.0, 0.0, 0.0))
        gloo.set_state(depth_test=True, blend=True)

        finfo = open(os.path.join(outdir, "blocks.txt"), "w")
        for path in sys.argv[1:]:
            filename = os.path.basename(path)
            blockname = filename.replace(".json", "")
            print("Taking block %s" % blockname)
            blockdef = mcmodel.load_blockdef(path)
            variants = mcmodel.get_blockdef_variants(blockdef)

            glblock = render.Block(blockdef)
            for index, variant in enumerate(variants):
                print("Rendering variant %d/%d" % (index+1, len(variants)))
                gloo.clear(color=True, depth=True)

                actual_model = np.dot(render.create_model_transform(rotation=rotation), model)

                #modeldef = mcmodel.load_model(sys.argv[1])
                #glmodel = render.Model(modeldef)
                #glmodel.render(actual_model, view, projection)
                try:
                    glblock.render(variant, actual_model, view, projection)
                except np.linalg.linalg.LinAlgError:
                    print("Unable to render block %s variant %d" % (blockname, index+1))
                    continue

                variant_name = mcmodel.encode_variant(variant)
                if variant_name == "":
                    variant_name = "-"
                block_filename = "%s_%d.png" % (blockname, index)
                print("%s %s %s" % (blockname, variant_name, block_filename), file=finfo)

                image = Image.fromarray(fbo.read("color"))
                image.save(os.path.join(outdir, block_filename))

        finfo.close()
        fbo.deactivate()
        self.close()

if __name__ == "__main__":
    c = Canvas()
    app.run()
