#!/usr/bin/env python

import os
import sys
import numpy as np
import json
import math
import time
from PIL import Image
from glumpy import app, gl, glm, gloo, data, key
from glumpy import transforms

import mcmodel
import render

window = app.Window()

w, h = 32, 32
rotation = 0
outdir = "out"

@window.event
def on_draw(dt):
    window.clear()

    model, view, projection = render.create_transform_ortho(aspect=1.0, fake_ortho=True, offscreen=True)

    texture = np.zeros((h, w, 4), dtype=np.uint8).view(gloo.Texture2D)
    depth = np.zeros((h, w), dtype=np.float32).view(gloo.DepthTexture)
    fbo = gloo.FrameBuffer(color=[texture], depth=depth)
    fbo.activate()

    gl.glViewport(0, 0, w, h)

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
            gl.glClearColor(0.0, 0.0, 0.0, 0.0)
            # really clear depth buffer
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            actual_model = np.dot(render.create_model_transform(rotation=rotation), model)

            #modeldef = mcmodel.load_model(sys.argv[1])
            #glmodel = render.Model(modeldef)
            #glmodel.render(actual_model, view, projection)
            glblock.render(variant, actual_model, view, projection)

            variant_name = mcmodel.encode_variant(variant)
            if variant_name == "":
                variant_name = "-"
            block_filename = "%s_%d.png" % (blockname, index)
            print("%s %s %s" % (blockname, variant_name, block_filename), file=finfo)

            image = Image.fromarray(texture.get())
            image.save(os.path.join(outdir, block_filename))

    finfo.close()

    fbo.deactivate()
    window.close()

@window.event
def on_init():
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glDepthFunc(gl.GL_LESS)

    #gl.glEnable(gl.GL_CULL_FACE)
    
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendEquationSeparate(gl.GL_FUNC_ADD, gl.GL_FUNC_ADD)
    gl.glBlendFuncSeparate(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA, gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA)

app.run()
