#!/usr/bin/env python

import sys
import numpy as np
import json
import math
from PIL import Image
from glumpy import app, gl, glm, gloo, data, key
from glumpy import transforms

import model
import render

window = app.Window(color=(0.30, 0.30, 0.35, 1.00))

view = np.eye(4, dtype=np.float32)
projection = np.eye(4, dtype=np.float32)

#cube = render.Cube([])

m = model.load_model(sys.argv[1])
element = render.Element(m["elements"][0], m["textures"])
mmm = render.Model(m)

first_render = True
run_rotation = False
phi = 15

@window.event
def on_draw(dt):
    global first_render, phi, projection

    window.clear()

    if run_rotation:
        phi += 0.2

    texture = None
    fbo = None
    if first_render:
        w, h = 32, 32
        texture = np.zeros((h, w, 4), dtype=np.uint8).view(gloo.Texture2D)
        depth = np.zeros((h, w), dtype=np.float32).view(gloo.DepthTexture)
        fbo = gloo.FrameBuffer(color=[texture], depth=depth)
        fbo.activate()

        projection = glm.ortho(-1, 1, -1, 1, 2.0, 50.0)
        glm.scale(projection, 1.0, -1.0, 1.0)

        gl.glViewport(0, 0, w, h)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_DEPTH_TEST)

    model = np.eye(4, dtype=np.float32)
    # for ortho
    glm.scale(model, 1.0, 0.816, 1.0)
    glm.scale(model, 1.0 / math.sqrt(2))
    #glm.scale(model, 0.5)
    glm.rotate(model, 45, 0, 1, 0)
    # hmm it's about 19 degrees?
    glm.rotate(model, 30, 1, 0, 0)

    # for perspective
    #glm.rotate(model, phi, 0, 1, 0)
    #glm.rotate(model, 25, 1, 0, 0)

    mmm.render(model, view, projection)

    if first_render:
        image = Image.fromarray(texture.get())
        image.save("block.png")
        fbo.deactivate()
        first_render = False

@window.event
def on_resize(width, height):
    global view, projection
    view = glm.translation(0, 0, -5)
    aspect = width / height
    #projection = glm.perspective(45.0, width / float(height), 2.0, 100.0)
    projection = glm.ortho(-aspect, aspect, -1.0, 1.0, 2.0, 50.0)

@window.event
def on_key_press(code, mod):
    global run_rotation

    if code == ord("Q"):
        window.close()

    if code == key.SPACE:
        run_rotation = not run_rotation

@window.event
def on_init():
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glDepthFunc(gl.GL_LESS)

    #gl.glEnable(gl.GL_CULL_FACE)
    
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendEquationSeparate(gl.GL_FUNC_ADD, gl.GL_FUNC_ADD)
    gl.glBlendFuncSeparate(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA, gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA)

app.run()
