#!/usr/bin/env python

import sys
import numpy as np
import math
import random
from vispy import app, gloo
from vispy.util import transforms

import mcmodel
import render

views = ["perspective", "isometric", "topdown"]
rotations = ["top-left", "top-right", "bottom-right", "bottom-left"]
modes = ["color", "uv"]
blending_modes = ["translucent", "premultiplied"] #, "opaque"]

class Canvas(app.Canvas):
    def __init__(self, blockstate):
        super().__init__(keys="interactive")

        self.model, self.view, self.projection = None, None, None

        self.run_phi = False
        self.phi = 0

        self.blockstate = blockstate
        self.glblock = render.Block(blockstate)
        self.variants = blockstate.variants
        print("Block has variants:")
        for variant in self.variants:
            print("-", variant)

        self.variant_index = 0
        self.view_index = 0
        self.rotation_index = 0
        self.mode_index = 0
        self.blending_mode_index = 0

        #gloo.gl.glEnable(gloo.gl.GL_DEPTH_TEST)
        #gloo.gl.glDepthFunc(gloo.gl.GL_LESS)

        #gloo.gl.glEnable(gloo.gl.GL_BLEND)
        #gloo.gl.glBlendEquationSeparate(gloo.gl.GL_FUNC_ADD, gloo.gl.GL_FUNC_ADD)
        #gloo.gl.glBlendFuncSeparate(gloo.gl.GL_SRC_ALPHA, gloo.gl.GL_ONE_MINUS_SRC_ALPHA, gloo.gl.GL_ONE, gloo.gl.GL_ONE_MINUS_SRC_ALPHA)

        self._timer = app.Timer("auto", connect=self.on_timer, start=True)

        self.show()

    def on_resize(self, event):
        self.model, self.view, self.projection = None, None, None

        w, h = event.physical_size
        gloo.set_viewport(0, 0, w, h)

    def on_key_press(self, event):
        if event.key == "v":
            self.view_index = (self.view_index + 1) % len(views)
            self.model, self.view, self.projection = None, None, None

        if event.key == "Left":
            self.rotation_index = (self.rotation_index - 1) % len(rotations)
            print("Rendering rotation %d: %s" % (self.rotation_index, rotations[self.rotation_index]))

        if event.key == "Right":
            self.rotation_index = (self.rotation_index + 1) % len(rotations)
            print("Rendering rotation %d: %s" % (self.rotation_index, rotations[self.rotation_index]))

        if event.key == "Down":
            self.variant_index = (self.variant_index - 1) % len(self.variants)
            print("Rendering variant %d: %s" % (self.variant_index, self.variants[self.variant_index]))
        if event.key == "Up":
            self.variant_index = (self.variant_index + 1) % len(self.variants)
            print("Rendering variant %d: %s" % (self.variant_index, self.variants[self.variant_index]))

        if event.key == "m":
            self.mode_index = (self.mode_index + 1) % len(modes)

        if event.key == "b":
            self.blending_mode_index = (self.blending_mode_index + 1) % len(blending_modes)
            print("Blending mode: %s" % blending_modes[self.blending_mode_index])

        if event.key == "Space" or event.key == " ":
            self.run_phi = not self.run_phi

        if event.key == ord("Q"):
            self.close()

    def on_timer(self, event):
        self.update()

    def on_draw(self, event):
        blending_mode = blending_modes[self.blending_mode_index]
        render.set_blending(blending_mode)

        gloo.set_clear_color((0.30, 0.30, 0.35, 1.00))
        gloo.clear(color=True, depth=True)
        w, h = self.physical_size

        if self.model is None:
            aspect = w / h
            v = views[self.view_index]

            if v == "perspective":
                self.model, self.view, self.projection = render.create_transform_perspective(aspect=aspect)
            elif v == "isometric":
                self.model, self.view, self.projection = render.create_transform_ortho(aspect=aspect, view="isometric", fake_ortho=True)
            elif v == "topdown":
                self.model, self.view, self.projection = render.create_transform_ortho(aspect=aspect, view="topdown", fake_ortho=True)
            else:
                assert False, "Invalid view type '%s'" % view

        rotation = self.rotation_index
        if self.run_phi:
            self.phi += 0.2
        actual_model = render.apply_model_rotation(self.model, rotation=0, phi=self.phi)

        current_variant = self.variants[self.variant_index]
        current_mode = modes[self.mode_index]
        self.glblock.render(current_variant, actual_model, self.view, self.projection, rotation=rotation, mode=current_mode)

        v = lambda *a: np.array(a, dtype=np.float32)
        render.draw_line(v(0, 0, 0), v(10, 0, 0), actual_model, self.view, self.projection, color=(1, 0, 0, 1))
        render.draw_line(v(0, 0, 0), v(0, 10, 0), actual_model, self.view, self.projection, color=(0, 1, 0, 1))
        render.draw_line(v(0, 0, 0), v(0, 0, 10), actual_model, self.view, self.projection, color=(0, 0, 1, 1))

if __name__ == "__main__":
    assets = mcmodel.Assets(sys.argv[1])
    blockstate = assets.get_blockstate(sys.argv[2])

    c = Canvas(blockstate)
    app.run()
