#!/usr/bin/env python

import sys
import numpy as np
import math
from vispy import app, gloo

import mcmodel
import render

blockdef = mcmodel.load_blockdef(sys.argv[1])
glblock = render.Block(blockdef)
variants = sorted(mcmodel.get_blockdef_variants(blockdef), key=mcmodel.encode_variant)
print("Block has variants:")
for variant in variants:
    print("-", variant)

views = ["perspective", "ortho", "fake_ortho"]
rotations = ["top-left", "top-right", "bottom-right", "bottom-left"]
modes = ["color", "uv"]

class Canvas(app.Canvas):
    def __init__(self):
        super().__init__(keys="interactive")

        self.model, self.view, self.projection = None, None, None

        self.run_phi = False
        self.phi = 0

        self.variant_index = 0
        self.view_index = 0
        self.rotation_index = 0
        self.mode_index = 0

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

        if event.key == "Right":
            self.rotation_index = (self.rotation_index + 1) % len(rotations)

        if event.key == "Down":
            self.variant_index = (self.variant_index - 1) % len(variants)
            print("Rendering variant %d: %s" % (self.variant_index, variants[self.variant_index]))
        if event.key == "Up":
            self.variant_index = (self.variant_index + 1) % len(variants)
            print("Rendering variant %d: %s" % (self.variant_index, variants[self.variant_index]))

        if event.key == "m":
            self.mode_index = (self.mode_index + 1) % len(modes)

        if event.key == "Space":
            self.run_phi = not self.run_phi

        if event.key == ord("Q"):
            self.close()

    def on_timer(self, event):
        self.update()

    def on_draw(self, event):
        gloo.set_state(depth_test=True, blend=True)

        gloo.set_clear_color((0.30, 0.30, 0.35, 1.00))
        gloo.clear(color=True, depth=True)
        w, h = self.physical_size

        if self.model is None:
            aspect = w / h
            v = views[self.view_index]

            if v == "perspective":
                self.model, self.view, self.projection = render.create_transform_perspective(aspect=aspect)
            elif v == "ortho":
                self.model, self.view, self.projection = render.create_transform_ortho(aspect=aspect, fake_ortho=False)
            elif v == "fake_ortho":
                self.model, self.view, self.projection = render.create_transform_ortho(aspect=aspect, fake_ortho=True)
            else:
                assert False, "Invalid view type '%s'" % view

        rotation = self.rotation_index
        if self.run_phi:
            self.phi += 0.2
        actual_model = np.dot(render.create_model_transform(0, self.phi), self.model)

        current_variant = variants[self.variant_index]
        current_mode = modes[self.mode_index]
        glblock.render(current_variant, actual_model, self.view, self.projection, mode=current_mode, rotation=rotation)

        v = lambda *a: np.array(a, dtype=np.float32)
        render.draw_line(v(0, 0, 0), v(10, 0, 0), actual_model, self.view, self.projection, color=(1, 0, 0, 1))
        render.draw_line(v(0, 0, 0), v(0, 10, 0), actual_model, self.view, self.projection, color=(0, 1, 0, 1))
        render.draw_line(v(0, 0, 0), v(0, 0, 10), actual_model, self.view, self.projection, color=(0, 0, 1, 1))

if __name__ == "__main__":
    c = Canvas()
    app.run()
