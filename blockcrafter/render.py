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
import numpy as np
import math
import time
from PIL import Image
from vispy import app, gloo, geometry
from vispy.util import transforms

from blockcrafter import mcmodel

VERTEX = """
attribute vec3 a_position;
attribute vec2 a_texcoord;
attribute vec3 a_normal;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform mat4 u_texcoord;

varying vec3 v_position;
varying vec2 v_texcoord;
varying vec2 v_texcoord0;
varying vec3 v_normal;

void main() {
    v_position = a_position;

    // TODO this seems to cause a slight interpolation somehow
    v_texcoord = (u_texcoord * (vec4(a_texcoord - 0.5, 0.0, 1.0))).xy + 0.5;
    v_texcoord0 = a_texcoord;
    v_normal = a_normal;

    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
}
"""

FRAGMENT_BLOCK_COLOR = """
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform mat4 u_normal;

uniform sampler2D u_texture;
uniform vec3 u_light_direction;
uniform int u_face_index;

varying vec3 v_position;
varying vec2 v_texcoord;
varying vec2 v_texcoord0;
varying vec3 v_normal;

void main() {
    vec4 t_color = texture2D(u_texture, v_texcoord);

    //vec4 t_color0 = texture2D(u_texture, v_texcoord0);
    if (t_color.a <= 0.00001) {
        discard;
    }

    gl_FragColor = t_color;

    /*
    // debugging for uvlock-rotated textures
    //t_color.rgb = mix(t_color.rgb, t_color0.rgb, 0.2);

    // calculate normal in eye space
    vec3 n = normalize((u_normal * vec4(v_normal, 1.0)).xyz);

    // two dot products and then max out of it because
    // I don't care so much about backsides not lit
    float d1 = dot(n,  u_light_direction);
    float d2 = dot(-n, u_light_direction);
    float d = max(d1, d2);

    // intensity of the light
    // the sqrt is just a mapping how I think it looks nice
    float intensity = min(max(d, 0.0), 1.0);
    intensity = sqrt(intensity);
    //gl_FragColor = vec4(vec3(intensity), 1.0);

    // how much is light applied
    float alpha = 1.0;
    gl_FragColor = vec4(t_color.rgb * (alpha * intensity + (1.0 - alpha)), t_color.a);
    */
}
"""

FRAGMENT_BLOCK_UV = """
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform mat4 u_normal;

uniform sampler2D u_texture;
uniform vec3 u_light_direction;
uniform int u_face_index;

varying vec3 v_position;
varying vec2 v_texcoord;
varying vec3 v_normal;

void main() {
    vec4 t_color = texture2D(u_texture, v_texcoord);
    if (t_color.a <= 0.00001) {
        discard;
    }

    float face = float(u_face_index) / 6.0;

    // Process the value of the Z position adn scale it to be
    // stored in the alpha channel of pixel, hence why
    // the blending is disabled for UV mode.
    // It's purpose is mainly to merge blocks, such as waterlog
    vec4 t_rot = u_model * vec4(v_position, 1.0);
    float t_z = min( 1/256 + max((t_rot.z + 1.0) * 0.5, 0), 1.0);

    gl_FragColor = vec4(vec3(v_texcoord.xy, face), t_z);
}
"""

FRAGMENT_LINE_COLOR = """
uniform vec4 u_color;

void main() {
    gl_FragColor = u_color;
}
"""

class Lines(gloo.Program):
    def __init__(self, count, vertex=VERTEX, fragment=FRAGMENT_LINE_COLOR):
        super().__init__(vertex, fragment, count=count)

        self["a_position"] = gloo.VertexBuffer(np.zeros((count, 3), dtype=np.float32))

    def render(self, points, model, view, projection, color=(1.0, 1.0, 1.0, 1.0)):
        if not isinstance(points[0], (tuple, list)):
            points = np.stack(points, axis=0)
        self["a_position"].set_data(points)

        self["u_model"] = model
        self["u_view"] = view
        self["u_projection"] = projection

        self["u_color"] = color

        self.draw(gloo.gl.GL_LINE_STRIP)

line_program = None
def draw_line(p0, p1, model, view, projection, color=(1.0, 1.0, 1.0, 1.0)):
    global line_program
    if line_program is None:
        line_program = Lines(count=2)

    line_program.render([p0, p1], model, view, projection, color=color)

# from stackoverflow: https://stackoverflow.com/a/13849249
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

class Element:

    CUBE_POINTS = [
        [ 1,  1,  1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1,  1,  1],
        [ 1, -1,  1],
        [ 1, -1, -1],
        [-1, -1, -1],
        [-1, -1,  1],
    ]

    CUBE_FACES = [
        [1, 0, 4, 5],
        [3, 2, 6, 7],
        [1, 2, 3, 0],
        [6, 5, 4, 7],
        [0, 3, 7, 4],
        [2, 1, 5, 6],
    ]

    CUBE_TEXCOORDS = [
        ( 1, -1),
        (-1, -1),
        (-1,  1),
        ( 1,  1),
    ]

    CUBE_NORMALS = [
        [1,  0,  0],
        [-1, 0,  0],
        [0,  1,  0],
        [0, -1,  0],
        [0,  0,  1],
        [0,  0, -1],
    ]

    CUBE_TEXTURE_DIRS = [
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, -1],
        [0, 0, -1],
        [0, 1, 0],
        [0, 1, 0],
    ]

    _color_program = None 
    _uv_program = None

    @classmethod
    def get_program(cls, mode):
        if mode == "color":
            if cls._color_program is None:
                cls._color_program = gloo.Program(VERTEX, FRAGMENT_BLOCK_COLOR)
                cls._color_program["a_position"] = gloo.VertexBuffer(np.zeros((4, 3), dtype=np.float32))
                cls._color_program["a_normal"] = gloo.VertexBuffer(np.zeros((4, 3), dtype=np.float32))
            return cls._color_program
        if mode == "uv":
            if cls._uv_program is None:
                cls._uv_program = gloo.Program(VERTEX, FRAGMENT_BLOCK_UV)
                cls._uv_program["a_position"] = gloo.VertexBuffer(np.zeros((4, 3), dtype=np.float32))
                cls._uv_program["a_normal"] = gloo.VertexBuffer(np.zeros((4, 3), dtype=np.float32))
            return cls._uv_program
        assert False, "Invalid mode!"

    def __init__(self, model, element):
        super().__init__()

        self.faces = Element.load_faces(model, element)
        self.rotation = element.get("rotation", None)

        self.xyz0 = (np.array(element["from"]) - 8.0) / 16.0 * 2.0
        self.xyz1 = (np.array(element["to"]) - 8.0) / 16.0 * 2.0

        self.scale = (self.xyz1 - self.xyz0) * 0.5
        self.translate = (self.xyz1 + self.xyz0) * 0.5
        self.points = np.array(Element.CUBE_POINTS) * self.scale + self.translate
        self.indices = gloo.IndexBuffer(np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32))

    def render_face(self, face_index, texture, uvs, model, view, projection, element_rotation, element_transform, uvlock):
        program = self.current_program

        # ---
        # --- set up attributes ---#
        # ---

        points = self.points[Element.CUBE_FACES[face_index]].astype(np.float32)
        normal = np.array(Element.CUBE_NORMALS[face_index], dtype=np.float32)
        program["a_position"].set_data(points + 0.00001 * normal)
        program["a_normal"].set_data(np.stack([normal] * 4))

        uv0, uv1 = uvs
        scale = (uv1 - uv0) * 0.5
        translate = (uv1 + uv0) * 0.5
        program["a_texcoord"] = np.array(Element.CUBE_TEXCOORDS, dtype=np.float32) * scale + translate

        # ---
        # --- set up uniforms ###
        # ---

        # we need to do some transformation magic to handle uvlock correctly

        # actual texture dir in model coordinates
        texture_dir = np.array(Element.CUBE_TEXTURE_DIRS[face_index], dtype=np.float32)

        # I am going to speak about world coordinates now
        # but actually I mean the model coordinates after element is rotated by element/block

        # get the face normal in world coordinates to determine where texture should point to
        # (that face normal in world coordinates describes as which face actually this face appears to viewer)
        cube_normal = np.round(np.dot(np.append(normal, [0]), element_transform)[:3])

        # if uvlock is wanted, apply correction now to texture
        if uvlock:
            target_texture_dir = None
            if abs(cube_normal[1]) > 0.001:
                # top face, should point to east
                target_texture_dir = np.array(Element.CUBE_TEXTURE_DIRS[2], dtype=np.float32)
            else:
                # side face, should point to top
                target_texture_dir = np.array(Element.CUBE_TEXTURE_DIRS[0], dtype=np.float32)
            # go from world -> model
            element_transform_inv = np.array(np.matrix(element_transform).I)
            target_texture_dir = np.round(np.dot(np.append(target_texture_dir, [0]), element_transform_inv)[:3]).astype(np.float32)

            # now we can get the angle we have to rotate the texture
            angle = np.round(math.degrees(angle_between(texture_dir, target_texture_dir)))
            # also get a direction related to face normal
            direction = np.dot(np.cross(normal, texture_dir), target_texture_dir)
            if direction < 0:
                angle = 360 - angle

            program["u_texcoord"] = transforms.rotate(angle, (0, 0, 1))
        else:
            program["u_texcoord"] = np.eye(4, dtype=np.float32)

        program["u_texture"] = texture
        program["u_light_direction"] = [-0.1, 1.0, 1.0]

        # we pass index of face to shader too
        # (mostly for the shader that exports uv coordinates)

        if uvlock:
            # find face index if element wouldn't be rotated (as which face it appears to viewer)
            # faces rotated not by x*90 degrees just get face index 6
            actual_face = 6
            for i in range(6):
                if np.allclose(Element.CUBE_NORMALS[i], cube_normal):
                    actual_face = i
                    break
            program["u_face_index"] = actual_face
        else:
            program["u_face_index"] = face_index

        # ---
        # --- actual drawing ---
        # ---
        program.draw("triangles", self.indices)

        # old code to debug normals / texture direction stuff
        #center = np.sum(points, axis=0) / len(points) + 0.001 * normal
        #draw_line(center, center + normal * 0.5, model, view, projection, (1.0, 1.0, 0.0, 1.0))
        #draw_line(center, center + texture_dir * 0.5, model, view, projection, (0.0, 1.0, 1.0, 1.0))
        #center += 0.005 * normal
        #draw_line(center, center + cube_target_texture_dir * 0.5, model, view, projection, (0.0, 1.0, 0.0, 1.0))
        #if direction < 0:
        #    draw_line(center, center + normal * 0.25, model, view, projection, (1.0, 0.0, 0.0, 1.0))

    def render(self, model, view, projection, mode="color", block_rotation=0, element_transform=np.eye(4, dtype=np.float32), uvlock=False):
        element_rotation = np.eye(4, dtype=np.float32)
        if self.rotation:
            rotationdef = self.rotation
            axis = {"x" : [1, 0, 0],
                    "y" : [0, 1, 0],
                    "z" : [0, 0, 1]}[rotationdef["axis"]]
            origin = (np.array(rotationdef.get("origin", [8, 8, 8]), dtype=np.float32) - 8.0) / 16.0 * 2.0
            element_rotation = np.dot(transforms.translate(-origin), np.dot(transforms.rotate(rotationdef["angle"], axis), transforms.translate(origin)))

        program = Element.get_program(mode)
        self.current_program = program

        # add rotation of block and element to element transformation
        block_rotation = transforms.rotate(-90 * block_rotation, (0, 1, 0))
        element_transform = np.dot(element_rotation, np.dot(element_transform, block_rotation))
        complete_model = np.dot(element_transform, model)
        program["u_model"] = complete_model
        program["u_view"] = view
        program["u_projection"] = projection
        #program["u_normal"] = np.array(np.matrix(np.dot(view, complete_model)).I.T)

        for i, (texture, uvs) in enumerate(self.faces):
            if texture is None:
                continue
            self.render_face(i, texture, uvs, complete_model, view, projection, element_rotation=element_rotation, element_transform=element_transform, uvlock=uvlock or mode == "uv")

    @staticmethod
    def load_faces(model, element):
        # order of minecraft directions to order of cube sides
        mc_to_opengl = [
            "east",   # pos x
            "west",   # neg x
            "up",     # pos y
            "down",   # neg y
            "south",  # pos z
            "north",  # neg z
        ]

        faces = {}
        for direction, facedef in element["faces"].items():
            # TODO, cache loaded and processed textures!
            texture_name = model.resolve_texture(facedef["texture"])
            if texture_name is None:
                raise RuntimeError("Face in direction '%s' has no texture associated" % direction)
            f = model.load_texture(texture_name)
            uvs = np.array(facedef.get("uv", [0, 0, 16, 16]), dtype=np.float32) / 16.0
            uv0, uv1 = uvs[:2], uvs[2:]

            image = Image.open(f).convert("RGBA")
            if texture_name.startswith("block/") and image.size[0] != image.size[1]:
                assert image.size[0] < image.size[1]
                s = image.size[0]
                image = image.crop((0, 0, s, s))
            if "rotation" in facedef:
                image = image.rotate(-facedef["rotation"])

            data = np.array(image)
            semi_transparent = np.all((data[:, :, 3] == 0) | (data[:, :, 3] == 255))
            w, h = image.size
            image = image.resize((w*2, h*2), resample=Image.NEAREST)
            data = np.array(image)
            if semi_transparent:
                data[:, :, 3] = (data[:, :, 3] > 255/2.0) * 255

            if "blockcrafterTint" in facedef:
                r, g, b = facedef["blockcrafterTint"]
                data[:, :, 0] = data[:, :, 0] * r
                data[:, :, 1] = data[:, :, 1] * g
                data[:, :, 2] = data[:, :, 2] * b
            faces[direction] = (gloo.Texture2D(data=data, interpolation="linear"), (uv0, uv1))
            f.close()

        # gather faces in order for cube sides
        # remember: each side is (texture, (uv0, uv1))
        sides = [ faces.get(direction, None) for direction in mc_to_opengl ]
        # so this is how an non-existant side looks like
        empty = (None, None)
        sides = [ (s if s is not None else empty) for s in sides ]
        assert len(sides) == 6
        return sides

class Model:
    def __init__(self, modeldef):
        self.elements = []
        for elementdef in modeldef.elements:
            self.elements.append(Element(modeldef, elementdef))

    def render(self, model, view, projection, block_rotation=0, mode="color", modelref={}):
        m = np.eye(4, dtype=np.float32)
        if "x" in modelref:
            m = np.dot(m, transforms.rotate(-modelref["x"], (1, 0, 0)))
        if "y" in modelref:
            m = np.dot(m, transforms.rotate(-modelref["y"], (0, 1, 0)))
        if "z" in modelref:
            m = np.dot(m, transforms.rotate(-modelref["z"], (0, 0, 1)))

        uvlock = modelref.get("uvlock", False)
        for element in self.elements:
            element.render(model, view, projection, mode=mode, block_rotation=block_rotation, element_transform=m, uvlock=uvlock)

class Block:
    def __init__(self, blockstate):
        self.blockstate = blockstate
        self.models = {}
        self.variants = {}

    def _load_variant(self, variant):
        modelrefs = []
        for model, transformation in self.blockstate.evaluate_variant(variant):
            if not model.name in self.models:
                self.models[model.name] = Model(model)
            modelrefs.append((self.models[model.name], transformation))
        return modelrefs

    def render(self, variant, model, view, projection, rotation=0, mode="color"):
        variant_str = mcmodel.encode_variant(variant)
        if variant_str not in self.variants:
            self.variants[variant_str] = self._load_variant(variant)

        modelrefs = self.variants[variant_str]
        for glmodel, transformation  in modelrefs:
            glmodel.render(model, view, projection, block_rotation=rotation, mode=mode, modelref=transformation)

def create_transform_ortho(aspect=1.0, view="isometric", fake_ortho=True):
    model = np.eye(4, dtype=np.float32)

    if view == "isometric":
        if fake_ortho:
            # 0.816479 = 0.5 * sqrt(3) * x = 0.5 * sqrt(2)
            # scale of y-axis to make sides and top of same height: (1.0, 0.81649, 1.0)
            # scale to get block completely into viewport (-1;1): (1.0 / math.sqrt(2), ..., ...)
            model = np.dot(model, transforms.scale((1.0 / math.sqrt(2), 0.816479 / math.sqrt(2), 1.0 / math.sqrt(2))))
        else:
            # this scale factor is just for nicely viewing the block image manually
            model = np.dot(model, transforms.scale((0.5, 0.5, 0.5)))

        # and do that nice tilt
        model = np.dot(model, np.dot(transforms.rotate(45, (0, 1, 0)), transforms.rotate(30, (1, 0, 0))))
    elif view == "topdown":
        model = np.dot(model, transforms.rotate(90, (1, 0, 0)))
    elif view == "side":
        # same thing with scaling factor as with isometric view
        #f = 1.0 / math.sqrt(2)
        f = 0.5 / math.cos(math.radians(45))
        model = np.dot(model, transforms.scale((f / math.cos(math.radians(45)), f, f)))
        model = np.dot(model, transforms.rotate(45, (1, 0, 0)))
    elif view == "default":
        pass
    else:
        assert False, "Invalid view '%s'!" % view

    view = transforms.translate((0, 0, -5))
    projection = transforms.ortho(-aspect, aspect, -1, 1, 2.0, 50.0)
    return model, view, projection

def create_transform_perspective(aspect=1.0):
    model = np.dot(transforms.rotate(45, (0, 1, 0)), transforms.rotate(25, (1, 0, 0)))
    view = transforms.translate((0, 0, -5))
    projection = transforms.perspective(45.0, aspect, 2.0, 50.0)
    return model, view, projection

def apply_model_rotation(model, rotation=0, phi=0.0):
    rotation = transforms.rotate(-rotation * 90 + phi, (0, 1, 0))
    return np.dot(rotation, model)

def apply_face_culling(on=True):
    if on:
        gloo.set_state(cull_face=True)
    else:
        gloo.set_state(cull_face=False)

def set_blending(mode):
    if mode == "premultiplied":
        gloo.set_state(blend=True, depth_test=True)
        gloo.gl.glBlendEquationSeparate(gloo.gl.GL_FUNC_ADD, gloo.gl.GL_FUNC_ADD)
        gloo.gl.glBlendFuncSeparate(gloo.gl.GL_ONE, gloo.gl.GL_ONE_MINUS_SRC_ALPHA, gloo.gl.GL_ONE, gloo.gl.GL_ONE_MINUS_SRC_ALPHA)
    else:
        gloo.set_state(mode, clear_color=(0.0, 0.0, 0.0, 0.0))
