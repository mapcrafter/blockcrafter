import os
import numpy as np
import math
import time
from PIL import Image
from vispy import app, gloo, io, geometry
from glumpy import glm

import mcmodel

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

FRAGMENT = """
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
    vec4 t_color0 = texture2D(u_texture, v_texcoord0);
    if (t_color.a <= 0.00001) {
        discard;
    }
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
}
"""

FRAGMENT_TEXCOORD = """
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

    gl_FragColor = vec4(vec3(v_texcoord.xy, float(u_face_index) / 6.0), 1.0);
}
"""


FRAGMENT_COLOR = """
uniform vec4 u_color;

void main() {
    gl_FragColor = u_color;
}
"""

class Lines(gloo.Program):
    def __init__(self, count, vertex=VERTEX, fragment=FRAGMENT_COLOR):
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
        [0, 1, 2, 3],
        [5, 4, 7, 6],
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
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
    ]

    def __init__(self, elementdef, texturesdef):
        super().__init__()

        self.elementdef = elementdef

        self.faces = Element.load_faces(elementdef, texturesdef)
        self.rotation = elementdef.get("rotation", None)

        self.xyz0 = (np.array(elementdef["from"]) - 8.0) / 16.0 * 2.0
        self.xyz1 = (np.array(elementdef["to"]) - 8.0) / 16.0 * 2.0

        self.scale = (self.xyz1 - self.xyz0) * 0.5
        self.translate = (self.xyz1 + self.xyz0) * 0.5
        self.points = np.array(Element.CUBE_POINTS) * self.scale + self.translate
        self.indices = gloo.IndexBuffer(np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32))

        self.program = gloo.Program(VERTEX, FRAGMENT)
        self.program["a_position"] = gloo.VertexBuffer(np.zeros((4, 3), dtype=np.float32))
        self.program["a_normal"] = gloo.VertexBuffer(np.zeros((4, 3), dtype=np.float32))

    def render_face(self, face_index, texture, uvs, model, view, projection, element_transform, uvlock):
        program = self.program

        points = self.points[Element.CUBE_FACES[face_index]].astype(np.float32)
        normal = np.array(Element.CUBE_NORMALS[face_index], dtype=np.float32)
        program["a_position"].set_data(points)
        program["a_normal"].set_data(np.stack([normal] * 4))

        uv0, uv1 = uvs
        scale = (uv1 - uv0) * 0.5
        translate = (uv1 + uv0) * 0.5
        program["a_texcoord"] = np.array(Element.CUBE_TEXCOORDS, dtype=np.float32) * scale + translate

        program["u_texture"] = texture
        program["u_light_direction"] = [-0.1, 1.0, 1.0]

        # actual texture dir in model coordinates
        texture_dir = np.array(Element.CUBE_TEXTURE_DIRS[face_index], dtype=np.float32)

        # I am going to speak about world coordinates now
        # but actually I mean the model coordinates after element is rotated by element/block

        # get the face normal in world coordinates to determine where texture should point to
        target_texture_dir = None
        cube_normal = np.round(np.dot(np.append(normal, [0]), element_transform)[:3])
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
        # if uvlock is wanted, apply correction now to texture
        if uvlock:
            texture_transform = np.eye(4, dtype=np.float32)
            glm.rotate(texture_transform, angle, 0, 0, 1)
            program["u_texcoord"] = texture_transform
        else:
            program["u_texcoord"] = np.eye(4, dtype=np.float32)
        program.draw("triangles", self.indices)

        # code to debug normals / texture direction stuff
        #center = np.sum(points, axis=0) / len(points) + 0.001 * normal
        #draw_line(center, center + normal * 0.5, model, view, projection, (1.0, 1.0, 0.0, 1.0))
        #draw_line(center, center + texture_dir * 0.5, model, view, projection, (0.0, 1.0, 1.0, 1.0))
        #center += 0.005 * normal
        #draw_line(center, center + cube_target_texture_dir * 0.5, model, view, projection, (0.0, 1.0, 0.0, 1.0))
        #if direction < 0:
        #    draw_line(center, center + normal * 0.25, model, view, projection, (1.0, 0.0, 0.0, 1.0))

    def render(self, model, view, projection, element_transform=np.eye(4, dtype=np.float32), uvlock=False):
        rotation = np.eye(4, dtype=np.float32)
        if "rotation" in self.elementdef:
            rotationdef = self.elementdef["rotation"]
            axis = {"x" : [1, 0, 0],
                    "y" : [0, 1, 0],
                    "z" : [0, 0, 1]}[rotationdef["axis"]]
            origin = (np.array(rotationdef.get("origin", [8, 8, 8]), dtype=np.float32) - 8.0) / 16.0 * 2.0
            origin *= -1.0
            glm.translate(rotation, *origin)
            glm.rotate(rotation, rotationdef["angle"], *axis)
            origin *= -1.0
            glm.translate(rotation, *origin)

        program = self.program

        # add rotation of world (only 90*x degrees) to element transformation
        element_transform = np.dot(rotation, element_transform)
        complete_model = np.dot(element_transform, model)
        program["u_model"] = complete_model
        program["u_view"] = view
        program["u_projection"] = projection
        program["u_normal"] = np.array(np.matrix(np.dot(view, complete_model)).I.T)

        for i, (texture, uvs) in enumerate(self.faces):
            if texture is None:
                continue
            self.render_face(i, texture, uvs, complete_model, view, projection, element_transform=element_transform, uvlock=uvlock)

    @staticmethod
    def load_faces(elementdef, texturesdef):
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
        for direction, facedef in elementdef["faces"].items():
            path = mcmodel.resolve_texture(texturesdef, facedef["texture"])
            if path is None:
                raise RuntimeError("Face in direction '%s' has no texture associated" % direction)
            uvs = np.array(facedef.get("uv", [0, 0, 16, 16]), dtype=np.float32) / 16.0
            uv0, uv1 = uvs[:2], uvs[2:]
            faces[direction] = (gloo.Texture2D(data=io.imread(path)), (uv0, uv1))

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
        for elementdef in modeldef["elements"]:
            self.elements.append(Element(elementdef, modeldef["textures"]))

    def render(self, model, view, projection, modelref={}, rotation=0):
        m = np.eye(4, dtype=np.float32)
        if "x" in modelref:
            glm.rotate(m, modelref["x"], 1, 0, 0)
        if "y" in modelref:
            glm.rotate(m, modelref["y"], 0, 1, 0)
        glm.rotate(m, rotation * 90, 0, 1, 0)

        uvlock = modelref.get("uvlock", False)
        for element in self.elements:
            element.render(model, view, projection, element_transform=m, uvlock=uvlock)

class Block:
    def __init__(self, blockdef):
        self.blockdef = blockdef
        self.models = {}
        self.variants = {}

    def _load_modeldef(self, name):
        modeldef = mcmodel.load_modeldef(os.path.join(mcmodel.MODEL_BASE, name + ".json"))
        model = Model(modeldef)
        return model

    def _load_variant(self, variant):
        modelrefs = []
        for modelref in mcmodel.get_blockdef_modelrefs(self.blockdef, variant):
            # TODO
            if type(modelref) == list:
                modelref = modelref[0]
            if not modelref["model"] in self.models:
                self.models[modelref["model"]] = self._load_modeldef(modelref["model"])
            modelrefs.append(modelref)
        return modelrefs

    def render(self, variant, model, view, projection, rotation=0):
        variant_str = mcmodel.encode_variant(variant)
        if variant_str not in self.variants:
            self.variants[variant_str] = self._load_variant(variant)

        modelrefs = self.variants[variant_str]
        for modelref in modelrefs:
            glmodel = self.models[modelref["model"]]
            glmodel.render(model, view, projection, modelref=modelref, rotation=rotation)

def create_transform_ortho(aspect=1.0, offscreen=False, fake_ortho=True):
    model = np.eye(4, dtype=np.float32)
    if fake_ortho:
        # 0.816479 = 0.5 * sqrt(3) * x = 0.5 * sqrt(2)
        # scale of y-axis to make sides and top of same height
        glm.scale(model, 1.0, 0.816479, 1.0)
        # scale to get block completely into viewport (-1;1)
        glm.scale(model, 1.0 / math.sqrt(2))
    else:
        glm.scale(model, 0.5)
    glm.rotate(model, 45, 0, 1, 0)
    glm.rotate(model, 30, 1, 0, 0)

    view = glm.translation(0, 0, -5)

    projection = glm.ortho(-aspect, aspect, -1, 1, 2.0, 50.0)
    if offscreen:
        glm.scale(projection, 1.0, -1.0, 1.0)

    return model, view, projection

def create_transform_perspective(aspect=1.0, offscreen=False):
    model = np.eye(4, dtype=np.float32)
    glm.rotate(model, 45, 0, 1, 0)
    glm.rotate(model, 25, 1, 0, 0)

    view = glm.translation(0, 0, -5)

    projection = glm.perspective(45.0, aspect, 2.0, 50.0)
    if offscreen:
        glm.scale(projection, 1.0, -1.0, 1.0)

    return model, view, projection

def create_model_transform(rotation=0, phi=0.0):
    model = np.eye(4, dtype=np.float32)
    glm.rotate(model, rotation * 90 + phi, 0, 1, 0)
    return model

