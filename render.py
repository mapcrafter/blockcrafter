import os
import numpy as np
import math
import time
from PIL import Image
#from glumpy import app, gl, glm, gloo, data
from vispy import app, gloo, io, geometry
from glumpy import glm

import mcmodel

def geom_cube():
    vtype = [('a_position', np.float32, 3),
             ('a_texcoord', np.float32, 2),
             ('a_normal',   np.float32, 3)]
    itype = np.uint32

    # Vertices positions
    p = np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
                  [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1]], dtype=float)
    # Face Normals
    n = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0],
                  [-1, 0, 1], [0, -1, 0], [0, 0, -1]])
    # Texture coords
    t = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])

    faces_p = [0, 1, 2, 3,  0, 3, 4, 5,   0, 5, 6, 1,
               1, 6, 7, 2,  7, 4, 3, 2,   4, 7, 6, 5]
    faces_n = [0, 0, 0, 0,  1, 1, 1, 1,   2, 2, 2, 2,
               3, 3, 3, 3,  4, 4, 4, 4,   5, 5, 5, 5]
    faces_t = [0, 1, 2, 3,  0, 1, 2, 3,   0, 1, 2, 3,
               3, 2, 1, 0,  0, 1, 2, 3,   0, 1, 2, 3]

    vertices = np.zeros(24, vtype)
    vertices['a_position'] = p[faces_p]
    vertices['a_normal']   = n[faces_n]
    #vertices['a_texcoord'] = t[faces_t]

    filled = np.resize(
       np.array([0, 1, 2, 0, 2, 3], dtype=itype), 6 * (2 * 3))
    filled += np.repeat(4 * np.arange(6, dtype=itype), 6)

    outline = np.resize(
        np.array([0, 1, 1, 2, 2, 3, 3, 0], dtype=itype), 6 * (2 * 4))
    outline += np.repeat(4 * np.arange(6, dtype=itype), 8)
    vertices = vertices.view(gloo.VertexBuffer)
    filled   = filled.view(gloo.IndexBuffer)

    return vertices, filled

vertex = """
uniform mat4   u_model;         // Model matrix
uniform mat4   u_view;          // View matrix
uniform mat4   u_projection;    // Projection matrix
attribute vec3 a_position;      // Vertex position
attribute vec3 a_normal;        // Vertex normal
//attribute vec2 a_texcoord;      // Vertex texture coordinates
varying vec3   v_normal;        // Interpolated normal (out)
varying vec3   v_position;      // Interpolated position (out)
varying vec3   v_texcoord;      // Interpolated fragment texture coordinates (out)

void main()
{
    // Assign varying variables
    v_normal   = a_normal;
    v_position = a_position;
    //v_texcoord = a_texcoord;
    v_texcoord = a_position;

    // Final position
    gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
}
"""

fragment = """
uniform mat4      u_model;           // Model matrix
uniform mat4      u_view;            // View matrix
uniform mat4      u_normal;          // Normal matrix
uniform mat4      u_projection;      // Projection matrix
uniform samplerCube u_texture;         // Texture 
uniform vec3      u_light_position;  // Light position
uniform vec3      u_light_intensity; // Light intensity

varying vec3      v_normal;          // Interpolated normal (in)
varying vec3      v_position;        // Interpolated position (in)
varying vec3      v_texcoord;        // Interpolated fragment texture coordinates (in)
void main()
{
    // Calculate normal in world coordinates
    vec3 normal = normalize(u_normal * vec4(v_normal,1.0)).xyz;

    // Calculate the location of this fragment (pixel) in world coordinates
    vec3 position = vec3(u_view*u_model * vec4(v_position, 1));

    // Calculate the vector from this pixels surface to the light source
    vec3 surfaceToLight = u_light_position - position;

    // Calculate the cosine of the angle of incidence (brightness)
    float brightness = dot(normal, surfaceToLight) /
                      (length(surfaceToLight) * length(normal));
    brightness = max(min(brightness,1.0),0.0);

    // Calculate final color of the pixel, based on:
    // 1. The angle of incidence: brightness
    // 2. The color/intensities of the light: light.intensities
    // 3. The texture and texture coord: texture(tex, fragTexCoord)

    // Get texture color
    vec4 t_color = textureCube(u_texture, v_texcoord);

    // Final color
    //gl_FragColor = vec4(t_color.rgb * (0.1 + 0.9*brightness * u_light_intensity), t_color.a);
    gl_FragColor = t_color;
}
"""

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
    //v_texcoord = (u_texcoord * (vec4(a_texcoord - 0.5, 0.0, 1.0))).xy + 0.5;
    v_texcoord = a_texcoord;
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
    gl_FragColor = vec4(mix(t_color.rgb, t_color0.rgb, 0.2) * (alpha * intensity + (1.0 - alpha)), t_color.a);
    //gl_FragColor = t_color.rgba;
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

class CubemapCube(gloo.Program):
    def __init__(self, cubemap, vertex=vertex, fragment=fragment, light=(-2, 2, 2)):
        super().__init__(vertex, fragment)

        self._vertices, self._indices = geom_cube()
        self.bind(self._vertices)

        self["u_texture"] = cubemap

        self["u_model"] = self.model
        self["u_view"] = glm.translation(0, 0, -5)
        self["u_projection"] = np.eye(4, dtype=np.float32)

        self["u_light_position"] = light
        self["u_light_intensity"] = 1.0, 1.0, 1.0

    @property
    def model(self):
        return np.eye(4, dtype=np.float32)

    def render(self, model):
        self.draw(gl.GL_TRIANGLES, self._indices)

class Lines(gloo.Program):
    def __init__(self, count, vertex=VERTEX, fragment=FRAGMENT_COLOR):
        super().__init__(vertex, fragment, count=count)

        self["a_position"] = gloo.VertexBuffer(np.zeros((count, 3), dtype=np.float32))

    def render(self, points, model, view, projection, color=(1.0, 1.0, 1.0, 1.0)):
        if not isinstance(points[0], (tuple, list)):
            #print("stacking", points)
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

def get_face_transformation(face_index):
    # rotations required to get each face
    faces = [
        (90, 0, 1, 0), # pos x
        (-90, 0, 1, 0), # neg x
        (-90, 1, 0, 0), # pos y
        (90, 1, 0, 0), # neg y
        (0, 0, 0, 1), # pos z
        (180, 0, 1, 0), # neg z
    ]

    transform = np.eye(4, dtype=np.float32)
    glm.translate(transform, 0, 0, 1)
    glm.rotate(transform, *faces[face_index])
    return transform

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

class CubeFace(gloo.Program):
    def __init__(self, texture, face_index=0, texcoord=None, vertex=VERTEX, fragment=FRAGMENT):
        super().__init__(vertex, fragment, count=4)

        self.texture = texture
        self.face_index = face_index
        self.face_transform = get_face_transformation(face_index)

        position = np.array([(-1,-1, 0, 1), (-1,+1, 0, 1), (+1,-1, 0, 1), (+1,+1, 0, 1)], dtype=np.float32)
        if texcoord is None:
            texcoord = [(0, 1), (0, 0), (1, 1), (1, 0)]
        normal = np.array([0, 0, 1, 1], dtype=np.float32)
        texture_dir1 = np.array([0, 1, 0, 0], dtype=np.float32)
        texture_dir2 = np.array([1, 0, 0, 0], dtype=np.float32)

        self.position = np.dot(position, self.face_transform)[:, :3]
        self.normal = np.dot(normal, self.face_transform)[:3]
        self.texture_dir1 = np.dot(texture_dir1, self.face_transform)[:3]
        self.texture_dir2 = np.dot(texture_dir2, self.face_transform)[:3]

        self["a_position"] = self.position
        self["a_texcoord"] = texcoord
        self["a_normal"] = self.normal

        self["u_face_index"] = face_index

    def render(self, model, view, projection, normal_matrix=None, element_transform=None):
        # TODO this should be the other order
        old_model = model
        model = np.dot(element_transform, model)
        self["u_model"] = model
        self["u_view"] = view
        self["u_projection"] = projection
        self["u_normal"] = normal_matrix

        cube_transform = np.dot(element_transform, self.face_transform)
        cube_normal = np.dot(np.array([0, 0, 1, 1], dtype=np.float32), cube_transform)[:3]
        cube_texture_dir = np.dot(np.array([0, 1, 0, 0], dtype=np.float32), cube_transform)[:3]
        # target more like without face_transform?
        target = np.dot(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32), self.face_transform)[:3]

        faces = ["+x", "-x", "+y", "-y", "+z", "-z"]
        self.texture_transform = np.eye(4, dtype=np.float32)
        if abs(cube_normal[1]) < 0.001:
            #print("face %s side" % faces[self.face_index])
            target_texture_dir1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            #target = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            #print("face %d is probably side texture, local normal %s, cube normal %s" % (self.face_index, np.round(self.normal), np.round(cube_normal)))
            #print("target: %s" % target)
            angle = math.degrees(angle_between(target, cube_texture_dir))
            #print("angle: %s" % angle)
            #glm.rotate(self.texture_transform, -angle, 0, 0, 1)
        else:
            #print("face %s top" % faces[self.face_index])
            target_texture_dir1 = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        
        #target_texture_dir1 = np.dot(np.append(target_texture_dir1, [0]), np.array(np.matrix(self.face_transform).I))[:3].astype(np.float32)
        # should be now the texture dir we want (on default +z plane)
        angle = math.degrees(angle_between(target_texture_dir1, cube_texture_dir))
        glm.rotate(self.texture_transform, +angle, 0, 0, 1)
        #print("texture dir should've been", np.round(target_texture_dir1))
        #print("texture dir is", np.round(cube_texture_dir))
        #print("angle", angle)

        self["u_texcoord"] = self.texture_transform

        self["u_texture"] = self.texture
        self["u_light_direction"] = [-0.1, 1.0, 1.0]

        self.draw("triangle_strip")

        ##for vertex in self.position:
        #    draw_line(0.8 * vertex, 0.8 * vertex + 0.25 * self.normal, model, view, projection, color=(1.0, 1.0, 0.0, 1.0))

        #center = (np.sum(self.position, axis=0) / len(self.position))[:3]
        #center = center + self.normal * 0.01
        
        #draw_line(center, center + 1.0 * self.texture_dir1, model, view, projection, color=(1.0, 0.0, 1.0, 1.0))
        #draw_line(center, center + 0.25 * self.texture_dir2, model, view, projection, color=(1.0, 0.0, 1.0, 1.0))

        #center = center + self.normal * 0.02
        #draw_line(center, center + 1.0 * target_texture_dir1, old_model, view, projection, color=(0.0, 1.0, 1.0, 1.0))

        #draw_line((-1, -1, -1), (-1 + 0, -1 + 1, -1), old_model, view, projection, color=(1.0, 1.0, 0.0, 1.0))

class Cube:
    def __init__(self, sides):
        self.faces = []
        for i, side in enumerate(sides):
            texture = gloo.Texture2D(data=side[0])
            uv0, uv1 = side[1]
            texcoord = np.array([(uv0[0], uv1[1]), uv0, uv1, (uv1[0], uv0[1])], dtype=np.float32)
            self.faces.append(CubeFace(texture, face_index=i, texcoord=texcoord))

    def render(self, model, view, projection):
        assert False
        normal_matrix = np.array(np.matrix(np.dot(view, model)).I.T)
        for face in self.faces:
            face.render(model, view, projection, normal_matrix=normal_matrix)

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
        [2, 3, 0, 1],
        [7, 6, 5, 4],
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

    @property
    def model(self):
        model = np.eye(4, dtype=np.float32)

        #if self.elementdef.get("__comment", "") != "Obsidian base":
        #    return model

        # apply from/to attributes
        scale = (self.xyz1 - self.xyz0) * 0.5
        scalem = np.eye(4, dtype=np.float32)
        #glm.scale(scalem, scale[0], scale[1], scale[2])
        #glm.scale(model, scale[0], scale[1], scale[2])
        #glm.scale(scalem, 1.0, 0.5, 1.0)

        #if self.elementdef.get("__comment", "") == "Obsidian base":
        #    glm.scale(scalem, 1.0, 0.2, 1.0)

        rotationm = np.eye(4, dtype=np.float32)
        # apply rotation attributes
        if self.rotation:
            axis = {"x" : [1, 0, 0],
                    "y" : [0, 1, 0],
                    "z" : [0, 0, 1]}[self.rotation["axis"]]
            origin = (np.array(self.rotation.get("origin", [8, 8, 8]), dtype=np.float32) - 8.0) / 16.0 * 2.0
            origin /= scale
            origin *= -1.0
            glm.translate(rotationm, *origin)
            glm.rotate(rotationm, self.rotation["angle"], *axis)
            origin *= -1.0
            glm.translate(rotationm, *origin)

        translate = ((self.xyz0 + self.xyz1) / 2.0) / scale
        translatem = np.eye(4, dtype=np.float32)
        glm.translate(translatem, translate[0], translate[1], translate[2])
        #print("scale: ", scale, "translate:", translate)

        print(self.elementdef)
        print("scale:", scale)
        print("translate:", translate)
        print("translate actually:", ((self.xyz0 + self.xyz1) / 2.0))
        print("---")

        return np.dot(rotationm, np.dot(translatem, scalem))
        return model

    def render_face(self, face_index, texture, uvs, model, view, projection):
        program = self.program

        points = self.points[Element.CUBE_FACES[face_index]].astype(np.float32)
        program["a_position"].set_data(points)
        normals = np.array(4 * [Element.CUBE_NORMALS[face_index]], dtype=np.float32)
        program["a_normal"].set_data(normals)

        uv0, uv1 = uvs
        scale = (uv1 - uv0) * 0.5
        translate = (uv1 + uv0) * 0.5
        program["a_texcoord"] = np.array(Element.CUBE_TEXCOORDS, dtype=np.float32) * scale + translate

        program["u_texture"] = texture
        program["u_light_direction"] = [-0.1, 1.0, 1.0]

        program.draw("triangles", self.indices)

    def render(self, model, view, projection, element_transform=np.eye(4, dtype=np.float32)):
        #super().render(np.dot(self.model, model), view, projection)

        #element_transform = np.dot(element_transform, self.model)
        #element_transform = self.model

        #model_view = np.dot(view, np.dot(model, element_transform))
        #normal_matrix = np.array(np.matrix(model_view).I.T)
        #for face in self.faces:
        #    face.render(model, view, projection, normal_matrix=normal_matrix, element_transform=element_transform)

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

        complete_model = np.dot(np.dot(rotation, element_transform), model)
        program["u_model"] = complete_model
        program["u_view"] = view
        program["u_projection"] = projection
        program["u_normal"] = np.array(np.matrix(np.dot(view, complete_model)).I.T)

        for i, (texture, uvs) in enumerate(self.faces):
            if texture is None:
                continue
            self.render_face(i, texture, uvs, model, view, projection)

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
        sides = [ faces.get(direction, None) for direction in mc_to_opengl ]
        # get first side that is not empty
        #empty = (np.zeros((1, 1, 4), dtype=np.uint8), ((0, 0), (16, 16)))
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
        #model = np.dot(m, model)

        if "uvlock" in modelref:
            # TODO
            pass

        for element in self.elements:
            element.render(model, view, projection, element_transform=m)

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

