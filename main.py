import torch
import torch.nn as nn
import neural_rendering
import argparse
import numpy as np
import aux_info
import visualizer
import tkinter as tk
from PIL import ImageTk, Image
import time
import pygame
import math
import utils
import matplotlib.pyplot as plt
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import cv2
from neural_rendering import NeuralMaterialSavable
from neural_rendering import NeuralMaterialLive


class FullyConnected1(torch.nn.Module):
    def __init__(self, num_in, num_out=3):
        super(FullyConnected1, self).__init__()

        self.num_in = num_in
        self.num_out = num_out

        self.func = torch.nn.Sequential(
            torch.nn.Conv2d(num_in, 25, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(25, 25, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(25, 25, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(25, self.num_out, 1),
        )

    def forward(self, x):
        return self.func(x)


if torch.cuda.is_available():
    print("CUDA is available!")
    print("GPU device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA is not avail able.")


args = argparse.Namespace(inm=True, vis=True)
args.outm = "output.pth"
args.stats = False
args.debug = False
args.export = None
args.use_offset = True
args.max_iter = 30000
args.batch = 4
args.t_pg = False
args.dataset = "cliff.hdf5" ###
args.loss = "comb1"
args.levels = None
args.boost = 1.0
args.cosine_mult = False
args.workers = 8
args.experiment = "StandardRawLongShadowMaskOnly"
# temp = neural_rendering.NeuralMaterialSavable(neural_rendering.NeuralMaterialLive(args))
args.vis = True
neural_material = neural_rendering.NeuralMaterialLive(args)
# neural_material.vars.resolution = 1000
# neural_material.vars.number_mip_maps_levels = 1
# neural_material.vars.iter_count = 0

# visualizer.main(neural_material.vars)




def main(material, light_dir_x, light_dir_y):
    light_dir = np.array([light_dir_x, light_dir_y])
    camera_dir = np.array([0., 0.])
    input_info = aux_info.InputInfo()
    input_info.light_dir_x = light_dir[0]
    input_info.light_dir_y = light_dir[1]

    location = material.vars.generate_locations()
    print(location, location.shape)
    input_info.camera_dir_x = camera_dir[0]
    input_info.camera_dir_y = camera_dir[1]
    input_info.mipmap_level_id = 0.0
    mimpap_type = 0
    input, mipmap_level_id = material.vars.generate_input(input_info, use_repeat=True)
    # print(input, input.shape)
    result, eval_output = material.vars.evaluate(input, location, level_id=mipmap_level_id, mimpap_type=mimpap_type,
                                              camera_dir=list(camera_dir))
    zero_ch = result.shape[1]
    result = result.repeat([1, 3, 1, 1])
    result = result[:, :3, :, :]
    result[:, zero_ch:, :, :] = 0
    result = result.data.cpu().numpy()[0, :, :, :].transpose([1, 2, 0])
    print("RESULT", result, result.shape)  ###
    import matplotlib.pyplot as plt

    new_view = result
    new_view = utils.tensor.to_output_format(new_view)
    # new_view = new_view * 255
    new_view = np.repeat(new_view, 1, axis=0)
    new_view = np.repeat(new_view, 1, axis=1)

    # plt.imshow(new_view)
    # plt.show()
    return new_view


# def update():
#     global image_data, a, b
#     a = float(entry_a.get())
#     b = float(entry_b.get())
#     start = time.time()
#     image_data = main(neural_material, a, b)
#     end = time.time()
#     image = Image.fromarray((image_data * 255).astype(np.uint8))
#     photo = ImageTk.PhotoImage(image)
#     label.configure(image=photo)
#     label.image = photo
#     execution = end - start
#     # print(start, end, execution)
#     seconds.set("{:.2f}".format(execution))
#
# window = tk.Tk()
# a = 0
# b = 0
# seconds = tk.StringVar()
# start = time.time()
# image_data = main(neural_material, a, b)
# end = time.time()
# execution = end - start
# seconds.set("{:.2f}".format(execution))
#
# image = Image.fromarray((image_data * 255).astype(np.uint8))
# photo = ImageTk.PhotoImage(image)
# label = tk.Label(window, image=photo)
# label.pack()
#
# label_a = tk.Label(window, text="Light Direction x:")
# label_a.pack()
# entry_a = tk.Entry(window)
# entry_a.insert(0, "0")
# entry_a.pack()
#
# label_b = tk.Label(window, text="Light Direction y:")
# label_b.pack()
# entry_b = tk.Entry(window)
# entry_b.insert(0, "0")
# entry_b.pack()
#
# ###
# label_s = tk.Label(window, text="Time (in seconds):")
# label_s.pack()
# label_s1 = tk.Label(window, textvariable=seconds)
# label_s1.pack()
#
#
# button = tk.Button(window, text="Update", command=update)
# button.pack()
#
# window.mainloop()

# import pygame
# import numpy as np
# from pygame.locals import *
# from OpenGL.GL import *
# from OpenGL.GLU import *
#
# pygame.init()
# display = (800, 600)
# pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
# gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
# glTranslatef(0.0, 0.0, -3) #
#
# # glEnable(GL_LIGHTING)
# # glEnable(GL_LIGHT0)
# # glLightfv(GL_LIGHT0, GL_POSITION, [0, -1, 1, 0])
# # glLightfv(GL_LIGHT0, GL_DIFFUSE, [1, 1, 1, 1])
#
# img_array = main(neural_material,-1,0)
# img_array = np.array(img_array * 255, dtype=np.uint8)
# # print(img_array)
#
# image_surface = pygame.image.fromstring(img_array.tobytes(), (512, 512), "RGB")
#
# textureData = pygame.image.tostring(image_surface, "RGB", 1)
# width = image_surface.get_width()
# height = image_surface.get_height()
#
# texid = glGenTextures(1)
# glBindTexture(GL_TEXTURE_2D, texid)
# glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, textureData )
# glGenerateMipmap(GL_TEXTURE_2D)
#
# while True:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#             quit()
#
#     glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
#
#     glBindTexture(GL_TEXTURE_2D, texid)
#     glEnable(GL_TEXTURE_2D)
#     qobj = gluNewQuadric()
#     gluQuadricTexture(qobj, GL_TRUE)
#     gluSphere(qobj, 1, 50, 50)  #
#     gluDeleteQuadric(qobj)
#     glDisable(GL_TEXTURE_2D)
#
#     pygame.display.flip()
#     pygame.time.wait(10)

import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

def to_local(wi, wo, n):
    x = np.cross(n, [1, 0, 0]) if np.abs(n[0]) < 0.9 else np.cross(n, [0, 1, 0]) #x axis
    x = x / np.linalg.norm(x) if np.linalg.norm(x) != 0 else x # normalize
    y = np.cross(n, x) #y axis
    transform = np.array([x, y, n])
    local_wi = np.dot(wi, transform)
    local_wo = np.dot(wo, transform)
    return local_wi[:2], local_wo[:2] #return 2d

class Camera:
    def __init__(self, f, c, R, t):
        self.f = f #focal length float
        self.c = c #offset of principle point 2*1
        self.R = R #camera position 3*3
        self.t = t #camera translation 3*1

    def project(self, pts3):
        pts3_cam = self.R.T @ (pts3.reshape(-1, 1) - self.t.reshape(-1, 1))
        pts2_cam = self.f * pts3_cam[:2, :] / pts3_cam[2, :]
        pts2 = pts2_cam + self.c.reshape(-1, 1)
        return pts2

num_points = 512
u = np.linspace(0, 2 * np.pi, num_points)
v = np.linspace(0, np.pi, num_points)
u, v = np.meshgrid(u, v)

x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)
positions = np.stack([x, y, z], axis=-1)
normals = positions

# print(positions[0][0])

light_position = np.array([-2.0, 0.0, -3.0])  #?
light_directions = light_position - positions
light_directions = light_directions / np.linalg.norm(light_directions, axis=-1, keepdims=True)

#set fov = 45. width/height = 800,600. translation = -3
fov = 45
width = 800
height = 600
camera_position = np.array([0, 0, -3])
f = 1 / math.tan(math.radians(fov) / 2)
c = np.array([width / 2, height / 2])
R = np.eye(3)
t = -camera_position
camera = Camera(f, c, R, t)

#only for initialization
location = neural_material.vars.generate_locations()
input_info = aux_info.InputInfo()
input_info.light_dir_x = 0
input_info.light_dir_y = 0
input_info.camera_dir_x = 0
input_info.camera_dir_y = 0
input_info.mipmap_level_id = 0.0
input_info.camera_dir_x = 0
input_info.camera_dir_y = 0
input, mipmap_level_id = neural_material.vars.generate_input(input_info, use_repeat=True)

for i in range(512):
    for j in range(512):
        pts2 = camera.project(positions[i][j])
        # location shape 1,2,512,512
        location[0][0][i][j] = pts2[0][0]
        location[0][1][i][j] = pts2[1][0] #TODO: improve here
        view_direction = np.array([0,0,-1])
        localwi, localwo = to_local(light_directions[i][j], view_direction, normals[i][j])
        #input shape 1,4,512,512. first two wo, last two wi
        input[0][0][i][j] = localwo[0]
        input[0][1][i][j] = localwo[1]
        input[0][2][i][j] = localwi[0]
        input[0][3][i][j] = localwi[1] #TODO: improve here

#
result, eval_output = neural_material.vars.evaluate(input, location, level_id=mipmap_level_id, mimpap_type=0,
                                              camera_dir=list(np.array([0,0])))
zero_ch = result.shape[1]
result = result.repeat([1, 3, 1, 1])
result = result[:, :3, :, :]
result[:, zero_ch:, :, :] = 0
result = result.data.cpu().numpy()[0, :, :, :].transpose([1, 2, 0])
print("RESULT", result, result.shape)  ###

new_view = result
new_view = utils.tensor.to_output_format(new_view)
new_view = np.repeat(new_view, 1, axis=0)
new_view = np.repeat(new_view, 1, axis=1)
plt.imshow(new_view)
plt.show()

pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
glTranslatef(0.0, 0.0, -3) #

glEnable(GL_LIGHTING)
glEnable(GL_LIGHT0)
glLightfv(GL_LIGHT0, GL_POSITION, [-2, 0, -3, 1])
glLightfv(GL_LIGHT0, GL_DIFFUSE, [1, 1, 1, 1])
glLightfv(GL_LIGHT0, GL_AMBIENT, [2, 2, 2, 1])


img_array = new_view
img_array = np.array(img_array * 255, dtype=np.uint8)
# print(img_array)

image_surface = pygame.image.fromstring(img_array.tobytes(), (512, 512), "RGB")

textureData = pygame.image.tostring(image_surface, "RGB", 1)
width = image_surface.get_width()
height = image_surface.get_height()

texid = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, texid)
glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, textureData )
glGenerateMipmap(GL_TEXTURE_2D)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    glBindTexture(GL_TEXTURE_2D, texid)
    glEnable(GL_TEXTURE_2D)
    qobj = gluNewQuadric()
    gluQuadricTexture(qobj, GL_TRUE)
    gluSphere(qobj, 1, 50, 50)  #
    gluDeleteQuadric(qobj)
    glDisable(GL_TEXTURE_2D)

    pygame.display.flip()
    pygame.time.wait(10)
