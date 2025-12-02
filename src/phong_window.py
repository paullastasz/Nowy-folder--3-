import os.path

import moderngl
import numpy as np
from PIL import Image
from pyrr import Matrix44

from base_window import BaseWindow

import sys
import json
import random


class PhongWindow(BaseWindow):

    def __init__(self, **kwargs):
        super(PhongWindow, self).__init__(**kwargs)
        self.frame = 0

    def init_shaders_variables(self):
        self.model_view_projection = self.program["model_view_projection"]
        self.model_matrix = self.program["model_matrix"]
        self.material_diffuse = self.program["material_diffuse"]
        self.material_shininess = self.program["material_shininess"]
        self.light_position = self.program["light_position"]
        self.camera_position = self.program["camera_position"]

    def on_render(self, time: float, frame_time: float):
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        camera_position = np.array([5.0, 5.0, 15.0], dtype='float32')

        valid_pos = False
        model_translation = None
        while not valid_pos:
            model_translation = np.random.uniform(-4.0, 4.0, 3).astype('float32')
            
          
            #large
            #dopasowanie pod large
            model_translation[0] = np.random.uniform(2.0, 5.0) # X
            model_translation[1] = np.random.uniform(2.0, 5.0) # Y
            
            
            model_translation[2] = np.random.uniform(9.0, 12.5)
            # 2.0 dla zwyklego i easy, medium to 4
            if np.linalg.norm(camera_position - model_translation) > 2.0:
                valid_pos = True

        #  <0, 255> -> normalizacja dla renderera
        material_diffuse = np.random.uniform(0.0, 1.0, 3).astype('float32')

        # połyskliwości [3, 20]
        material_shininess = random.uniform(3.0, 20.0)

        # pozycji światła <-20, 20>
        light_position = np.random.uniform(-20, 20, 3).astype('float32')

        model_matrix = Matrix44.from_translation(model_translation)
        proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)
        lookat = Matrix44.look_at(
            camera_position,
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )

        model_view_projection = proj * lookat * model_matrix

        self.model_view_projection.write(model_view_projection.astype('f4').tobytes())
        self.model_matrix.write(model_matrix.astype('f4').tobytes())
        self.material_diffuse.write(np.array(material_diffuse, dtype='f4').tobytes())
        self.material_shininess.write(np.array([material_shininess], dtype='f4').tobytes())
        self.light_position.write(np.array(light_position, dtype='f4').tobytes())
        self.camera_position.write(np.array(camera_position, dtype='f4').tobytes())

        self.vao.render()

        if self.output_path:
            img_filename = f'image_{self.frame:04d}.png'
            
            img = (
                Image.frombuffer('RGBA', self.wnd.size, self.wnd.fbo.read(components=4))
                .transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            )
            img.save(os.path.join(self.output_path, img_filename))

            # Relatywne pozycje dla sieci neuronowej - zapisanie do JSON tu ze wzgledu na optymalizacje czasu treningu
            rel_light = light_position - model_translation
            rel_camera = camera_position - model_translation

            data = {
                "file_name": img_filename,
                "object_position": model_translation.tolist(),
                "light_position": light_position.tolist(),
                "camera_position": camera_position.tolist(),
                "material_diffuse": material_diffuse.tolist(),
                "material_shininess": float(material_shininess),
                "relative_light_vector": rel_light.tolist(),
                "relative_view_vector": rel_camera.tolist()
            }

            json_filename = f'data_{self.frame:04d}.json'
            with open(os.path.join(self.output_path, json_filename), 'w') as f:
                json.dump(data, f, indent=4)

            if self.frame % 100 == 0:
                print(f"Progress: {self.frame}/3000")

            self.frame += 1

        if self.frame >= 3000:
            print("Generowanie zakończone. Utworzono 3000 par plików.")
            sys.exit(0)
