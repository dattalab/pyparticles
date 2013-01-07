from __future__ import division

import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GLUT.freeglut import *
from OpenGL.GL import shaders
from OpenGL.arrays import vbo
from OpenGL.GL.ARB.draw_instanced import *
from OpenGL.GL.ARB.texture_buffer_object import *
from OpenGL.GL.framebufferobjects import *
from OpenGL.GL.ARB.depth_texture import *

# Get a texture up on the screen
class TextureTest(object):
    """docstring for TextureTest"""
    def __init__(self, width, height):
        super(TextureTest, self).__init__()
        self.width = width
        self.height = height
        
        self.lastkey = ''
        self.lastx = self.width/2.
        self.lasty = self.height/2.

    def on_keypress(self, key, x, y):
        self.lastkey = key

        if key == 'x':
            sys.exit(1)

    def on_motion(self, x, y):
        # store the mouse coordinate
        self.lastx = float(x)
        self.lasty = float(y)

    def display(self):

        # Bind the framebuffer (we'll draw into that, as opposed to the render buffer)
        glBindFramebuffer(GL_FRAMEBUFFER, self.outputTexture)

        # Get setup to draw
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Bind the shader
        # glUseProgram(self.shaderProgram)


        # Texture stuff!
        glEnable(GL_TEXTURE_2D)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
        glBindTexture(GL_TEXTURE_2D, self.textures[0])
        glUniform2f(self.mouse_img_size_location, self.mouse_width, self.mouse_height)

        # Just draw a square
        glBegin(GL_QUADS)
        glColor4f(0.0, 1.0, 0.0, 1.0)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(-1, -1, -0.1)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(-1, 1, -0.1)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(1, 1, -0.1)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(1, -1, -0.1)
        glEnd()

        # Read off the pixels

        # Clean up after ourselves
        # glUseProgram(0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)



    def setup_fbo(self):
        """
        We need to setup the FBO to be bound to a texture, that we can render in, and
        subsequently read from
        """
        self.frameBuffer = glGenFrameBuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.frameBuffer)

        # Setup two textures, which we can render between
        self.textures = []
        for i in range(2):
            t = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, t)
            glTexImage2D(GL_TEXTURE_2D, 0, 
                    GL_DEPTH_COMPONENT, 
                    self.width, self.height, 
                    0, GL_DEPTH_COMPONENT, GL_FLOAT, None)

            self.textures.append(t)
        









