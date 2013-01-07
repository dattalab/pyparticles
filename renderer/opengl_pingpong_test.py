"""
We'll be using offscreen rendering extensively here. 
The concepts involved are framebuffer objects, render-to-texture,
sampler2D, fragment shaders and ping-pong (reduction) shading.

The approach is this:
Create a framebuffer object (FBO), bind it. 
The framebuffer is an abstract "destination" of sorts for OpenGL. 
It is where pixels will be stored after rendering.
An FBO needs within itself a concrete destination for the pixels. 
Here, we'll be attaching a texture to the framebuffer which will
receive the pixels. 
Render the mice
At this point, we go to town reducing the differences.
"""

from __future__ import division

import numpy as np
from pylab import *

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
from OpenGL.GL.ARB.shadow import *

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
        # glBindFramebuffer(GL_FRAMEBUFFER, self.outputTexture)

        # Get setup to draw
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glClearColor(0.5, 0.5, 0.5, 1.0)        
        glDepthFunc(GL_LEQUAL)

        # Bind the shader
        # glUseProgram(self.shaderProgram)


        # Texture stuff!
        glBindTexture(GL_TEXTURE_2D, self.textures[0])
        glBindFramebuffer(GL_FRAMEBUFFER, self.frameBuffer)

        # Just draw a square
        glBegin(GL_QUADS)
        # One big quad
        glColor4f(0.0, 1.0, 0.0, 1.0)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(-1, -1, -0.1)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(-1, 1, -0.1)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(1, 1, -0.1)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(1, -1, -0.1)

        # A quarter-sized quad
        glColor4f(1.0, 0.0, 0.0, 1.0)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(-1, -1, 0.5)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(-1, 0, 0.5)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(0, 0, 0.5)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(0, -1, 0.5)

        glEnd()

        # Read off the pixels
        self.data = glReadPixels(0,0,self.width,self.height, GL_DEPTH_COMPONENT, GL_FLOAT)

        # Clean up after ourselves
        # glUseProgram(0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)


    def create_texture(self, width, height):
        
        t = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, t)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
            width, height, 0,
            GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, None
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0)


        return t

    def setup_fbo(self):
        """
        We need to setup the FBO to be bound to a texture, that we can render in, and
        subsequently read from
        """
        self.setup_textures()
        self.frameBuffer = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.frameBuffer)

        # Attach our first texture to the depth attachment point
        glBindTexture(GL_TEXTURE_2D, self.textures[0])
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.textures[0], 0)

        # Every FBO has to have a color attachment. We don't persist it
        # because we won't be using it. We only care about depth.
        color = glGenRenderbuffers(1)
        glBindRenderbuffer( GL_RENDERBUFFER, color )
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, self.width, self.height)
        glFramebufferRenderbuffer( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color )
        glBindRenderbuffer(GL_RENDERBUFFER, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def setup_textures(self):
        # Setup two textures, which we can render between
        self.textures = []
        for i in range(2):
            self.textures.append(self.create_texture(self.width, self.height))


    def gl_init(self):
        glutInit([])
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
        glutInitWindowSize(self.width, self.height)
        glutCreateWindow('Texture Rendering Test')

        glutKeyboardFunc(self.on_keypress)
        glutMotionFunc(self.on_motion)
        glutDisplayFunc(self.display)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_NORMALIZE)
        glEnable(GL_TEXTURE_2D)

        self.setup_fbo()



if __name__ == "__main__":
    t = TextureTest(300, 300)
    t.gl_init()
    t.display()
    figure(); imshow(t.data)
    # glutMainLoop()