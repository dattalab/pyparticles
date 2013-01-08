"""
We'll be using offscreen rendering extensively here. 
The concepts involved are framebuffer objects, render-to-texture,
sampler2D, fragment shaders and ping-pong (reduction) shading.

The approach is this:
- Create a framebuffer object (FBO), bind it. 
The framebuffer is an abstract "destination" of sorts for OpenGL. 
It is where pixels will be stored after rendering.
An FBO needs to have attached to itself a concrete destination for the pixels. 
Here, we'll be attaching a texture which will receive the pixels. 
- Render the mice into that texture

- Create a viewport that has as many pixels as there are mice
- For each coordinate, calculate the sum in that arena

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
    def __init__(self, width, height, viewport_width, viewport_height):
        super(TextureTest, self).__init__()

        def ispow2(num):
            return num != 0 and ((num & (num - 1)) == 0)

        assert ispow2(width), "Width must be power of two"
        assert ispow2(height), "Height must be power of two"
        assert ispow2(viewport_width), "Viewport height must be power of two"
        assert ispow2(viewport_height), "Viewport height must be power of two"

        assert width == height, "For now, width and height must be the same"
        assert viewport_width == viewport_height, "Viewport width and height must be the same"

        self.width = width
        self.height = height
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        
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

        # Step 1
        # Draw textures

        # Bind the framebuffer (we'll draw into that, as opposed to the render buffer)
        glBindFramebuffer(GL_FRAMEBUFFER, self.frameBuffer)
        print "Drawing into texture %d" % 0
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.textures[0], 0)
        glViewport(0,0,self.width, self.height)
        # Get setup to draw
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, 1, 0, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glClearColor(0.5, 0.5, 0.5, 1.0)        
        glDepthFunc(GL_LEQUAL)

        glUseProgram(self.renderShaderProgram)

        # Okay, draw some very simple depth into a texture
        glBegin(GL_QUADS)

        # One big quad
        glColor4f(0.0, 1.0, 0.0, 1.0)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(0, 0, 0.0)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(0, 1, 0.0)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(1, 1, 0.0)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(1, 0, 0.0)

        new_width = 0.5 # self.viewport_width / self.width
        glColor4f(1.0, 0.0, 0.0, 1.0)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(new_width, new_width, 1.0)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(new_width, 2*new_width, 1.0)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(new_width*2, new_width*2,1.0)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(new_width*2, new_width, 1.0)

        glEnd()

        glUseProgram(0)


        self.orig_data = glReadPixels(0,0,self.width,self.height, GL_DEPTH_COMPONENT, GL_FLOAT)

        
        
        
        # PING-PONG REDUCTION
        this_width = int(self.width)
        this_height = int(self.height)
        for i in range(int(np.log2(self.width/self.viewport_width))):
            source_tex_id = mod(i,2)
            dest_tex_id = 1 - source_tex_id

            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.textures[dest_tex_id], 0)

            print "Reducing %d^2 (tex:%d) to %d^2 (tex:%d)" % (this_width, source_tex_id, this_width/2, dest_tex_id)
            glViewport(0,0, int(this_width/2), int(this_height/2))
            # Get setup to draw
            glClear(GL_DEPTH_BUFFER_BIT)

            # Bind the shader
            glUseProgram(self.reductionShaderProgram)

            # Texture stuff!
            data_size_loc = glGetUniformLocation(self.reductionShaderProgram, "data_size")
            glUniform2f(data_size_loc, this_width, this_height)
            viewport_size_loc = glGetUniformLocation(self.reductionShaderProgram, "viewport_size")
            glUniform2f(viewport_size_loc, int(this_width/2), int(this_height/2))

            # This is the texture that contains data to be reduced
            data_texture_loc = glGetUniformLocation(self.reductionShaderProgram, "data_texture")
            glActiveTexture(GL_TEXTURE0+1)
            glBindTexture(GL_TEXTURE_2D, self.textures[source_tex_id])
            glUniform1i(data_texture_loc, 1)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, 0)

            # Just draw a square
            glBegin(GL_QUADS)
            glColor4f(0.0, 1.0, 0.0, 1.0)
            glTexCoord2f(0.0, 0.0)
            glVertex3f(0, 0, 0.0)
            glTexCoord2f(0.0, 1.0)
            glVertex3f(0, 1, 0.0)
            glTexCoord2f(1.0, 1.0)
            glVertex3f(1, 1, 0.0)
            glTexCoord2f(1.0, 0.0)
            glVertex3f(1, 0, 0.0)
            glEnd()

            glUseProgram(0)

            this_width /= 2
            this_height /= 2

        # Okay, we've drawn a great square into a texture, 
        # and its depth values have been handled by the shader
        # Now, it's time to try some reductions. 

        # Read off the pixels
        print "Final reduction to %d^2" % this_width
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.textures[source_tex_id], 0)
        self.reduced_data = glReadPixels(0,0,this_width,this_height, GL_DEPTH_COMPONENT, GL_FLOAT)

        # Clean up after ourselves
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0)


    def create_texture(self, width, height):
        
        t = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, t)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
            width, height, 0,
            GL_DEPTH_COMPONENT, GL_FLOAT, None
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE)
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

        # Fill the second texture with some interesting data
        y,x = np.mgrid[0:self.height, 0:self.width]
        self.some_data = 0.5*(1.+np.sin(2.*np.pi*y/self.height))* np.sin(np.pi*x/self.width)
        glBindTexture(GL_TEXTURE_2D, self.textures[1])
        # glPixelStoref(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
            self.width, self.height, 0,
            GL_DEPTH_COMPONENT, GL_FLOAT, self.some_data
        )


    def setup_reduction_shader(self):

        vShader = shaders.compileShader("""
            #version 120
            void main() {
                gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
            }
            """, GL_VERTEX_SHADER)

        fShader = shaders.compileShader("""
            #version 120
            uniform sampler2D data_texture;
            uniform vec2 viewport_size;
            uniform vec2 data_size;
            
            void main() {

                // First, figure out the range of values we'll need to snag
                vec2 scale = data_size / viewport_size;
                vec2 ll = (floor(gl_FragCoord.xy)) * scale;
                vec2 ur = (floor(gl_FragCoord.xy)+1) * scale;

                ivec2 ll_pixel = ivec2(ll);
                ivec2 ur_pixel = ivec2(ur);
                
                int width = int(scale.x);
                int height = int(scale.y);

                float num_pixels = width*height;
                float target_depth = 0.0;

                for (int i=ll_pixel.x; i < ur_pixel.x; ++i) {
                    float x = i/data_size.x;
                    for (int j=ll_pixel.y; j < ur_pixel.y; ++j) {
                        float y = j/data_size.y;
                        float this_depth = texture2D(data_texture, vec2(x,y)).r;
                        target_depth += this_depth/num_pixels;
                    }
                }
                gl_FragDepth = target_depth;

            }
            """, GL_FRAGMENT_SHADER)

        self.reductionShaderProgram = shaders.compileProgram(vShader, fShader)


    def setup_render_shader(self):
        vShader = shaders.compileShader("""
            #version 120
            void main() {
                gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
            }
            """, GL_VERTEX_SHADER)

        fShader = shaders.compileShader("""
            #version 120
            void main() {
                gl_FragDepth = gl_FragCoord.z;
            }
            """, GL_FRAGMENT_SHADER)

        self.renderShaderProgram = shaders.compileProgram(vShader, fShader)



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
        self.setup_reduction_shader()
        self.setup_render_shader()

if __name__ == "__main__":
    a = 32*16
    b = 32
    t = TextureTest(a,a,b,b)
    t.gl_init()
    t.display()
    figure(figsize=(12,6)); 
    subplot(1,2,1)
    title("Original data")
    # t.orig_data = (1.0 - t.orig_data)
    imshow(t.orig_data, origin='lowerleft')
    # xticks(range(a))
    # yticks(range(a))
    colorbar()
    subplot(1,2,2)
    title("Reduced data")
    scale = (t.width/t.viewport_width)*(t.height/t.viewport_height)
    # print scale
    imshow(t.reduced_data, origin='lowerleft')
    colorbar()
    # xticks(range(b))
    # yticks(range(b))

    # glutMainLoop()