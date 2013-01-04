import numpy as np
import ctypes as c

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GLUT.freeglut import *
from OpenGL.GL import shaders
from OpenGL.arrays import vbo
from OpenGL.GL.ARB.draw_instanced import *
from OpenGL.GL.ARB.texture_buffer_object import *
from OpenGL.GL.framebufferobjects import *

from OpenGL.GL.ARB.transform_feedback2 import *
from OpenGL.GL.EXT.transform_feedback import *
from OpenGL.GL.NV.transform_feedback import *
from OpenGL.raw.GL.NV.geometry_program4 import *
from OpenGL.raw.GL import *
from mako.template import Template

program = None
program2 = None
transformBuffer, feedbackBuffer = None, None
t = 0.0
query = None

import time
lasttime = time.time()
avgrate = 0
iframe = 0
color_loc = 0

num_joints = 9
uploadUniforms = True

def init():
    glClearColor( 0.5,0.5,0.5, 1 );
    
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glOrtho( -1,1, -1,1, 0.2,2 );
    glMatrixMode( GL_MODELVIEW );

    setup_transformfeedbackbuffer()

    setup_shaders()

    configure_transformfeedbackbuffer()

    configure_shaders()

    global query
    query = glGenQueries(1);

def setup_transformfeedbackbuffer():
    global transformBuffer, feedbackBuffer, program

    data = np.array([
        [-0.5,  -0.5,  -0.3, 1 ],
        [-0.6,  -0.45, -0.3, 1 ],
        [-0.45, -0.47, -0.3, 1 ],
        [0.5,  0.5,  -0.3, 1 ],
        [0.6,  0.45, -0.3, 1 ],
        [0.45, 0.47, -0.3, 1 ]
    ], dtype='float32')
    data = np.hstack((data, np.zeros((6,num_joints*3)))).astype('float32')

    transformBuffer = vbo.VBO(data, 
                            usage="GL_DYNAMIC_DRAW", 
                            target="GL_ARRAY_BUFFER")
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, transformBuffer)

    feedback_data = np.zeros((6, 16+4), dtype='float32')
    feedbackBuffer = vbo.VBO(feedback_data, 
                            usage="GL_DYNAMIC_DRAW", 
                            target="GL_ARRAY_BUFFER")
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, feedbackBuffer)


def setup_shaders():
    global program, program2

    vs = glCreateShader(GL_VERTEX_SHADER)
    vs_source = """
    % for i in range(num_joints):
    attribute vec3 rotation${i};
    % endfor
    varying mat4 posish;
    varying vec4 a_color;
    uniform vec3 translations[${num_joints}];
    uniform mat4 bindingMatrixInverse[${num_joints}];

    void main()
    {   

        gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
        gl_Position = gl_Position*bindingMatrixInverse[0];
        posish[0] = gl_Position.xyzw;
        a_color = vec4(1.0, 0.0, 0.0, 1.0);
    }
    """
    makoTemplate = Template(vs_source)
    vs_source = makoTemplate.render(num_joints=num_joints)
    glShaderSource(vs, vs_source)
    glCompileShader(vs)

    result = glGetShaderiv( vs, GL_COMPILE_STATUS )
    if not(result):
        # TODO: this will be wrong if the user has
        # disabled traditional unpacking array support.
        raise RuntimeError(
            """Shader compile failure (%s): %s"""%(
                result,
                glGetShaderInfoLog( vs ),
            ),
            vs_source,
            GL_VERTEX_SHADER,
        )

    
    program = glCreateProgram()
    glAttachShader(program, vs)
    glDeleteShader(vs)

    vs_render = shaders.compileShader("""
    attribute vec4 incolor;
    varying vec4 some_color;
    void main()
    {
        gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
        some_color = incolor.xyzw;
    }
    """, GL_VERTEX_SHADER)
    fs_render = shaders.compileShader("""
    varying vec4 some_color;
    void main()
    {
        gl_FragColor = some_color;
    }
    """, GL_FRAGMENT_SHADER)
    program2 = shaders.compileProgram(vs_render, fs_render)
    
    global color_loc
    color_loc = glGetAttribLocation(program2, "incolor")
    
def configure_shaders():

    import ctypes as c
    global program
    glLinkProgram(program)
    glValidateProgram(program)
    validation = glGetProgramiv(program, GL_VALIDATE_STATUS)
    if validation == GL_FALSE:
        raise RuntimeError(
            """Validation failure (%s): %s"""%(
            validation,
            glGetProgramInfoLog( program ),
        ))
    link_status = glGetProgramiv( program, GL_LINK_STATUS )
    if link_status == GL_FALSE:
        raise RuntimeError(
            """Link failure (%s): %s"""%(
            link_status,
            glGetProgramInfoLog( program ),
        ))

    # l = c.c_int(0)
    # the_length = c.pointer(l)
    # s = c.c_int(0)
    # the_size = c.pointer(s)
    # t = c.c_int(0)
    # the_type = c.pointer(t)
    # n = c.create_string_buffer(30)
    # the_name = n
    # glGetTransformFeedbackVaryingEXT(program,
    #       0,
    #       30,
    #       the_length, the_size, the_type, the_name)

    # print the_type.contents.value == GL_FLOAT_VEC4
    # print the_name.value


def configure_transformfeedbackbuffer():
    import ctypes as c
    global program
    # Learned this crazy trick from
    # https://groups.google.com/group/pyglet-users/tree/browse_frm/month/2008-2/3d2fbc1f8dc29e33?rnum=301&start=250&_done=/group/pyglet-users/browse_frm/month/2008-2?start%3D250%26sa%3DN%26&pli=1
    varyings = ["posish", "a_color"]
    arr = (c.c_char_p * (len(varyings) + 1))()
    arr[:-1] = varyings
    arr[ len(varyings) ] = None
    arr = c.cast(arr, c.POINTER(c.POINTER(GLchar))) 
    print arr
    glTransformFeedbackVaryingsEXT(program, 2, arr, GL_INTERLEAVED_ATTRIBS_EXT)

def display():
    
    global t, transformBuffer, feedbackBuffer, program, program2, query
    global lasttime, avgrate, iframe

    thistime = time.time()
    this_rate = 1.0/(thistime - lasttime)
    avgrate = (this_rate + iframe*avgrate)/(iframe+1.0)
    iframe += 1.0
    print "Avg: %0.2f Hz (current: %0.2f Hz)" % (avgrate, this_rate)
    lasttime = thistime


    glLoadIdentity();
    t += 0.1;
    glRotatef( t, 0.0, 0.0, 1.0 );
    # glColor3f( 1,0,0 );
    glViewport( 0,0, 256,256 );

    glClear(GL_COLOR_BUFFER_BIT)

    # glEnable(GL_RASTERIZER_DISCARD);

    # start transform feedback so that vertices get targetted to 'feedbackBuffer'
    glUseProgram(program)
    glBindBufferBaseEXT(GL_TRANSFORM_FEEDBACK_BUFFER, 0, feedbackBuffer)

    
    
    transformBuffer.bind()

    stride = (4+num_joints*3)*4
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer( 4, GL_FLOAT, stride, transformBuffer )

    global uploadUniforms
    if uploadUniforms:
        # Uniforms
        # ========================================
        # Throw up the translations
        translations = np.tile(np.arange(num_joints)[:,np.newaxis], (1,3))
        translation_location = glGetUniformLocation(program, "translations")
        glUniform3fv(translation_location, num_joints, translations)

        # Put up the inverse binding matrices
        Bi = np.array([np.eye(4)*5 for i in range(num_joints)]).astype('float32')
        bindingMatrixInverse_location = glGetUniformLocation(program, "bindingMatrixInverse")
        glUniformMatrix4fv(bindingMatrixInverse_location, num_joints, True, Bi)
        # ========================================
        uploadUniforms = False


    # glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, query)
    glBeginTransformFeedbackEXT( GL_TRIANGLES );
    
    glDrawArrays( GL_TRIANGLES, 0, 2*3 );

    # end transform feedback
    glEndTransformFeedbackEXT();
    # glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN)

    glDisableClientState( GL_VERTEX_ARRAY );
    # glDisable(GL_RASTERIZER_DISCARD);

    # primitives_written = glGetQueryObjectuiv(query, GL_QUERY_RESULT)
    # print "Wrote %d primitives" % primitives_written

    glColor3f( 0,1,0 );
    glViewport( 256,0, 256,256 );


    stride = (16+4)*4
    glUseProgram(program2)
    feedbackBuffer.bind()

    # Get the data out of the VBO
    p = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE)
    ptr = ctypes.cast(p, ctypes.POINTER(ctypes.c_float * 20*6))
    array = np.frombuffer(ptr.contents, 'f')
    glUnmapBuffer(GL_ARRAY_BUFFER)
    
    # Draw the stuff that we pre-computed!
    glEnableClientState(GL_VERTEX_ARRAY);
    global color_loc
    glVertexAttribPointer(color_loc, 4, GL_FLOAT, False, stride, feedbackBuffer+16*4)
    glEnableVertexAttribArray(color_loc)
    glVertexPointer( 4, GL_FLOAT, stride, feedbackBuffer );
        
    glDrawArrays( GL_TRIANGLES, 0, 2*3 );

    glDisableVertexAttribArray(color_loc)
    glDisableClientState( GL_VERTEX_ARRAY );
    

    glUseProgram(0)

    glutSwapBuffers();


def on_keypress(key, x, y):

    if key == 'x':
        sys.exit(1)


def main():
    glutInit([]);
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH );
    glutInitWindowSize( 512,256 );
    glutCreateWindow( "Transform Feedback Demo" );
    glutDisplayFunc( display );
    glutIdleFunc( display );
    glutKeyboardFunc(on_keypress)
    init();
    glutMainLoop();

if __name__ == "__main__":
    main()
