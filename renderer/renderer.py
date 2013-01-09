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
from mako.template import Template

import transformations as tr
import time
import Joints


from pylab import *

class MouseScene(object):
    """A class containing methods for drawing a skinned polygon mesh
    as quickly as possible."""
    def __init__(self, scenefile, \
                        scale_width = 4.0, scale_height = 4.0, scale_length = 4.0, \
                        mouse_width=640, mouse_height=480, \
                        numCols=32, numRows=32, \
                        showTiming=True):

        """For a given scenefile (the output of 
            get_poly_and_skin_info_maya.py), 
            Display the polygon and the joint positions.
        """
        super(MouseScene, self).__init__()
        self.mouse_width = mouse_width
        self.mouse_height = mouse_height
        self.numCols = numCols
        self.numRows = numRows
        self.num_mice = self.numCols * self.numRows
        self.showTiming = showTiming
        self.pull_down_pixels = False

        self.width = self.mouse_width*numCols
        self.height = self.mouse_height*numRows
        if isscalar(scale_width):
            self.scale_width = np.repeat(scale_width, self.num_mice, axis=0)
        else:
            self.scale_width = scale_width
            assert len(self.scale_width) == self.num_mice, "Must have a scale value per mouse"

        if isscalar(scale_height):
            self.scale_height = np.repeat(scale_height, self.num_mice, axis=0)
        else:
            self.scale_height = scale_height
            assert len(self.scale_height) == self.num_mice, "Must have a scale value per mouse"
        if isscalar(scale_length):
            self.scale_length = np.repeat(scale_length, self.num_mice, axis=0)
        else:
            self.scale_length = scale_length
            assert len(self.scale_length) == self.num_mice, "Must have a scale value per mouse"


        self.scenefile = scenefile
        
        # Load in the mesh and skeleton
        f = np.load(self.scenefile)
        self.faceNormals = f['normals']
        v = f['vertices']
        self.vertices = np.ones((len(v),4), dtype='f')
        self.vertices[:,:3] = v
        self.vertex_idx = f['faces']
        self.num_vertices = self.vertices.shape[0]
        self.num_indices = self.vertex_idx.size
        joint_transforms = f['joint_transforms']
        joint_weights = f['joint_weights']
        joint_poses = f['joint_poses']
        joint_rotations = f['joint_rotations']
        joint_translations = f['joint_translations']
        num_joints = len(joint_translations)

        # Find the vertex with the maximum number of joints influencing it
        self.num_joint_influences =  (joint_weights>0).sum(1).max()
        self.num_bones = num_joints

        # Save a list of all of the rotations of all the mice to be displayed
        self.rotations = np.zeros((self.num_mice, self.num_bones, 3), dtype='float32')
        for i in range(self.num_mice):
            for j in range(self.num_bones):
                self.rotations[i,j,:] = joint_rotations[j]

        # Create rotation angles for each 
        self.offset_theta_yaw = np.zeros((self.num_mice,), dtype='float32')
        self.offset_theta_roll = np.zeros((self.num_mice,), dtype='float32')
        self.offset_x = np.zeros((self.num_mice,), dtype='float32')
        self.offset_y = np.zeros((self.num_mice,), dtype='float32')
        self.offset_z = np.zeros((self.num_mice,), dtype='float32')

        # Load up the joints properly into a joint chain
        jointChain = Joints.LinearJointChain()
        for i in range(self.num_bones):
            J = Joints.Joint(rotation=joint_rotations[i],\
                             translation=joint_translations[i])
            jointChain.add_joint(J)
        self.skin = Joints.SkinnedMesh(self.vertices, joint_weights, jointChain)

        # Set up some constants
        self.lastkey = ''
        self.lastx = self.width/2.
        self.lasty = self.height/2.
        self.jointPoints = None
        self.index_vbo = None
        self.mesh_vbo = None
        self.shaderProgram = None
        
        # Timing variables
        self.lasttime = time.time()
        self.avgrate = 0.0
        self.iframe = 0.0
        
    def get_joint_rotations(self):
        return self.rotations

    def set_joint_rotations(self, new_rotations):
        assert new_rotations.shape == self.rotations.shape, \
                    "The new rotations shape must be num_mice x num_joints x 3"
        for i in range(self.num_mice):
            for j in range(self.num_bones):
                self.rotations[i,j,:] = new_rotations[i,j,:]


    def set_mouse_image(self, image):
        import Image
        self.mouse_img = image.astype('float32')
        if image.shape != (self.mouse_height, self.mouse_width):
            I = Image.fromarray(self.mouse_img)
            self.mouse_img = np.array(I.resize((self.mouse_width, self.mouse_height)), dtype='float32')
        self.mouse_img = self.mouse_img/self.get_clipZ()

        width,height = self.mouse_img.shape
        img_for_texture = self.mouse_img[:,:].ravel()

        glBindTexture(GL_TEXTURE_2D, self.mouseTexture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, img_for_texture)
        glBindTexture(GL_TEXTURE_2D, 0)


    def maprange(self, val, source_range=(-100, 500), dest_range=(-5,5), clip=True):
        if clip:
            val = np.clip(val, source_range[0], source_range[1])

        # Normalize
        val = (val - source_range[0]) / (source_range[1] - source_range[0])
        # And remap
        val = val*(dest_range[1]-dest_range[0]) + dest_range[0]

        return val

    def update_vertex_mesh(self):
        self.vertices = self.skin.get_posed_vertices()[:,:3] # leave off the scale parameter
        self.vert_data[:,:3] = self.vertices
        self.mesh_vbo[:] = self.vert_data

    def setup_vbos(self):
        """Initialize a VBO with all-zero entries
        of the correct size. If you have a skin, 
        call update_vertex_mesh() afterwards.
        """

        vidx = self.vertex_idx.ravel().astype('uint16')
        self.index_vbo = vbo.VBO(vidx, target=GL_ELEMENT_ARRAY_BUFFER)

        self.vertices = self.skin.get_posed_vertices()[:,:3].astype('float32') # leave off the scale parameter

        # vertices: x,y,z
        # vertex weights: per-bone weight
        # joint index: which joint each weight correspond to

        # Calculate the indices of the non-zero joint weights
        # In the process, if we have vertices that have less
        # than the maximum number of joint influences,
        # we'll have to add in dummy joints that have no influence.
        # (this greatly simplifies things in the shader code)
        joint_idx = np.zeros((self.num_vertices, self.num_joint_influences), dtype='int')
        nonzero_joint_weights = np.zeros((self.num_vertices, self.num_joint_influences), dtype='float32')

        for i in range(self.num_vertices):
            idx = np.argwhere(self.skin.joint_weights[i,:] > 0).ravel()
            if len(idx) != self.num_joint_influences:
                num_to_add = self.num_joint_influences - len(idx)
                joints_to_add = np.setdiff1d(range(self.num_bones), idx)[:num_to_add]
                idx = np.hstack((idx, joints_to_add))
            joint_idx[i] = idx
            nonzero_joint_weights[i,:] = self.skin.joint_weights[i,joint_idx[i,:]]

        self.vert_data = np.hstack((self.vertices[:,:3], nonzero_joint_weights, joint_idx)).astype('float32')
        self.mesh_vbo = vbo.VBO(self.vert_data)



    # GLUT Callback functions (handling keypress, mouse movement, etc)
    # ============================================================
    def on_keypress(self, key, x, y):
        self.lastkey = key

        if key == 'x':
            sys.exit(1)

    def on_motion(self, x, y):
        # store the mouse coordinate
        self.lastx = float(x)
        self.lasty = float(y)

        # Update our mouse
        deg_range = (-60, 60)
        horz_deg = -self.maprange(self.lastx, (0,self.width), deg_range)
        deg_range = (-60, 60)
        vert_deg = -self.maprange(self.lasty, (0,self.height), deg_range)

        ijoint = 2
        try:
            ijoint = int(self.lastkey)
        except:
            ijoint = 2

        self.skin.jointChain.joints[ijoint].rotation[1] = horz_deg
        self.skin.jointChain.joints[ijoint].rotation[2] = vert_deg
        self.skin.jointChain.solve_forward(0)
        self.jointPoints = self.skin.jointChain.get_joint_world_positions()

    def on_reshape(self, this_width, this_height):
        pass

    def get_clipZ(self):
        return np.max(self.scale_height)*2.0

    def display(self):

        # Tiling parameters
        numCols = self.numCols
        numRows = self.numRows

        # Timing
        if self.showTiming:
            thistime = time.time()
            this_rate = 1.0/(thistime - self.lasttime)
            self.avgrate = (this_rate + self.iframe*self.avgrate)/(self.iframe+1.0)
            self.iframe += 1.0
            print "Avg: %0.2f Hz (current: %0.2f Hz)" % (self.avgrate, this_rate)
            self.lasttime = thistime

        
        glBindFramebuffer(GL_FRAMEBUFFER, self.frameBuffer)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.frameBufferTextures[0], 0)

        # Drawing preparation (view angle adjustment, mostly)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        xmin = -self.mouse_width/2.0
        xmax = self.mouse_width*(numCols-1) - xmin
        ymin = -self.mouse_height/2.0
        ymax = self.mouse_height*(numRows-1) - ymin
        znear = self.get_clipZ()
        zfar = 0.0

        glOrtho(xmin, xmax, ymin, ymax, znear, zfar)

        # Prepare to draw the poly mesh
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Now, rotate to draw the mouse model
        ## Top-down projection

        # Bind our VBOs     
        self.mesh_vbo.bind()
        self.index_vbo.bind()


        # Turn on our shaders
        glUseProgram(self.shaderProgram)
        
        # MOUSE RENDERING
        # ================================================================================
        # ================================================================================

        # Let OpenGL know where we're keeping our
        # vertices, joint weights and joint indices
        glEnableVertexAttribArray(self.joint_weights_location)
        glEnableVertexAttribArray(self.joint_indices_location)
        stride = (3 + self.num_joint_influences*2)*4
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, stride, self.mesh_vbo)
        glVertexAttribPointer(self.joint_weights_location,
                        self.num_joint_influences, GL_FLOAT, 
                        False, stride, self.mesh_vbo+3*4)
        glVertexAttribPointer(self.joint_indices_location,
                        self.num_joint_influences, GL_FLOAT, 
                        False, stride, self.mesh_vbo+3*4+self.num_joint_influences*4)

        # Figure out what our offsets are going to be
        x = self.mouse_width*np.mod(np.arange(numCols*numRows),numCols) + self.offset_x
        y = self.mouse_height*np.floor_divide(np.arange(numCols*numRows), numCols) + self.offset_y
        z = self.offset_z
        theta_yaw = self.offset_theta_yaw
        theta_roll = self.offset_theta_roll


        # Okay, this is where the drawing actually happens.
        # For now, we're drawing each mouse separately.
        # This is expensive and stupid, but instanced drawing
        # is a bit off.
        # TODO: implement instanced drawing.

        for i in range(numCols*numRows):

            glLoadIdentity()
            glRotate(-90, 1., 0., 0.)
            val = 0.0

            # Send up the uniforms
            # glUniform1f(self.offsetx_location, x[i])
            # glUniform1f(self.offsety_location, y[i])
            glUniform1f(self.offsetz_location, z[i])
            glUniform1f(self.scale_width_location, self.scale_width[i])
            glUniform1f(self.scale_length_location, self.scale_length[i])
            glUniform1f(self.scale_height_location, self.scale_height[i])
            glUniform1f(self.theta_yaw_location, -theta_yaw[i])
            glUniform1f(self.theta_roll_location, -theta_roll[i])
            glUniform2fv(self.height_range_location, 1, (0.0, 1.0))
            glUniform3fv(self.rotation_location, self.num_bones, self.rotations[i])
            
            # Rotate the mouse, and draw the mouse
            glTranslate(x[i], 0.0, y[i])
            
            # glRotate(val, 0.0, 0.0, 1.0)
            # glRotate(-theta[i], 0., 1., 0.)
            
            glDrawElements(GL_TRIANGLES, self.num_indices, GL_UNSIGNED_SHORT, self.index_vbo)
            # glRotate(theta[i], 0., 1., 0.)
            # glRotate(-val, 0.0, 0.0, 1.0)
            glTranslate(-x[i], 0.0, -y[i])

        # Time for cleanup. Unbind the VBOs and disable draw modes.
        self.mesh_vbo.unbind()
        self.index_vbo.unbind()
        glDisableClientState(GL_VERTEX_ARRAY)

        # Turn off our shaders
        glUseProgram(0)

        if self.pull_down_pixels:
            self.data = glReadPixels(0,0,self.width,self.height, GL_DEPTH_COMPONENT, GL_FLOAT)

        # REAL DATA SUBTRACTION
        # ================================================================================
        # ================================================================================
        # Draw into our second framebuffer texture
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.frameBufferTextures[1], 0)
        glViewport(0,0,self.width, self.height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Turn on our subtraction shader
        glUseProgram(self.subtractionShaderProgram)

        # Draw from the texture we just drew into
        source_texture_loc = glGetUniformLocation(self.subtractionShaderProgram, "source_texture")
        glActiveTexture(GL_TEXTURE0+1)
        glBindTexture(GL_TEXTURE_2D, self.frameBufferTextures[0])
        glUniform1i(source_texture_loc, 1)
        # The source image size
        source_size_loc = glGetUniformLocation(self.subtractionShaderProgram, "source_size")
        glUniform2f(source_size_loc, self.width, self.height)

        # We'll be subtracting the mouse texture from the previous image
        mouse_texture_loc = glGetUniformLocation(self.subtractionShaderProgram, "mouse_texture")
        glActiveTexture(GL_TEXTURE0+2)
        glBindTexture(GL_TEXTURE_2D, self.mouseTexture)
        glUniform1i(mouse_texture_loc, 2)
        # The mouse's texture size is important as well.
        mouse_size_loc = glGetUniformLocation(self.subtractionShaderProgram, "mouse_size")
        glUniform2f(mouse_size_loc, self.mouse_width, self.mouse_height)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)

        # Load up a fresh perspective
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, 1, 0, 1, -1, self.get_clipZ())
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()


        # Just draw a square. That's our palette to render into.
        glBegin(GL_QUADS)
        glColor4f(0.0, 1.0, 0.0, 1.0)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(xmin, ymin, 0)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(xmin, ymax, 0)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(xmax, ymax, 0)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(xmax, ymin, 0)
        glEnd()

        # Turn off the shader. We're done. 
        glUseProgram(0)
        
        # REDUCTION TO LIKELIHOODS
        # ================================================================================
        # ================================================================================        
        
        # Now, swap out the textures backing the depth attachment of the framebuffer
        output_size = [self.numCols, self.numRows]

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.frameBufferTextures[0], 0)
        glViewport(0,0,output_size[0], output_size[1])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        xmin = 0
        xmax = 1
        ymin = 0
        ymax = 1
        znear = -1
        zfar = 1
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(xmin, xmax, ymin, ymax, znear, zfar)


        # Prepare to draw the poly mesh
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glUseProgram(self.reductionShaderProgram)

        source_size_loc = glGetUniformLocation(self.reductionShaderProgram, "source_size")
        glUniform2f(source_size_loc, self.width, self.height)
        dest_size_loc = glGetUniformLocation(self.reductionShaderProgram, "dest_size")
        glUniform2f(dest_size_loc, output_size[0], output_size[1])

        source_texture_loc = glGetUniformLocation(self.reductionShaderProgram, "source_texture")
        glActiveTexture(GL_TEXTURE0+1)
        glBindTexture(GL_TEXTURE_2D, self.frameBufferTextures[1])
        glUniform1i(source_texture_loc, 1)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)

        # Just draw a square
        glBegin(GL_QUADS)
        # One big quad
        glColor4f(0.0, 1.0, 0.0, 1.0)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(xmin, ymin, 0)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(xmin, ymax, 0)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(xmax, ymax, 1)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(xmax, ymin, 1)
        glEnd()

        glUseProgram(0)
        
        self.likelihood = glReadPixels(0,0,output_size[0], output_size[1], GL_DEPTH_COMPONENT, GL_FLOAT)
        self.likelihood = -self.likelihood
        # glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0)
        # glBindTexture(GL_TEXTURE_2D, self.frameBufferTextures[0])
        # self.likelihood = glGetTexImagef(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_FLOAT)
        # glBindTexture(GL_TEXTURE_2D, 0)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        

        # ================================================================================
        if self.pull_down_pixels:
            
            # data = self.original_data
            # data = data.ravel().reshape((self.height, self.width, 1))[:,:,0]
            self.data = self.get_clipZ()*(1-self.data)
            posed_mice = np.zeros((self.numRows*numCols, self.mouse_height, self.mouse_width), dtype='float32')
            for i in range(self.numRows):
                startr = i*self.mouse_height
                endr = startr+self.mouse_height
                for j in range(self.numCols):
                    startc = j*self.mouse_width
                    endc = startc+self.mouse_width
                    posed_mice[i*self.numRows+j] = self.data[startr:endr,startc:endc]
                    
            self.posed_mice = posed_mice

        # If we are using a framebuffer, we'll finally unbind it.
        glBindFramebuffer(GL_FRAMEBUFFER, 0)


    def setup_shaders(self):
        if not glUseProgram:
            print 'Missing Shader Objects!'
            sys.exit(1)

        vertexShaderString = """
        #version 120
        // Application to vertex shader
        varying vec4 vertex_color;
        uniform float offsetx;
        uniform float offsety;
        uniform float offsetz;
        uniform float scale_width;
        uniform float scale_height;
        uniform float scale_length;
        uniform float theta_yaw;
        uniform float theta_roll;
        uniform vec2 height_range;

        uniform vec3 rotation[${num_joints}]; // rotations on each joint
        uniform vec3 translation[${num_joints}]; // translations of each joint from the previous
        uniform mat4 bindingMatrixInverse[${num_joints}]; // the inverse binding matrix
        attribute vec4 joint_weights;
        attribute vec4 joint_indices;

        mat4 lastJointWorldMatrix = mat4(1.0);
        mat4 jointWorldMatrix = mat4(1.0);
        mat4[${num_joints}] posingMatrix;
        mat4 localRotation = mat4(1.0);


        mat4 calcLocalRotation(in vec3 rotation, in vec3 translation) {

            vec3 this_rotation = radians(rotation);
            vec3 cosrot = cos(this_rotation);
            vec3 sinrot = sin(this_rotation);
            vec4 pt;

            mat4 Rx = mat4(cosrot.x);
            Rx[3].w = 1.0;
            mat4 Ry = mat4(cosrot.y);
            Ry[3].w = 1.0;
            mat4 Rz = mat4(cosrot.z);
            Rz[3].w = 1.0;

            Rx[0].x += 1.0 - cosrot.x;
            Ry[1].y += 1.0 - cosrot.y;
            Rz[2].z += 1.0 - cosrot.z;

            Rx[1].z += -sinrot.x;
            Rx[2].y += sinrot.x;

            Ry[0].z += sinrot.y;
            Ry[2].x += -sinrot.y;

            Rz[0].y += -sinrot.z;
            Rz[1].x += sinrot.z;

            mat4 Tout = Rx*Ry*Rz;

            Tout[0].w = translation.x;
            Tout[1].w = translation.y;
            Tout[2].w = translation.z;

            return Tout;
        }

        mat4 mat_at_i(mat4 A[${num_joints}], int the_index) {
            if (the_index == 0) { return A[0]; }

            % for i in range(num_joints):

            else if (the_index == ${i}) { return A[${i}]; }

            % endfor
            
        }

        mat4[${num_joints}] set_mat_at_i(mat4 A[${num_joints}], mat4 B, int the_index) {
            if (the_index == 0) { A[0] = B; return A; }

            % for i in range(num_joints):

            else if (the_index == ${i}) { A[${i}] = B; return A; }

            % endfor
        }
        
        void main()
        {   

            // For each joint
            // 1. calculate its local rotation matrix
            // 2. calculate its world position from the previous joint's world matrix
            // 3. multiply its inverse binding matrix into its world matrix as the posing matrix
            // 4. multiply the posing matrix into the vertex
            
            // NOTE: there can be no for loops over declared arrays
            // on Apple graphics hardware. It is absolutely ridiculous. 
            // So, I'm using a templated loop with Mako. 

            % for i in range(num_joints):

            // Calculate a joint's local rotation matrix
            localRotation = calcLocalRotation(rotation[${i}], translation[${i}]);
        
            // Calculate its world position
            jointWorldMatrix = localRotation*lastJointWorldMatrix;
            
            // Multiply the inverse binding matrix into the world matrix
            posingMatrix[${i}] = bindingMatrixInverse[${i}] * jointWorldMatrix;
        
            // Get ready for the next iteration
            lastJointWorldMatrix = jointWorldMatrix;

            % endfor


            // Calculate a joint's local rotation matrix
            vec4 vertex = vec4(0., 0., 0., 0.0);
            int index;

            % for i in range(num_joint_influences):

            index = int(joint_indices[${i}]);
            vertex += joint_weights[${i}] * gl_Vertex * mat_at_i(posingMatrix, index);

            % endfor

            vertex.x *= scale_width;
            vertex.z *= scale_length;
            vertex.y *= scale_height;

            // vertex[0] = vertex[0]+offsetx;
            // vertex[1] = vertex[1]+offsetz;
            // vertex[2] = vertex[2]+offsety;

            // TODO: get the roll working again
            mat4 yaw = calcLocalRotation(vec3(0.0, theta_yaw, 0.0), vec3(0.0));
            vertex = vertex * yaw;

            // Transform vertex by modelview and projection matrices
            gl_Position = gl_ModelViewProjectionMatrix * vertex;

            // Pass on the vertex color 
            float height = vertex[1];
            float the_color = (height-height_range[0])/(height_range[1]-height_range[0]);
            vertex_color = vec4(the_color, the_color, the_color, 1.0);

        }

        """

        makoTemplate = Template(vertexShaderString)
        vertexShaderString = makoTemplate.render(num_joints=self.num_bones,\
                                    num_joint_influences=self.num_joint_influences)
        vertexShader = shaders.compileShader(vertexShaderString, GL_VERTEX_SHADER)
        
        fragmentShader = shaders.compileShader("""
        #version 120
        #extension GL_ARB_texture_rectangle : enable 
        #extension GL_ARB_draw_buffers : enable 
        varying vec4 vertex_color;
        uniform sampler2D mouse_tex;
        uniform vec2 mouse_img_size;
        void main() {
            gl_FragColor = vertex_color;
            float z = gl_FragCoord.z;

            vec2 position = gl_FragCoord.xy / mouse_img_size.xy;
            vec4 texvec = texture2D(mouse_tex, position);
            float diff = abs(texvec.r - z);
            gl_FragColor.rgb = vec3(diff);

            gl_FragDepth = gl_FragCoord.z;
            gl_FragDepth = diff;
        }

        """, GL_FRAGMENT_SHADER)

        self.shaderProgram = shaders.compileProgram(vertexShader, fragmentShader)

        # Now, let's make sure our uniform and attribute value value will be sent 
        # for uniform in ['joints', 'scale', 'offsetx', 'offsety', 'rotation', 'translation', 'bindingMatrixInverse']:
        for uniform in ['scale_length', 'scale_width', 'scale_height', \
                            'offsetx', 'offsety', 'offsetz', \
                            'theta_yaw', 'theta_roll', \
                            'height_range',\
                            'rotation', 'translation', 'bindingMatrixInverse',
                            'mouse_tex', 'mouse_img_size']:
            location = glGetUniformLocation(self.shaderProgram, uniform)
            name = uniform+"_location"
            setattr(self, uniform+"_location", location)
        for attribute in ['joint_weights', 'joint_indices']:
            location = glGetAttribLocation(self.shaderProgram, attribute)
            setattr(self, attribute+"_location", location)

        # There's a couple uniforms that never change
        # For now, the inverse binding matrix, and the joint translations
        # (the skeleton morphology does not change, and joints only rotate, 
        #   they don't slide)
        glUseProgram(self.shaderProgram)
        joints = self.skin.jointChain.joints
        Bi = np.array([np.array(j.Bi.copy()) for j in joints]).astype('float32')
        glUniformMatrix4fv(self.bindingMatrixInverse_location, self.num_bones, True, Bi)
        translations = np.array([j.translation.copy() for j in joints]).astype('float32')
        glUniform3fv(self.translation_location, self.num_bones, translations)
        glUseProgram(0)

    def setup_subtraction_shader(self):
        vShader = shaders.compileShader("""
            #version 120
            void main() {
                gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
            }
            """, GL_VERTEX_SHADER)

        fShader = shaders.compileShader("""
            #version 120
            uniform sampler2D source_texture;
            uniform sampler2D mouse_texture;
            uniform vec2 source_size;
            uniform vec2 mouse_size;

            void main() {
                vec2 source_coord = gl_FragCoord.xy / source_size;
                vec2 mouse_coord = gl_FragCoord.xy / mouse_size;
                float source_depth = texture2D(source_texture, source_coord).r;
                float mouse_depth = texture2D(mouse_texture, mouse_coord).r;
                gl_FragDepth = abs(1.0 - source_depth - mouse_depth);
            }
            """, GL_FRAGMENT_SHADER)

        self.subtractionShaderProgram = shaders.compileProgram(vShader, fShader)

    def setup_reduction_shader(self):
        vShader = shaders.compileShader("""
            #version 120
            void main() {
                gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
            }
            """, GL_VERTEX_SHADER)

        fShader = shaders.compileShader("""
            #version 120
            uniform sampler2D source_texture;
            uniform sampler2D mouse_texture;
            uniform vec2 dest_size;
            uniform vec2 source_size;
            uniform float bias;
            
            void main() {

                // First, figure out the range of values we'll need to snag
                vec2 scale = source_size / dest_size;

                vec2 ll = (floor(gl_FragCoord.xy)) * scale;
                vec2 ur = (floor(gl_FragCoord.xy)+1) * scale;

                ivec2 ll_pixel = ivec2(ll);
                ivec2 ur_pixel = ivec2(ur);
                
                int width = int(scale.x);
                int height = int(scale.y);
                float num_pixels = width*height;

                float target_depth = 0.0;
                for (int i=ll_pixel.x; i < ur_pixel.x; ++i) {
                    float x = i/source_size.x;
                    for (int j=ll_pixel.y; j < ur_pixel.y; ++j) {
                        float y = j/source_size.y;
                        float this_depth = texture2D(source_texture, vec2(x,y)).r;
                        target_depth += this_depth/num_pixels;
                    }
                }
                gl_FragDepth = target_depth;

            }
            """, GL_FRAGMENT_SHADER)



        self.reductionShaderProgram = shaders.compileProgram(vShader, fShader)

        glUseProgram(0)

    def setup_texture(self):
        import Image
        # from load_data import load_behavior_data
        # image = load_behavior_data("../Test Data", 35+1, 'images')[-1]
        # self.mouse_img = image.T[::-1,:].astype('float32')/self.get_clipZ()
        self.mouse_img = np.zeros((self.mouse_width, self.mouse_height), dtype='float32')        

        width,height = self.mouse_img.shape

        self.mouseTexture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.mouseTexture)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
        glBindTexture(GL_TEXTURE_2D, 0)

    def setup_fbo(self):

        # Make a framebuffer and bind it
        self.frameBuffer = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.frameBuffer)

        # Setup two textures that will accept all drawn pixels
        self.frameBufferTextures = []
        for i in range(2):
            self.frameBufferTextures.append(glGenTextures(1))
            glBindTexture(GL_TEXTURE_2D, self.frameBufferTextures[i])
            glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
                self.width, self.height, 0,
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

        # Tell the framebuffer to send pixels to the texture we just made
        # glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.frameBufferTextures[0], 0)

        # Every FBO has to have a color attachment. We don't persist it
        # because we won't be using it. We only care about depth.
        color = glGenRenderbuffers(1)
        glBindRenderbuffer( GL_RENDERBUFFER, color )
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, self.width, self.height)
        glFramebufferRenderbuffer( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color )
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def gl_init(self):

        glutInit([])

        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
        glutInitWindowSize(self.width, self.height)
        glutCreateWindow('Mouse Model')

        glutKeyboardFunc(self.on_keypress)
        glutMotionFunc(self.on_motion)
        glutDisplayFunc(self.display)
        
        # States to set
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_NORMALIZE)

        # Setup our VBOs and shaders
        self.setup_vbos()
        self.update_vertex_mesh()
        self.setup_shaders()
        self.setup_texture()
        self.setup_fbo()
        self.setup_reduction_shader()
        self.setup_subtraction_shader()


    def get_likelihood(self, new_img, x, y, theta, particle_data, return_posed_mice=False):
        """Calculate the likelihood of a list of particles given a mouse mouse_image

        particle_data - num_particles x num_variables
        mouse_image - the current mouse image
        x           - the x position of the mouse
        y           - the y position of the mouse
        theta       - the theta angle of the mouse

        mousescene - an instance of MouseScene, which controls the rendering
        likelihood_array (optional) - num_particles array 
                                    (provide if you don't want a memory copy)
        """

        # Set the new mouse image
        self.set_mouse_image(new_img)

        # Check the number of particles
        num_particles, num_vars = particle_data.shape

        # If we have more particles than mice, 
        # we'll have to do multiple rendering passes to get all the likelihoods
        num_passes = int(np.ceil(num_particles / self.num_mice))


        # Here we extract the parameters from the particle_data, 
        # as we think they should be sitting.
        # So, right now, that's 
        # - offsetx
        # - offsety
        # - body angle
        # and for each of 9 joints,
        # {
        #   - vertical rotation from rest
        #   - horizontal rotation from rest
        # }
        
        all_likelihoods = np.zeros((num_particles,), dtype='float32')
        if return_posed_mice:
            posed_mice = np.zeros((num_particles, self.mouse_height, self.mouse_width), dtype='float32')
            self.pull_down_pixels = True

        this_particle_data = np.zeros((self.num_mice,num_vars))
        for i in range(num_passes):

            # Slice out our current particles to render
            start = i*self.num_mice
            end = start+self.num_mice
            # NOTE: end-start may be longer than particle_data[start:end]
            # this works because of numpy indexing, e.g. randn(10)[8:15]
            sz = particle_data[start:end].shape[0]
            this_particle_data[:sz] = particle_data[start:end]

            # Set the position and angle offsets
            self.offset_x = this_particle_data[:,0] - x
            self.offset_y = this_particle_data[:,1] - y
            self.offset_z = this_particle_data[:,2]
            self.offset_theta_yaw = this_particle_data[:,3] - theta
            self.offset_theta_roll = this_particle_data[:,4]
            self.scale_width = this_particle_data[:,5]
            self.scale_length = this_particle_data[:,6]
            self.scale_height = this_particle_data[:,7]

            # Set the joint rotations
            rotations = this_particle_data[:,8:]
            rotations = np.reshape(rotations, (self.num_mice, -1, 3))
            self.rotations = rotations

            # Display the mouse scene (or render to a framebuffer, alternatively)
            self.display()

            # Grab the computed likelihoods
            all_likelihoods[start:end] = self.likelihood.ravel()[:sz]
            if return_posed_mice:
                posed_mice[start:end] = self.posed_mice[:sz]
        
        if return_posed_mice:
            return all_likelihoods, posed_mice
        else:
            return all_likelihoods

def test_palette():

    path_to_behavior_data = os.path.join(os.path.dirname(__file__),'..','Test Data/Blurred Edge')
    which_img = 35
    # which_img = 731
    from load_data import load_behavior_data
    image = load_behavior_data(path_to_behavior_data, which_img+1, 'images')[-1]
    image = image.T[::-1,:].astype('float32')


    scenefile = os.path.join(os.path.dirname(__file__),"data/mouse_mesh_low_poly3.npz")

    num_particles = 32**2
    numCols = 32
    numRows = 32
    ms = MouseScene(scenefile, mouse_width=80, mouse_height=80, \
                                scale_width = 18.0, scale_height = 200.0, 
                                scale_length = 18.0, \
                                numCols=numCols, numRows=numRows)
    ms.gl_init()

    # Let's fill in our particles
    particle_data = np.zeros((num_particles, 8+ms.num_bones*3))

    # Set the horizontal offsets
    position_val = 0
    particle_data[1:,:2] = np.random.normal(loc=0, scale=1, size=(num_particles-1, 2))

    # Set the vertical offset
    particle_data[1:,2] = np.random.normal(loc=0.0, scale=3.0, size=(num_particles-1,))

    # Set the angles (yaw and roll)
    theta_val = 0
    particle_data[1:,3] = theta_val + np.random.normal(loc=0, scale=3, size=(num_particles-1,))
    particle_data[1:,4] = np.random.normal(loc=0, scale=0.01, size=(num_particles-1,))

    # Set the scales (width, length, height)
    particle_data[0,5] = np.max(ms.scale_width)
    particle_data[0,6] = np.max(ms.scale_length)
    particle_data[0,7] = np.max(ms.scale_height)
    particle_data[1:,5] = np.random.normal(loc=18, scale=2, size=(num_particles-1,))
    particle_data[1:,6] = np.random.normal(loc=18, scale=2, size=(num_particles-1,))
    particle_data[1:,7] = np.abs(np.random.normal(loc=200.0, scale=10, size=(num_particles-1,)))

    # Grab the baseline joint rotations
    rot = ms.get_joint_rotations().copy()
    particle_data[:,8::3] = rot[:,:,0]
    particle_data[:,9::3] = rot[:,:,1]
    particle_data[:,10::3] = rot[:,:,2]

    # Add noise to the baseline rotations (just the pitch and yaw for now)
    # particle_data[1:,8::3] += np.random.normal(scale=20, size=(num_particles-1, ms.num_bones))
    particle_data[1:,9+6::3] += np.random.normal(scale=20, size=(num_particles-1, ms.num_bones-2))
    particle_data[1:,10::3] += np.random.normal(scale=20, size=(num_particles-1, ms.num_bones))

    start = time.time()
    for i in range(30):
        ms.set_mouse_image(image)
        ms.pull_down_pixels = False
        ms.display()
    duration = time.time() - start
    print "Took %f seconds" % duration

    # image = np.zeros((80,80), dtype='float32')
    # likelihoods = ms.get_likelihood(image, \
    #                     x=position_val, y=position_val, \
    #                     theta=theta_val, \
    #                     particle_data=particle_data,
    #                     return_posed_mice=False)

    # figure(figsize=(10,5))
    # subplot(1,2,1)
    # imshow(ms.data)
    # colorbar()
    # title("Original Data")
    # subplot(1,2,2)
    # imshow(ms.likelihood)
    # title("Reduced Data")
    # colorbar()

    return ms

def test_single_mouse():
    path_to_behavior_data = os.path.join(os.path.dirname(__file__),'..','Test Data/Blurred Edge')
    # which_img = 35
    which_img = 731
    from load_data import load_behavior_data
    image = load_behavior_data(path_to_behavior_data, which_img+1, 'images')[-1]
    image = image.T[::-1,:].astype('float32')

    num_particles = 64**2
    numCols = 32
    numRows = 32
    scenefile = os.path.join(os.path.dirname(__file__),"data/mouse_mesh_low_poly3.npz")

    ms = MouseScene(scenefile, mouse_width=80, mouse_height=80, \
                                scale_width = 18.0, scale_height = 200.0, 
                                scale_length = 18.0, \
                                numCols=numCols, numRows=numRows)
    ms.gl_init()


    # Figure out the number of passes we'll be making
    num_passes = int(num_particles / ms.num_mice)

    # Let's fill in our particles
    particle_data = np.zeros((num_particles, 8+ms.num_bones*3))

    # Set the horizontal offsets
    position_val = 0
    particle_data[1:,:2] = np.random.normal(loc=0, scale=1, size=(num_particles-1, 2))

    # Set the vertical offset
    particle_data[1:,2] = np.random.normal(loc=0.0, scale=3.0, size=(num_particles-1,))

    # Set the angles (yaw and roll)
    theta_val = 0
    particle_data[1:,3] = theta_val + np.random.normal(loc=0, scale=3, size=(num_particles-1,))
    particle_data[1:,4] = np.random.normal(loc=0, scale=0.01, size=(num_particles-1,))

    # Set the scales (width, length, height)
    particle_data[0,5] = np.max(ms.scale_width)
    particle_data[0,6] = np.max(ms.scale_length)
    particle_data[0,7] = np.max(ms.scale_height)
    particle_data[1:,5] = np.random.normal(loc=18, scale=2, size=(num_particles-1,))
    particle_data[1:,6] = np.random.normal(loc=18, scale=2, size=(num_particles-1,))
    particle_data[1:,7] = np.abs(np.random.normal(loc=200.0, scale=10, size=(num_particles-1,)))

    # Grab the baseline joint rotations
    rot = ms.get_joint_rotations().copy()
    rot = np.tile(rot, (num_passes, 1, 1))
    particle_data[:,8::3] = rot[:,:,0]
    particle_data[:,9::3] = rot[:,:,1]
    particle_data[:,10::3] = rot[:,:,2]

    # Add noise to the baseline rotations (just the pitch and yaw for now)
    # particle_data[1:,8::3] += np.random.normal(scale=20, size=(num_particles-1, ms.num_bones))
    particle_data[1:,9+6::3] += np.random.normal(scale=20, size=(num_particles-1, ms.num_bones-2))
    particle_data[1:,10::3] += np.random.normal(scale=20, size=(num_particles-1, ms.num_bones))


    likelihoods = ms.get_likelihood(image, \
                        x=position_val, y=position_val, \
                        theta=theta_val, \
                        particle_data=particle_data,
                        return_posed_mice=False)


    # L = ms.likelihood.T.ravel()
    particle_rotations = np.hstack((particle_data[:,9::3], particle_data[:,10::3]))
    real_rotations = np.hstack((rot[:,:,1], rot[:,:,2]))
    rotation_diffs = np.sum((particle_rotations - real_rotations)**2.0, 1)

    figure();
    plot(rotation_diffs, likelihoods, '.k')
    ylabel("Likelihood")
    xlabel("Rotation angle differences")
    title("Rotation angle difference versus likelihood")

    binrange = (0,3000)
    num_bins = 10
    bins = np.linspace(binrange[0], binrange[1], num_bins)
    index = np.digitize(rotation_diffs, bins)
    means = [np.mean(likelihoods[index==i]) for i in range(num_bins)]
    errs = [np.std(likelihoods[index==i]) for i in range(num_bins)]
    errorbar(bins, means, yerr=errs, linewidth=2)


    # Find the five best mice
    idx = np.argsort(likelihoods)
    fivebest = np.hstack(posed_mice[idx[-5:]])
    # Show first the raw mouse, then my hand-posed mouse, and then the five best poses
    
    figure()
    title("Five best (best, far right)")
    imshow(np.hstack((image, posed_mice[0], fivebest)))
    vlines(ms.mouse_width, 0, ms.mouse_height, linewidth=3, color='w')
    text(ms.mouse_width/2.0, ms.mouse_width*0.9,'Real Mouse',
         horizontalalignment='center',
         verticalalignment='center',
         color='white')
    vlines(ms.mouse_width*2, 0, ms.mouse_height, linewidth=3, color='w')
    text(ms.mouse_width*1.5, ms.mouse_width*0.9,'Unposed Mouse',
         horizontalalignment='center',
         verticalalignment='center',
         color='white')

    return ms, rotation_diffs, likelihoods, particle_data, posed_mice


if __name__ == '__main__':
    
        # ms, rotation_diffs, likelihoods, particle_data, posed_mice = test_single_mouse()
        ms = test_palette()
        plt.show()

        # import Image
        # big_likelihood = -np.array(Image.fromarray(-ms.likelihood).resize(ms.diff_data.shape))
        # imshow(ms.diff_data-ms.get_clipZ()*0.1*big_likelihood)
