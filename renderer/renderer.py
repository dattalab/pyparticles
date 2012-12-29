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

import transformations as tr
import time
import Joints

from pylab import *

class MouseScene(object):
	"""A class containing methods for drawing a skinned polygon mesh
	as quickly as possible."""
	def __init__(self, scenefile, scale = 4.0, \
						mouse_width=640, mouse_height=480, \
						numCols=32, numRows=32, useFramebuffer=False, \
						debug=False, showTiming=True):

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
		self.debug = debug
		self.showTiming = showTiming

		assert np.mod(self.num_mice, 256) == 0, \
			"Number of mice must be a multiple of 1024"

		self.width = self.mouse_width*numCols
		self.height = self.mouse_height*numRows
		self.scale = scale
		self.useFramebuffer = useFramebuffer

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
		self.offset_theta = np.zeros((self.num_mice,), dtype='float32')
		self.offset_x = np.zeros((self.num_mice,), dtype='float32')
		self.offset_y = np.zeros((self.num_mice,), dtype='float32')


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

		num_elements_per_coord = self.vertices.shape[1]-1 + \
									self.num_joint_influences + \
									self.num_joint_influences		
		# vert_data = np.zeros((self.num_vertices, num_elements_per_coord), dtype='float32')

		# Calculate the indices of the non-zero joint weights
		joint_idx = np.zeros((self.num_vertices, self.num_joint_influences), dtype='int')
		nonzero_joint_weights = np.zeros((self.num_vertices, self.num_joint_influences), dtype='float32')
		for i in range(self.num_vertices):
			joint_idx[i,:] = np.argwhere(self.skin.joint_weights[i,:] > 0).ravel()
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

		if self.useFramebuffer:
			glBindFramebuffer(GL_FRAMEBUFFER, self.frameBuffer)

		# Drawing preparation (view angle adjustment, mostly)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()

		xmin = -self.mouse_width/2.0
		xmax = self.mouse_width*(numCols-1) - xmin
		ymin = -self.mouse_height/2.0
		ymax = self.mouse_height*(numRows-1) - ymin
		zmin = 20.*self.scale*4.0
		zmax = -50.*self.scale*4.0

		glOrtho(xmin, xmax, ymin, ymax, zmin, zmax)

		# Prepare to draw the poly mesh
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()

		# Experimental texture drawing code
		# ==============================
		



		# Now, rotate to draw the mouse model
		## Top-down projection
		glRotate(-90, 1., 0., 0.)

		## Skew to the side (if you want to view spine movement)
		if self.lastkey == 'r':
			glRotate(90, 1., 0., 0.)
			glRotate(90, 0., 1., 0.)

		# Bind our VBOs		
		self.mesh_vbo.bind()
		self.index_vbo.bind()

		# Turn on our shaders
		glUseProgram(self.shaderProgram)

		# Draw the poly mesh
		# ==============================

		# Let OpenGL know where we're keeping our
		# vertices, joint weights and joint indices
		glEnableVertexAttribArray(self.joint_weights_location)
		glEnableVertexAttribArray(self.joint_indices_location)
		stride = (3 + self.num_joint_influences*2)*4
		glEnableClientState(GL_VERTEX_ARRAY)
		glVertexPointer(3, GL_FLOAT, stride, self.mesh_vbo)
		glVertexAttribPointer(self.joint_weights_location,
						4, GL_FLOAT, 
						False, stride, self.mesh_vbo+3*4)
		glVertexAttribPointer(self.joint_indices_location,
						4, GL_FLOAT, 
						False, stride, self.mesh_vbo+3*4+self.num_joint_influences*4)

		# Figure out what our offsets are going to be
		x = self.mouse_width*np.mod(np.arange(numCols*numRows),numCols) + self.offset_x
		y = self.mouse_height*np.floor_divide(np.arange(numCols*numRows), numCols) + self.offset_y
		theta = self.offset_theta

		# How are we going to scale our mouse? For now, one scale
		# for all displayed mice
		# TODO: make scale settable per-mouse
		scale_array = np.repeat(self.scale, numCols*numRows, axis=0)

		# If we're displaying to the screen, we're debugging,
		# which means we'd like to see a bunch of random poses.
		if not self.useFramebuffer:
			old_rotations = np.copy(self.rotations)
			self.rotations[:,:,1:] += np.random.normal(scale=10,
												size=(self.num_mice, self.num_bones, 2))

		# Okay, this is where the drawing actually happens.
		# For now, we're drawing each mouse separately.
		# This is expensive and stupid, but instanced drawing
		# is a bit off.
		# TODO: implement instanced drawing.
		for i in range(numCols*numRows):

			# Send up the uniforms
			glUniform1f(self.offsetx_location, x[i])
			glUniform1f(self.offsety_location, y[i])
			glUniform1f(self.scale_location, scale_array[i])
			glUniform3fv(self.rotation_location, self.num_bones, self.rotations[i])
			
			# Rotate the mouse, and draw the mouse
			glRotate(-theta[i], 0., 1., 0.)
			glDrawElements(GL_TRIANGLES, self.num_indices, GL_UNSIGNED_SHORT, self.index_vbo)
			glRotate(theta[i], 0., 1., 0.)

		# Time for cleanup. Unbind the VBOs and disable draw modes.
		self.mesh_vbo.unbind()
		self.index_vbo.unbind()
		glDisableClientState(GL_VERTEX_ARRAY)

		# Turn off our shaders
		glUseProgram(0)

		if not self.useFramebuffer:
			glutSwapBuffers()
			self.rotations = np.copy(old_rotations)
		# ==============================



		# For now, this is how we get mice data
		# off of the GPU. Just grab the raw pixel data back from the GPU,
		# and do our likelihood computations locally.
		# The speedups to be gained here by moving it to the GPU 
		# aren't super huge, as far as I'm aware.
		data = glReadPixels(0,0,self.width,self.height, GL_RGB, GL_FLOAT)
		data = data.ravel().reshape((self.height, self.width, 3))[:,:,0]
		this_diff = (np.tile(self.mouse_img, (self.numRows, self.numCols)) - data)**2.0
		likelihood = np.zeros((self.numRows, self.numCols), dtype='float32')
		posed_mice = np.zeros((self.numRows*numCols, self.mouse_height, self.mouse_width), dtype='float32')
		for i in range(self.numRows):
			startr = i*self.mouse_height
			endr = startr+self.mouse_height
			for j in range(self.numCols):
				startc = j*self.mouse_width
				endc = startc+self.mouse_width
				likelihood[i,j] = -this_diff[startr:endr,startc:endc].sum()
				posed_mice[i*self.numRows+j] = data[startr:endr,startc:endc]
				
		self.posed_mice = posed_mice
		self.data = data
		self.likelihood = likelihood
		self.diffmap = this_diff

		# If we'd like to write the frames to disk, go ahead.
		if self.debug:
			np.savez("data/frame%d.npz"%self.iframe, \
							frame=data, \
							diffmap=this_diff, \
							likelihood = likelihood,
							mouse_img = self.mouse_img)


		# If we are using a framebuffer, we'll finally unbind it.
		if self.useFramebuffer:
			glBindFramebuffer(GL_FRAMEBUFFER, 0)


	def setup_shaders(self):
		if not glUseProgram:
			print 'Missing Shader Objects!'
			sys.exit(1)

		vertexShader = shaders.compileShader("""
		#version 120
		// Application to vertex shader
		varying vec4 vertex_color;
		uniform float offsetx;
		uniform float offsety;
		uniform float scale;

		uniform vec3 rotation[9]; // rotations on each joint
		uniform vec3 translation[9]; // translations of each joint from the previous
		uniform mat4 bindingMatrixInverse[9]; // the inverse binding matrix
		attribute vec3 joint_weights;
		attribute vec3 joint_indices;

		mat4 lastJointWorldMatrix = mat4(1.0);
		mat4 jointWorldMatrix = mat4(1.0);
		mat4[9] posingMatrix;
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

		mat4 mat_at_i(mat4 A[9], int the_index) {
			if (the_index == 0) { return A[0]; }
			else if (the_index == 1) { return A[1]; }
			else if (the_index == 2) { return A[2]; }
			else if (the_index == 3) { return A[3]; }
			else if (the_index == 4) { return A[4]; }
			else if (the_index == 5) { return A[5]; }
			else if (the_index == 6) { return A[6]; }
			else if (the_index == 7) { return A[7]; }
			else if (the_index == 8) { return A[8]; }
		}

		mat4[9] set_mat_at_i(mat4 A[9], mat4 B, int the_index) {
			if (the_index == 0) { A[0] = B; return A; }
			else if (the_index == 1) { A[1] = B; return A; }
			else if (the_index == 2) { A[2] = B; return A; }
			else if (the_index == 3) { A[3] = B; return A; }
			else if (the_index == 4) { A[4] = B; return A; }
			else if (the_index == 5) { A[5] = B; return A; }
			else if (the_index == 6) { A[6] = B; return A; }
			else if (the_index == 7) { A[7] = B; return A; }
			else if (the_index == 8) { A[8] = B; return A; }
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
			// I might consider using string templating in the future.

			
			// Calculate a joint's local rotation matrix
			localRotation = calcLocalRotation(rotation[0], translation[0]);
		
			// Calculate its world position
			jointWorldMatrix = localRotation*lastJointWorldMatrix;
			
			// Multiply the inverse binding matrix into the world matrix
			posingMatrix[0] = bindingMatrixInverse[0] * jointWorldMatrix;
		
			// Get ready for the next iteration
			lastJointWorldMatrix = jointWorldMatrix;


			// Now, repeat it another 8 times, with no loop.
			localRotation = calcLocalRotation(rotation[1], translation[1]);
			jointWorldMatrix = localRotation*lastJointWorldMatrix;
			posingMatrix[1] = bindingMatrixInverse[1] * jointWorldMatrix;
			lastJointWorldMatrix = jointWorldMatrix;

			localRotation = calcLocalRotation(rotation[2], translation[2]);
			jointWorldMatrix = localRotation*lastJointWorldMatrix;
			posingMatrix[2] = bindingMatrixInverse[2] * jointWorldMatrix;
			lastJointWorldMatrix = jointWorldMatrix;

			localRotation = calcLocalRotation(rotation[3], translation[3]);
			jointWorldMatrix = localRotation*lastJointWorldMatrix;
			posingMatrix[3] = bindingMatrixInverse[3] * jointWorldMatrix;
			lastJointWorldMatrix = jointWorldMatrix;

			localRotation = calcLocalRotation(rotation[4], translation[4]);
			jointWorldMatrix = localRotation*lastJointWorldMatrix;
			posingMatrix[4] = bindingMatrixInverse[4] * jointWorldMatrix;
			lastJointWorldMatrix = jointWorldMatrix;

			localRotation = calcLocalRotation(rotation[5], translation[5]);
			jointWorldMatrix = localRotation*lastJointWorldMatrix;
			posingMatrix[5] = bindingMatrixInverse[5] * jointWorldMatrix;
			lastJointWorldMatrix = jointWorldMatrix;

			localRotation = calcLocalRotation(rotation[6], translation[6]);
			jointWorldMatrix = localRotation*lastJointWorldMatrix;
			posingMatrix[6] = bindingMatrixInverse[6] * jointWorldMatrix;
			lastJointWorldMatrix = jointWorldMatrix;

			localRotation = calcLocalRotation(rotation[7], translation[7]);
			jointWorldMatrix = localRotation*lastJointWorldMatrix;
			posingMatrix[7] = bindingMatrixInverse[7] * jointWorldMatrix;
			lastJointWorldMatrix = jointWorldMatrix;

			localRotation = calcLocalRotation(rotation[8], translation[8]);
			jointWorldMatrix = localRotation*lastJointWorldMatrix;
			posingMatrix[8] = bindingMatrixInverse[8] * jointWorldMatrix;
			lastJointWorldMatrix = jointWorldMatrix;


			// Calculate a joint's local rotation matrix
			vec4 vertex = vec4(0., 0., 0., 0.0);
			int index;			
			for (int i=0; i < 3; ++i) {
				index = int(joint_indices[i]);
				vertex += joint_weights[i] * gl_Vertex * mat_at_i(posingMatrix, index);
			}
			vertex.xyz *= scale;

			vertex[0] = vertex[0]+offsetx;
			vertex[2] = vertex[2]+offsety;

			// Transform vertex by modelview and projection matrices
			gl_Position = gl_ModelViewProjectionMatrix * vertex;

			// Pass on the vertex color 
			float the_color = (vertex[1]/(7.0*scale))*0.8 + 0.2;
			vertex_color = vec4(the_color, the_color, the_color, 1.0);

		}

		""", GL_VERTEX_SHADER)
		
		fragmentShader = shaders.compileShader("""
		#version 120
		varying vec4 vertex_color;

		void main() {
			gl_FragColor = vertex_color;
		}

		""", GL_FRAGMENT_SHADER)

		self.shaderProgram = shaders.compileProgram(vertexShader, fragmentShader)

		# Now, let's make sure our uniform and attribute value value will be sent 
		# for uniform in ['joints', 'scale', 'offsetx', 'offsety', 'rotation', 'translation', 'bindingMatrixInverse']:
		for uniform in ['scale', 'offsetx', 'offsety', 'rotation', 'translation', 'bindingMatrixInverse']:
			location = glGetUniformLocation(self.shaderProgram, uniform)
			name = uniform+"_location"
			setattr(self, uniform+"_location", location)
		for attribute in ['joint_weights', 'joint_indices']:
			location = glGetAttribLocation(self.shaderProgram, attribute)
			setattr(self, attribute+"_location", location)

		# There's a couple uniforms that never change
		# For now, the inverse binding matrix, and the joint translations
		# (the skeleton morphology does not change, and joints only rotate, 
		#	they don't slide)
		glUseProgram(self.shaderProgram)
		joints = self.skin.jointChain.joints
		Bi = np.array([np.array(j.Bi.copy()) for j in joints]).astype('float32')
		glUniformMatrix4fv(self.bindingMatrixInverse_location, self.num_bones, True, Bi)
		translations = np.array([j.translation.copy() for j in joints]).astype('float32')
		glUniform3fv(self.translation_location, self.num_bones, translations)
		glUseProgram(0)


	def setup_texture(self):
		import Image

		f = np.load(os.path.join(os.path.dirname(__file__),"data/meanmouse.npz"))
		self.mouse_img = f['mouse_img'].astype('float32')
		I = Image.fromarray(self.mouse_img)
		self.mouse_img = np.array(I.resize((self.mouse_width, self.mouse_height)))
		width,height = self.mouse_img.shape
		img_for_texture = self.mouse_img[:,:].ravel()
		img_for_texture = np.repeat(img_for_texture, 3)

		self.texture_id = glGenTextures(1)
		glBindTexture(GL_TEXTURE_2D, self.texture_id)
		glPixelStoref(GL_UNPACK_ALIGNMENT, 1)
		glTexImage2D(GL_TEXTURE_2D, 0, 3, width, height, 0, GL_RGB, GL_FLOAT, img_for_texture)

	def setup_fbo(self):
		self.frameBuffer = glGenFramebuffers(1)
		glBindFramebuffer(GL_FRAMEBUFFER, self.frameBuffer)
		self.renderBuffer = glGenRenderbuffers(1)
		glBindRenderbuffer(GL_RENDERBUFFER, self.renderBuffer)
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, self.width, self.height)
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.renderBuffer)

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
		if not self.useFramebuffer:
			glutIdleFunc(self.display)
		
		# States to set
		glEnable(GL_DEPTH_TEST)
		glEnable(GL_NORMALIZE)
		glEnable(GL_BLEND)
		glEnable(GL_LINE_SMOOTH)
		glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
		glLineWidth(3.5)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

		# Setup our VBOs and shaders
		self.setup_vbos()
		self.update_vertex_mesh()
		self.setup_shaders()
		self.setup_texture()
		if self.useFramebuffer:
			self.setup_fbo()

	def get_likelihood(self, new_img, x, y, theta, particle_data, return_posed_mice=False):
		"""Calculate the likelihood of a list of particles given a mouse mouse_image

		particle_data - num_particles x num_variables
		mouse_image - the current mouse image
		x 			- the x position of the mouse
		y 			- the y position of the mouse
		theta		- the theta angle of the mouse

		mousescene - an instance of MouseScene, which controls the rendering
		likelihood_array (optional) - num_particles array 
									(provide if you don't want a memory copy)
		"""

		# Check the mouse image size
		assert new_img.shape == self.mouse_img.shape, \
					"New image must be shape of old image (%d, %d)" % (self.mouse_width, self.mouse_height)
		self.mouse_img =  new_img

		# Check the number of particles (must be a multiple of num_mice)
		num_particles, num_vars = particle_data.shape
		assert np.mod(num_particles, self.num_mice) == 0, \
					"Number of particles must be a multiple of the number of mice (%d)" % (self.num_mice)

		# If we have more particles than mice, 
		# we'll have to do multiple rendering passes to get all the likelihoods
		num_passes = int(num_particles / self.num_mice)


		# Here we extract the parameters from the particle_data, 
		# as we think they should be sitting.
		# So, right now, that's 
		# - offsetx
		# - offsety
		# - body angle
		# and for each of 9 joints,
		# {
		# 	- vertical rotation from rest
		#	- horizontal rotation from rest
		# }
		all_likelihoods = np.zeros((num_particles,), dtype='float32')
		if return_posed_mice:
			posed_mice = np.zeros((num_particles, self.mouse_height, self.mouse_width), dtype='float32')

		for i in range(num_passes):

			# Slice out our current particles to render
			start = i*self.num_mice
			end = start+self.num_mice
			this_particle_data = particle_data[start:end]

			# Set the position and angle offsets
			self.offset_x = this_particle_data[:,0] - x
			self.offset_y = this_particle_data[:,1] - y
			self.offset_theta = this_particle_data[:,2] - theta

			# Set the joint rotations
			rotations = this_particle_data[:,3:]
			rotations = np.reshape(rotations, (self.num_mice, -1, 3))
			self.rotations = rotations

			# Display the mouse scene (or render to a framebuffer, alternatively)
			self.display()

			# Grab the computed likelihoods
			all_likelihoods[start:end] = self.likelihood.ravel()
			if return_posed_mice:
				posed_mice[start:end] = self.posed_mice.copy()
		
		if return_posed_mice:
			return all_likelihoods, posed_mice
		else:
			return all_likelihoods

def test_single_mouse():
	from load_data import load_behavior_data

	path_to_behavior_data = "/Users/Alex/Dropbox/Science/Datta lab/Posture Tracking/Test Data"
	which_img = 731
	
	image = load_behavior_data(path_to_behavior_data, which_img+1, 'images')[-1]
	image = image.T[::-1,:].astype('float32')
	image /= 354.0;

	num_particles = 32**2
	numCols = 16
	numRows = 16
	scenefile = "data/mouse_mesh_low_poly.npz"

	useFramebuffer = True
	ms = MouseScene(scenefile, mouse_width=80, mouse_height=80, \
								scale = 2.0, \
								numCols=numCols, numRows=numRows, useFramebuffer=True)
	ms.gl_init()

	# Get the base, unposed, rotations
	rot = ms.get_joint_rotations().copy()
	# Tile it so it's the same size as the number of particles
	num_passes = int(num_particles / ms.num_mice)
	rot = np.tile(rot, (num_passes, 1, 1))

	# Fill in some particle data
	particle_data = np.zeros((num_particles, 3+9*3))
	particle_data[:,3::3] = rot[:,:,0]
	particle_data[:,4::3] = rot[:,:,1]
	particle_data[:,5::3] = rot[:,:,2]

	# Set the offsets
	position_val = 0
	particle_data[:,:2] = position_val + np.random.normal(scale=1, size=(num_particles, 2))

	# Set the angles
	theta_val = 0
	particle_data[:,2] = theta_val + np.random.normal(scale=1, size=(num_particles,))


	particle_data[1:,7::3] += np.random.normal(scale=10, size=(num_particles-1, ms.num_bones-1))
	particle_data[1:,8::3] += np.random.normal(scale=10, size=(num_particles-1, ms.num_bones-1))

	likelihoods, posed_mice = ms.get_likelihood(image, \
						x=position_val, y=position_val, \
						theta=theta_val, \
						particle_data=particle_data,
						return_posed_mice=True)

	# L = ms.likelihood.T.ravel()
	particle_rotations = np.hstack((particle_data[:,4::3], particle_data[:,5::3]))
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
	# savefig("/Users/Alex/Desktop/angle vs likelihood.png")
	# figure(); imshow(ms.likelihood)
	# figure(); imshow(ms.data); colorbar()
	# figure(); imshow(ms.diffmap); colorbar()


	# Find the five best mice
	idx = np.argsort(likelihoods)
	fivebest = np.hstack(posed_mice[idx[-5:]])
	# Show first the raw mouse, then my hand-posed mouse, and then the five best poses
	
	figure()
	title("Five best (best, far right)")
	imshow(np.hstack((ms.mouse_img, posed_mice[0], fivebest)))
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

	return ms, rotation_diffs, likelihoods, particle_data


if __name__ == '__main__':
	
	useFramebuffer = False
	if not useFramebuffer:
		scenefile = "data/mouse_mesh_low_poly.npz"
		ms = MouseScene(scenefile, mouse_width=80, mouse_height=80, \
									scale = 2.1, \
									numCols=32, numRows=32, useFramebuffer=useFramebuffer)
		ms.gl_init()
		glutMainLoop()
	else:
		test_single_mouse()
		# for i in range(10):
		# 	old_rotations = np.copy(ms.rotations)
		# 	ms.rotations[:,:,1:] += np.random.normal(scale=10,
		# 										size=(self.num_mice, self.num_bones, 2))
		# 	ms.display()
		# 	ms.rotations = old_rotations
