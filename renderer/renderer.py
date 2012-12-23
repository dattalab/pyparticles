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

class MouseScene(object):
	"""A class containing methods for drawing a skinned polygon mesh
	as quickly as possible."""
	def __init__(self, scenefile, scale = 4.0, \
						mouse_width=640, mouse_height=480, \
						numCols=1, numRows=1, useFramebuffer=False):

		"""For a given scenefile (the output of 
			get_poly_and_skin_info_maya.py), 
			Display the polygon and the joint positions.
		"""
		super(MouseScene, self).__init__()
		self.mouse_width = mouse_width
		self.mouse_height = mouse_height
		self.numCols = numCols
		self.numRows = numRows
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


		# Make sure we have a completely updated mesh
		self.update_vertex_mesh()

		# Bind our VBOs		
		self.mesh_vbo.bind()
		self.index_vbo.bind()

		# Turn on our shaders
		glUseProgram(self.shaderProgram)

		# Draw the poly mesh
		# ==============================

		# glEnableVertexAttribArray(self.position_location)
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

		x = self.mouse_width*np.mod(np.arange(numCols*numRows),numCols)
		y = self.mouse_height*np.floor_divide(np.arange(numCols*numRows), numCols)
		scale_array = np.repeat(self.scale, numCols*numRows, axis=0)

		joints = self.skin.jointChain.joints

		self.rotations = np.array([j.rotation.copy() for j in joints]).astype('float32')
		self.translations = np.array([j.translation.copy() for j in joints]).astype('float32')
		self.rotations = np.zeros_like(self.rotations)
		self.translations = np.zeros_like(self.translations)

		Bi = np.array([np.array(j.Bi.copy()) for j in joints]).astype('float32')
		glUniformMatrix4fv(self.bindingMatrixInverse_location, self.num_bones, True, Bi)

		jointBindingMatrix = []
		for i in range(numCols*numRows):
			ajoint = np.random.randint(1,self.num_bones)
			# oldrotation = self.skin.jointChain.joints[ajoint].rotation.copy()
			# horz_deg = np.random.normal()*30.
			# vert_deg = np.random.normal()*30.
			# self.skin.jointChain.joints[ajoint].rotation[1] += horz_deg
			# self.skin.jointChain.joints[ajoint].rotation[2] += vert_deg
			self.skin.jointChain.solve_forward(ajoint)
			this_b = np.array([np.array(j.M.copy()) for j in self.skin.jointChain.joints]).astype('float32')
			jointBindingMatrix.append(this_b)
			# self.skin.jointChain.joints[ajoint].rotation[1] -= horz_deg
			# self.skin.jointChain.joints[ajoint].rotation[2] -= vert_deg
			# self.skin.jointChain.solve_forward(ajoint)

		for i in range(numCols*numRows):
			glUniform1f(self.offsetx_location, x[i])
			glUniform1f(self.offsety_location, y[i])
			glUniform1f(self.scale_location, scale_array[i])
			glUniform3fv(self.rotation_location, self.num_bones, self.rotations)
			glUniform3fv(self.translation_location, self.num_bones, self.translations)
			glUniformMatrix4fv(self.joints_location, self.num_bones, True, jointBindingMatrix[i])
			glDrawElements(GL_TRIANGLES, self.num_indices, GL_UNSIGNED_SHORT, self.index_vbo)

		self.mesh_vbo.unbind()
		self.index_vbo.unbind()
		glDisableClientState(GL_VERTEX_ARRAY)
		# ==============================

		# Turn off the texture

		# Turn off our shaders
		glUseProgram(0)

		if not self.useFramebuffer:
			glutSwapBuffers()

		# Uncomment this if you want to read the data off of the card
		data = glReadPixels(0,0,self.width,self.height, GL_RGB, GL_FLOAT)
		data = data.ravel().reshape((self.height, self.width, 3))[:,:,0]
		this_diff = (np.tile(self.mouse_img, (self.numRows, self.numCols)) - data)**2.0
		likelihood = np.zeros((self.numRows, self.numCols), dtype='float32')
		for i in range(self.numRows):
			startr = i*self.mouse_height
			endr = startr+self.mouse_height
			for j in range(self.numCols):
				startc = j*self.mouse_width
				endc = startc+self.mouse_width
				likelihood[i,j] = this_diff[startr:endr,startc:endc].sum()
				
		self.data = data
		self.likelihood = likelihood
		self.diffmap = this_diff
		# np.savez("data/frame.npz", \
		# 				frame=data, \
		# 				diffmap=this_diff, \
		# 				likelihood = likelihood,
		# 				mouse_img = self.mouse_img)


		if self.useFramebuffer:
			glBindFramebuffer(GL_FRAMEBUFFER, 0)


	def setup_shaders(self):
		if not glUseProgram:
			print 'Missing Shader Objects!'
			sys.exit(1)

		vertexShader = shaders.compileShader("""
		// Application to vertex shader
		varying vec4 vertex_color;
		uniform float offsetx;
		uniform float offsety;
		uniform float scale;

		uniform vec3 rotation[9]; // rotations on each joint
		uniform vec3 translation[9]; // translations of each joint from the previous
		uniform mat4 joints[9]; // currently have 9 bones
		uniform mat4 bindingMatrixInverse[9]; // the inverse binding matrix
		attribute vec4 joint_weights;
		attribute vec4 joint_indices;

		mat4 calcLocalRotation(in vec3 rotation, in vec3 translation) {

			rotation = radians(-rotation);
			vec3 cosrot = cos(rotation);
			vec3 sinrot = sin(rotation);
			mat3 Rx = mat3(1.0);
			mat3 Ry = mat3(1.0);
			mat3 Rz = mat3(1.0);

			Rx[1].y = cosrot.x;
			Rx[1].z = -sinrot.x;
			Rx[2].y = sinrot.x;
			Rx[2].z = cosrot.x;

			Ry[0].x = cosrot.y;
			Ry[0].z = sinrot.y;
			Ry[2].x = -sinrot.y;
			Ry[2].z = cosrot.y;

			Rz[0].x = cosrot.z;
			Rz[0].y = -sinrot.z;
			Rz[1].x = sinrot.z;
			Rz[1].y = cosrot.z;

			mat3 T = Rz*Ry*Rx;
			mat4 Tout = mat4(1.0);
			Tout[0].xyz = T[0].xyz;
			Tout[1].xyz = T[1].xyz;
			Tout[2].xyz = T[2].xyz;
			Tout[0].w = translation.x;
			Tout[1].w = translation.y;
			Tout[2].w = translation.z;
			// Tout[3].xyz = translation.xyz;
			return Tout;
		}
		
		void main()
		{	
			mat4 lastJointWorldMatrix;
			mat4 jointWorldMatrix;
			mat4 posingMatrix[9];
			mat4 localRotation;
			vec4 vertex = vec4(0.0);
			int index;
			
			// For each joint
			// 1. calculate its local rotation matrix
			// 2. calculate its world position from the previous joint's world matrix
			// 3. multiply its inverse binding matrix into its world matrix as the posing matrix
			// 4. multiply the posing matrix into the vertex
			
			lastJointWorldMatrix = mat4(1.0);
			for (int i=0; i < 9; ++i) {
				// Calculate a joint's local rotation matrix
				// localRotation = calcLocalRotation(rotation[i], translation[i]);
				localRotation = calcLocalRotation(vec3(0.0), vec3(0.0));

				// Calculate its world position
				jointWorldMatrix = localRotation*lastJointWorldMatrix;
				
				// Multiply the inverse binding matrix into the world matrix
				// posingMatrix[i] = bindingMatrixInverse[i] * jointWorldMatrix;
				posingMatrix[i] = mat4(1.0);

				// Get ready for the next iteration
				lastJointWorldMatrix = jointWorldMatrix;
			
			}

			for (int i=0; i < 9; ++i) {
				posingMatrix[i] = mat4(1.0);
			}


			// Calculate a joint's local rotation matrix
			mat4 blah[9];
			for (int i=0; i < 9; ++i) {
				blah[i][0].xyzw = vec4(1.0, 0.0, 0.0, 0.0);
				blah[i][1].xyzw = vec4(0.0, 1.0, 0.0, 0.0);
				blah[i][2].xyzw = vec4(0.0, 0.0, 1.0, 0.0);
				blah[i][3].xyzw = vec4(0.0, 0.0, 0.0, 1.0);
			}
			
			for (int i=0; i < 3; ++i) {
				index = int(joint_indices[i]);
				mat4 whatever = blah[i];
				vertex += joint_weights[i] * gl_Vertex * whatever;
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

		""" % self.num_joint_influences, GL_VERTEX_SHADER)
		
		fragmentShader = shaders.compileShader("""
		varying vec4 vertex_color;

		void main() {
			gl_FragColor = vertex_color;
		}

		""", GL_FRAGMENT_SHADER)

		self.shaderProgram = shaders.compileProgram(vertexShader, fragmentShader)

		# Now, let's make sure our uniform and attribute value value will be sent 
		for uniform in ['joints', 'scale', 'offsetx', 'offsety', 'rotation', 'translation', 'bindingMatrixInverse']:
			location = glGetUniformLocation(self.shaderProgram, uniform)
			name = uniform+"_location"
			setattr(self, uniform+"_location", location)
		for attribute in ['joint_weights', 'joint_indices']:
			location = glGetAttribLocation(self.shaderProgram, attribute)
			setattr(self, attribute+"_location", location)

		# Uploading the binding matrices
		glUseProgram(self.shaderProgram)
		joints = self.skin.jointChain.joints
		Bi = np.array([np.array(j.Bi.copy()) for j in joints]).astype('float32')
		glUniformMatrix4fv(self.bindingMatrixInverse_location, self.num_bones, True, Bi)

		# Uploading joint translations (these stay fixed)
		self.translations = np.array([j.translation.copy() for j in joints]).astype('float32')
		glUniform3fv(self.translation_location, self.num_bones, self.translations)

		glUseProgram(0)


	def setup_texture(self):
		import Image

		f = np.load("data/meanmouse.npz")
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



def get_likelihood(particle_data, mouse_image, mousescene, likelihood_array=None):
	"""Calculate the likelihood of a list of particles given a mouse mouse_image

	particle_data - num_particles x num_variables
	mouse_image - the current mouse image
	mousescene - an instance of MouseScene, which controls the rendering
	likelihood_array (optional) - num_particles array 
								(provide if you don't want a memory copy)
	"""

	num_particles, num_vars = particle_data.shape
	width, height = mouse_image.shape

	if likelihood_array == None:
		likelihood_array = np.zeros((num_particles,), dtype='float32')

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

	offsetx, offsety = particle_data[:,0], particle_data[:,1]
	body_angle = particle_data[:,2]

	mousescene.rotations = particle_data[:,]



if __name__ == '__main__':
	scenefile = "data/mouse_mesh_low_poly.npz"
	useFramebuffer = False
	ms = MouseScene(scenefile, mouse_width=300, mouse_height=300, \
								scale = 10, \
								numCols=1, numRows=1, useFramebuffer=useFramebuffer)
	ms.gl_init()
	
	if not useFramebuffer:
		glutMainLoop()
	else:
		for i in range(10):
			ms.display()
