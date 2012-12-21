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
from pylab import *

sys.path.append("/Users/Alex/Dropbox/Science/Datta lab/Posture Tracking/")
import transformations as tr

import time

sys.path.append("/Users/Alex/Dropbox/Science/Datta lab/Posture Tracking/Playground")
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
		self.data[:,:3] = self.vertices[:,:3]
		self.mesh_vbo[:] = self.data

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
		data = np.zeros((self.num_vertices, num_elements_per_coord), dtype='float32')

		# Calculate the indices of the non-zero joint weights
		joint_idx = np.zeros((self.num_vertices, self.num_joint_influences), dtype='int')
		nonzero_joint_weights = np.zeros((self.num_vertices, self.num_joint_influences), dtype='float32')
		for i in range(self.num_vertices):
			joint_idx[i,:] = np.argwhere(self.skin.joint_weights[i,:] > 0).ravel()
			nonzero_joint_weights[i,:] = self.skin.joint_weights[i,joint_idx[i,:]]

		self.data = np.hstack((self.vertices[:,:3], nonzero_joint_weights, joint_idx)).astype('float32')
		self.mesh_vbo = vbo.VBO(self.data)



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
		# self.update_vertex_mesh()

		# Bind our VBOs		
		self.mesh_vbo.bind()
		self.index_vbo.bind()

		# Bind our texture
		# glEnable(GL_TEXTURE_2D)
		# glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
		# glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
		# glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
		# glBindTexture(GL_TEXTURE_2D, self.texture_id)


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
		# glVertexAttribPointer(self.position_location,
		# 				3, GL_FLOAT, 
		# 				False, stride, self.mesh_vbo)
		glVertexAttribPointer(self.joint_weights_location,
						4, GL_FLOAT, 
						False, stride, self.mesh_vbo+3*4)
		glVertexAttribPointer(self.joint_indices_location,
						4, GL_FLOAT, 
						False, stride, self.mesh_vbo+3*4+self.num_joint_influences*4)

		x = self.mouse_width*np.mod(np.arange(numCols*numRows),numCols)
		y = self.mouse_height*np.floor_divide(np.arange(numCols*numRows), numCols)
		scale_array = np.repeat(self.scale, numCols*numRows, axis=0)

		jointBindingMatrix = []
		for i in range(numCols*numRows):
			ajoint = np.random.randint(1,self.num_bones)
			oldrotation = self.skin.jointChain.joints[ajoint].rotation.copy()
			horz_deg = np.random.normal()*30.
			vert_deg = np.random.normal()*30.
			self.skin.jointChain.joints[ajoint].rotation[1] += horz_deg
			self.skin.jointChain.joints[ajoint].rotation[2] += vert_deg
			self.skin.jointChain.solve_forward(ajoint)
			this_b = np.array([np.array(j.M.copy()) for j in self.skin.jointChain.joints]).astype('float32')
			jointBindingMatrix.append(this_b)
			self.skin.jointChain.joints[ajoint].rotation[1] -= horz_deg
			self.skin.jointChain.joints[ajoint].rotation[2] -= vert_deg
			self.skin.jointChain.solve_forward(ajoint)


		for i in range(numCols*numRows):
			glUniform1f(self.offsetx_location, x[i])
			glUniform1f(self.offsety_location, y[i])
			glUniform1f(self.scale_location, scale_array[i])
			glUniform2f(self.mouse_img_size_location, self.mouse_width, self.mouse_height)
			glUniformMatrix4fv(self.joints_location,self.num_bones, True, jointBindingMatrix[i])
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

		# Put that frame on the screen
		# glutSwapBuffers()


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
		np.savez("/Users/Alex/Desktop/frame.npz", \
						frame=data, \
						diffmap=this_diff, \
						likelihood = likelihood,
						mouse_img = self.mouse_img)


		glBindFramebuffer(GL_FRAMEBUFFER, 0)

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
		uniform mat4 joints[9]; // currently have 9 bones
		attribute vec4 joint_weights;
		attribute vec4 joint_indices;

		uniform sampler2D mouse_texture;
		void main()
		{	
			vec4 vertex;
			int index;

			vertex = vec4(0., 0., 0., 0.0);
			
			index = int(joint_indices[0]);
			vertex += joint_weights[0] * gl_Vertex * joints[index];
			index = int(joint_indices[1]);
			vertex += joint_weights[1] * gl_Vertex * joints[index];
			index = int(joint_indices[2]);
			vertex += joint_weights[2] * gl_Vertex * joints[index];

			vertex.xyz *= scale;

			vertex[0] = vertex[0]+offsetx;
			vertex[2] = vertex[2]+offsety;

			// Transform vertex by modelview and projection matrices
			gl_Position = gl_ModelViewProjectionMatrix * vertex.xyzw;

			// Pass on the vertex color 
			float the_color = (vertex[1]/(7.0*scale))*0.8 + 0.2;
			// vertex_color = vec4(joint_weights.wzy, 1.0);
			vertex_color = vec4(the_color, the_color, the_color, 1.0);

		}

		""" % self.num_joint_influences, GL_VERTEX_SHADER)
		
		fragmentShader = shaders.compileShader("""
		varying vec4 vertex_color;
		uniform sampler2D mouse_tex;
		uniform vec2 mouse_img_size;

		void main() {
			vec2 position = gl_FragCoord.xy / mouse_img_size.xy;
			gl_FragColor = vertex_color;// + texture2D(mouse_tex, position);
		}

		""", GL_FRAGMENT_SHADER)

		self.shaderProgram = shaders.compileProgram(vertexShader, fragmentShader)

		# Now, let's make sure our uniform value will be sent 
		self.scale_location = glGetUniformLocation(self.shaderProgram, 'scale')
		self.offsetx_location = glGetUniformLocation(self.shaderProgram, 'offsetx')
		self.offsety_location = glGetUniformLocation(self.shaderProgram, 'offsety')
		self.joint_weights_location = glGetAttribLocation(self.shaderProgram, 'joint_weights')
		self.joint_indices_location = glGetAttribLocation(self.shaderProgram, 'joint_indices')
		self.joints_location = glGetUniformLocation(self.shaderProgram, "joints")
		self.mouse_img_size_location = glGetUniformLocation(self.shaderProgram, "mouse_img_size")

	def setup_texture(self):
		f = np.load("/Users/Alex/Dropbox/Science/Datta lab/Posture Tracking/meanmouse.npz")
		self.mouse_img = f['mouse_img'].astype('float32')
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

		# self.img = glGenTextures(1)
		# glBindTexture(GL_TEXTURE_2D, self.img)
		# glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
		# glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
		# glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, self.width, self.height, 0, GL_RGB, GL_FLOAT, None)

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
		# glutMotionFunc(self.on_motion)
		glutDisplayFunc(self.display)
		# glutReshapeFunc(self.on_reshape)
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


if __name__ == '__main__':
	scenefile = "/Users/Alex/Dropbox/Science/Datta lab/Posture Tracking/Models/mouse_mesh_low_poly.npz"
	useFramebuffer = False
	ms = MouseScene(scenefile, mouse_width=80, mouse_height=80, \
								scale = 2.5, \
								numCols=10, numRows=10, useFramebuffer=useFramebuffer)
	ms.gl_init()
	
	if not useFramebuffer:
		glutMainLoop()
	else:
		for i in range(10):
			ms.display()
