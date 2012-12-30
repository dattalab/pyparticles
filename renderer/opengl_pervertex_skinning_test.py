import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GLUT.freeglut import *
from OpenGL.GL import shaders
from OpenGL.arrays import vbo
# from OpenGL.GL.ARB.draw_instanced import *

sys.path.append("/Users/Alex/Dropbox/Science/Datta lab/Posture Tracking/")
import transformations as tr

import time

sys.path.append("/Users/Alex/Dropbox/Science/Datta lab/Posture Tracking/Playground")
import Joints

import Image

class MouseScene(object):
	"""A class containing methods for drawing a skinned polygon mesh
	as quickly as possible."""
	def __init__(self, scenefile, scale = 4.0, \
						mouse_width=640, mouse_height=480, \
						numCols=1, numRows=1):

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
		self.num_joint_influences =  int((joint_weights>0).sum(1).max())
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
		self.index_vbo = vbo.VBO(vidx, target=GL_ELEMENT_ARRAY_BUFFER, usage=GL_DYNAMIC_DRAW)

		self.vertices = self.skin.get_posed_vertices()[:,:3].astype('float32') # leave off the scale parameter

		# vertices: x,y,z
		# vertex weights: per-bone weight
		# joint index: which joint each weight correspond to

		# Calculate the indices of the non-zero joint weights
		joint_idx = np.zeros((self.num_vertices, self.num_joint_influences), dtype='int')
		nonzero_joint_weights = np.zeros((self.num_vertices, self.num_joint_influences), dtype='float32')
		for i in range(self.num_vertices):
			joint_idx[i,:] = np.argwhere(self.skin.joint_weights[i,:] > 0).ravel()
			nonzero_joint_weights[i,:] = self.skin.joint_weights[i,joint_idx[i,:]]

		self.data = np.hstack((self.vertices[:,:3], nonzero_joint_weights, joint_idx)).astype('float32')
		self.mesh_vbo = vbo.VBO(self.data, usage=GL_DYNAMIC_DRAW)



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
		# self.skin.jointChain.solve_forward(0)
		# self.jointPoints = self.skin.jointChain.get_joint_world_positions()

	def on_reshape(self, this_width, this_height):
		pass


	def on_display(self):

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

		# Now, rotate to draw the mouse model
		glRotate(-90, 1., 0., 0.)

		# Turn on our shaders
		glUseProgram(self.shaderProgram)

		# Draw the poly mesh
		# ==============================
		glEnableVertexAttribArray(self.joint_weights_location)
		glEnableVertexAttribArray(self.joint_indices_location)

		stride = (3 + self.num_joint_influences*2)*4
		weights_offset = 3*4 # offset by the x,y,z coordinates
		indices_offset = 3*4 + self.num_joint_influences*4 # offset by x,y,z and w1,w2,w3,...
		glEnableClientState(GL_VERTEX_ARRAY)
		glVertexPointer(3, GL_FLOAT, stride, self.mesh_vbo)
		glVertexAttribPointer(self.joint_weights_location,
						self.num_joint_influences, GL_FLOAT, 
						False, stride, self.mesh_vbo+weights_offset)
		glVertexAttribPointer(self.joint_indices_location,
						self.num_joint_influences, GL_FLOAT, 
						False, stride, self.mesh_vbo+indices_offset)

		x = self.mouse_width*np.mod(np.arange(numCols*numRows),numCols)
		y = self.mouse_height*np.floor_divide(np.arange(numCols*numRows), numCols)
		scale_array = np.repeat(self.scale, numCols*numRows, axis=0)

		joints = self.skin.jointChain.joints
		rotations = np.array([j.rotation.copy() for j in joints]).astype('float32')
		translations = np.array([j.translation.copy() for j in joints]).astype('float32')

		for i in range(numCols*numRows):
			glUniform1f(self.offsetx_location, x[i])
			glUniform1f(self.offsety_location, y[i])
			glUniform1f(self.scale_location, scale_array[i])
			glUniform3fv(self.rotation_location, self.num_bones, rotations)
			glUniform3fv(self.translation_location, self.num_bones, translations)
			glDrawElements(GL_TRIANGLES, self.num_indices, GL_UNSIGNED_SHORT, self.index_vbo)
		# glDrawElementsInstancedARB(GL_TRIANGLES, self.num_indices, GL_UNSIGNED_SHORT, self.index_vbo, self.numCols*self.numRows)

		# Turn off the texture right afterwards
		# glDisableClientState(GL_VERTEX_ARRAY)
		# ==============================

		# Turn off our shaders
		glUseProgram(0)
		
		# Put it all on the screen
		glutSwapBuffers()


		# Uncomment this if you want to read the data off of the card
		data = glReadPixels(0,0,self.width,self.height, GL_RED, GL_FLOAT)
		data = data.ravel().reshape((self.height, self.width, 1))[:,:,0]
		this_diff = (np.tile(self.mouse_img, (self.numRows, self.numCols)) - data)**2.0
		likelihood = np.zeros((self.numRows, self.numCols), dtype='float32')
		for i in range(self.numRows):
			startr = i*self.mouse_height
			endr = startr+self.mouse_height
			for j in range(self.numCols):
				startc = j*self.mouse_width
				endc = startc+self.mouse_width
				likelihood[i,j] = this_diff[startr:endr,startc:endc].sum()
		# np.savez("/Users/Alex/Desktop/frame.npz", \
		# 				frame=data, \
		# 				diffmap=this_diff, \
		# 				likelihood = likelihood,
		# 				mouse_img = self.mouse_img)

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
		for uniform in ['scale', 'offsetx', 'offsety', 'rotation', 'translation', 'bindingMatrixInverse']:
			location = glGetUniformLocation(self.shaderProgram, uniform)
			name = uniform+"_location"
			setattr(self, uniform+"_location", location)
		for attribute in ['joint_weights', 'joint_indices']:
			location = glGetAttribLocation(self.shaderProgram, attribute)
			setattr(self, attribute+"_location", location)

		glUseProgram(self.shaderProgram)
		joints = self.skin.jointChain.joints
		Bi = np.array([np.array(j.Bi.copy()) for j in joints]).astype('float32')
		glUniformMatrix4fv(self.bindingMatrixInverse_location, self.num_bones, True, Bi)
		glUseProgram(0)

	def setup_texture(self):
		f = np.load("/Users/Alex/Dropbox/Science/Datta lab/Posture Tracking/meanmouse.npz")
		self.mouse_img = f['mouse_img'].astype('float32')
		self.mouse_img = np.array(Image.fromarray(self.mouse_img).resize((self.mouse_width, self.mouse_height)))

		width,height = self.mouse_img.shape
		img_for_texture = self.mouse_img[:,:].ravel()
		img_for_texture = np.repeat(img_for_texture, 3)
		self.texture_id = glGenTextures(1)
		glBindTexture(GL_TEXTURE_2D, self.texture_id)
		glPixelStoref(GL_UNPACK_ALIGNMENT, 1)
		glTexImage2D(GL_TEXTURE_2D, 0, 3, width, height, 0, GL_RGB, GL_FLOAT, img_for_texture)


	def gl_init(self):

		glutInit([])
		glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
		glutInitWindowSize(self.width, self.height)
		glutCreateWindow('Mouse Model')

		glutKeyboardFunc(self.on_keypress)
		glutMotionFunc(self.on_motion)
		glutDisplayFunc(self.on_display)
		glutReshapeFunc(self.on_reshape)
		glutIdleFunc(self.on_display)
		
		# States to set
		glEnable(GL_DEPTH_TEST)
		glEnable(GL_AUTO_NORMAL)

		glEnable(GL_NORMALIZE)
		glEnable(GL_BLEND)
		glEnable(GL_LINE_SMOOTH)
		glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
		glLineWidth(3.5)
		glEnable(GL_POLYGON_SMOOTH);
		glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

		# Setup our VBOs and shaders
		self.setup_vbos()
		self.update_vertex_mesh()
		self.setup_shaders()
		self.setup_texture()

		# Bind our VBOs		
		self.mesh_vbo.bind()
		self.index_vbo.bind()



if __name__ == '__main__':
	scenefile = "/Users/Alex/Dropbox/Science/Datta lab/Posture Tracking/Models/mouse_mesh_low_poly.npz"
	ms = MouseScene(scenefile, mouse_width=80, mouse_height=80, \
								scale = 2.0, \
								numCols=16, numRows=16)
	ms.gl_init()
	import cProfile
	cProfile.run("glutMainLoop()", "prof")
	import pstats
	p = pstats.Stats('prof')
	p.sort_stats('cumulative').print_stats(10)

	# glutMainLoop()