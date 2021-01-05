#========================================================#
#     Author: Imoleayo Abel                              #
#     Course: Sensing and Estimation Robotics (ECE 276A) #
#    Quarter: Fall 2017	                                 #
# Instructor: Nikolay Atanasov                           #
#    Project: 02 - Orientation Tracking                  #
#       File: myquaternion.py                            #
#       Date: Nov-22-2017                                #
#                                                        #
#       #=======================================#        #
#       #  - Quaternions are (4,) numpy arrays  #        #
#       #  - Axes are (3,1) numpy matrices      #        #
#       #  - Matrices are (3,3) numpy matrices  #        #
#       #=======================================#        #
#                                                        #
#========================================================#
import numpy as np
from decimal import Decimal

#========================================================#
# Multiply two quaternions                               #
#                                                        #
#  Input: two quaternions: p, and q                      #
# Output: quaternion corresponding to product of p and q #
#========================================================#
def qMult(q,p):
	qs, qv = q[0], q[1:]
	ps, pv = p[0], p[1:]
	return np.concatenate(([qs*ps - qv.dot(pv)], qs*pv + ps*qv + np.cross(qv,pv)))

#=======================================#
# Get complex conjugate of a quaternion #
#                                       #
#  Input: quaternion q                  #
# Output: conjugate of q                #
#=======================================#
def qConj(q):
	return np.concatenate((q[:1], -q[1:]))

#==========================#
# Get norm of a quaternion #
#                          #
#  Input: quaternion q     #
# Output: norm of q        #
#==========================#
def qNorm(q):
	return np.sqrt(q.dot(q))

#=============================#
# Get inverse of a quaternion #
#                             #
#  Input: quaternion q        #
# Output: inverse of q        #
#=============================#
def qInv(q):
	sqr_norm = q.dot(q)
	if sqr_norm == 0: return q
	return qConj(q)/(1.0*sqr_norm)

#=================================#
# Rotate a vector by a quaternion #
#                                 #
#  Input: quaternion q            #
#         vector v                #
# Output: rotated version of v    #
#=================================#
def qRotate(q,v):
	q_norm = qNorm(q)
	if q_norm == 0: return v
	q = q/q_norm
	return qMult(q, qMult(np.concatenate(([0], v.reshape((3,)))),qInv(q)))[1:].reshape(v.shape)

#==============================#
# Get exponent of a quaternion #
#                              #
#  Input: quaternion q         #
# Output: Exponent of q        #
#==============================#
def qExp(q):
	qs, qv = q[0], q[1:]
	qv_norm = qNorm(qv)
	if qv_norm == 0:
		return np.array([np.exp(qs)*np.cos(qv_norm), 0, 0, 0])
	return np.exp(qs) * np.concatenate(([np.cos(qv_norm)], (np.sin(qv_norm)/(1.0*qv_norm))*qv))

#===============================#
# Get logarithm of a quaternion #
#                               #
#  Input: quaternion q          #
# Output: Logarithm of q        #
#===============================#
def qLog(q):
	qs, qv = q[0], q[1:]
	q_norm, qv_norm = qNorm(q), qNorm(qv)
	if qv_norm == 0:
		return np.array([np.log(q_norm), 0, 0, 0])
	return np.concatenate(([np.log(q_norm)], (np.arccos((1.0*qs)/q_norm)/qv_norm)*qv))

#===========================================#
# Normalize a quaternion to have unit norm  #
#  - Used as helper function for qAvg below #  
#                                           #
#  Input: quaternion q                      #
# Output: Normalized version of q           #
#===========================================#
def qUnitize(q):
	q_norm = qNorm(q)
	if q_norm == 0:
		return q
	return q/q_norm

#=====================================================================================#
# Compute the weighted average of a list of quaternioins                              #
#                                                                                     #
#  Input: (required) list of quaternions qList                                        #
#         (optional)  qWeights: corresponding weights (default: equal weight is used) #
#         (optional)       eps: error tolerance (default: 0.001)                      #
#         (optional) num_iters: maximum number of iterations to find convergence      #
# Output: Normalized version of q                                                     #
#=====================================================================================#
def qAvg(qList, qWeights=None, eps=0.001, num_iters=100):
	num_q = len(qList)
	if num_q == 0: return None
	if num_q == 1: return qList[0]

	# generate default weights
	if qWeights is None:
		qWeights = [1.0/num_q]*num_q

	q_avg = qList[0]
	for i in range(num_iters):
		# multiply all quaternions by inverse of average
		q_avg_inv = qInv(q_avg)
		q_e = [qMult(q_avg_inv, q) for q in qList]

		# error rotation vector from quaternion
		err = [2*qLog(q) for q in q_e]
		err_v = [q[1:] for q in err]

		# restrict angles to [-pi,pi)
		err_v = [(-np.pi + float(Decimal(qNorm(qv) + np.pi)%Decimal(2*np.pi)))*qUnitize(qv) \
		                                                                     for qv in err_v]
		# compute weighted sum of error                             
		err_v_wgtd_sum = np.array([0,0,0])
		for j, qv in enumerate(err_v):
			err_v_wgtd_sum = err_v_wgtd_sum + (qWeights[j]*qv) 

		# error rotation vector to quaternion
		q_avg = qMult(q_avg, qExp(np.concatenate(([0], 0.5*err_v_wgtd_sum))))

		# check for convergence
		if qNorm(err_v_wgtd_sum) < eps:
			return q_avg

	return qAvg

#===========================#
# Convert degree to radians #
#                           #
#  Input: degree deg        #
# Output: deg in radians    #
#===========================#
def deg2Rad(deg):
	return deg*np.pi/180.0

#===========================#
# Convert radian to degrees #
#                           #
#  Input: radian rad        #
# Output: rad in degrees    #
#===========================#
def rad2Deg(rad):
	return rad*180.0/np.pi

#=================================================================#
# Convert Euler angles to rotation matrix                         #
#                                                                 #
#  Input: Euler angles x,y,z OR a list of Euler angles [x,y,z]    #
# Output: XYZ rotation matrix corresponding to Euler angles x,y,z #
#=================================================================#
def euler2Rot(*args):
	x, y, z = args[0] if len(args) == 1 else args

	# initialize axes-specific rotation matrices
	Rx, Ry, Rz = np.eye(3), np.eye(3), np.eye(3)

	# populate rotation matrices for each axes
	cx, sx = np.cos(x), np.sin(x)
	Rx[1:,1:] = np.array([[cx, -sx],[sx, cx]])
	cy, sy = np.cos(y), np.sin(y)
	Ry[0][0], Ry[0][2], Ry[2][0], Ry[2][2] = cy, sy, -sy, cy
	cz, sz = np.cos(z), np.sin(z)
	Rz[:2,:2] = np.array([[cz, -sz],[sz, cz]])

	return Rz.dot(Ry.dot(Rx))

#==============================================================================================#
# Convert rotation matrix to Euler angles                                                      #
#  - Uses: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.371.6578&rep=rep1&type=pdf #
#                                                                                              #
#  Input: rotation matrix R                                                                    #
# Output: np array of Euler angles corresponding to R                                          #
#==============================================================================================#
def rot2Euler(R):
	x, y, z = 0, 0, 0
	if abs(abs(R[2][0]) - 1) > 0.00000001:
		y = -np.arcsin(R[2][0])
		cy = np.cos(y)
		x = np.arctan2(1.0*R[2][1]/cy, 1.0*R[2][2]/cy)
		z = np.arctan2(1.0*R[1][0]/cy, 1.0*R[0][0]/cy)
	else:
		z = 0
		if R[2][0] < 0:
			y = np.pi/2
			x = np.arctan2(R[0][1], R[0][2])
		else:
			y = -np.pi/2
			x = np.arctan2(-R[0][1], R[0][2])

	return np.array([x, y, z])

#============================================================================================#
# Convert axis-angle to rotation matrix                                                      #
#  - Uses: https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle #
#                                                                                            #
#  Input: axis and angle, OR a list of both                                                  #
# Output: rotation matrix corresponding to axis-angle                                        #
#============================================================================================#
def axis2Rot(*args):
	axis, angle = args[0] if len(args) == 1 else args
	x, y, z = axis.reshape((3,))
	c, s = np.cos(angle), np.sin(angle)

	return np.array([[ c + x*x*(1-c),  x*y*(1-c) - z*s, x*z*(1-c) + y*s],\
	                 [y*x*(1-c) + z*s,  c + y*y*(1-c),  y*z*(1-c) - x*s],\
	                 [z*x*(1-c) - y*s, z*y*(1-c) + x*s,  c + z*z*(1-c) ]])

#============================================================================================#
# Convert rotation matrix to axis-angle                                                      #
#  - Uses: http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToAngle/ #
#                                                                                            #
#  Input: rotation matrix R                                                                  #
# Output: np array of axis and angle corresponding to R                                      #
#============================================================================================#
def rot2Axis(R):
	angle = np.arccos((R[0][0] + R[1][1] + R[2][2] - 1)/2.0)

	# handle special case when angle is 0
	if abs(angle) < 0.00000001: return np.array([np.eye(3)[:,1].reshape((3,1)), angle]) 

	# handle special case when anple is pi (180 degrees)
	if abs(np.pi - angle) < 0.00000001: 
		axis = np.array([0,0,0])

		# get maximum diagonal element
		max_index = np.argmax(np.diag(R))  

		# use column of largest diagonal element to generate axis
		j, k = (max_index+1)%3, (max_index+2)%3
		axis[max_index] = np.sqrt((R[max_index][max_index] + 1)/2.0)
		axis[j] = (R[max_index][j] + R[j][max_index])/(4*axis[max_index])
		axis[k] = (R[max_index][k] + R[k][max_index])/(4*axis[max_index])
		return np.array([axis.reshape((3,1)), np.pi])

	# handle non-special cases	
	return np.array([(1/(2.0*np.sin(angle)))*np.array([[R[2][1]-R[1][2]], \
		                                               [R[0][2]-R[2][0]], \
		                                               [R[1][0]-R[0][1]]]), angle])

#=======================================================#
# Convert quaternion to axis-angle                      #
#                                                       #
#  Input: quaternion q                                  #
# Output: np array of axis and angle corresponding to q #
#=======================================================#
def quat2Axis(q):
	omega = 2*qLog(q)[1:]
	angle = qNorm(omega)
	if angle == 0: 
		return np.array([omega.reshape((3,1)), angle])
	return np.array([omega.reshape((3,1))/angle, angle])

#================================================#
# Convert axis-angle to quaternion               #
#                                                #
#  Input: axis and angle, OR a list of both      #
# Output: quaternion corresponding to axis-angle #
#================================================#
def axis2Quat(*args):
	axis, angle = args[0] if len(args) == 1 else args
	omega = angle*axis.reshape((3,))
	return qExp(np.concatenate(([0], 0.5*omega)))

#=======================================#
# Convert rotation matrix to quaternion #
#                                       #
#  Input: rotation matrix R             #
# Output: quaternion corresponding to R #
#=======================================#
def rot2Quat(R):
	return axis2Quat(rot2Axis(R))

#============================================#
# Convert quaternion to rotation matrix      #
#                                            #
#  Input: quaternion q                       #
# Output: rotation matrix corresponding to q #
#============================================#
def quat2Rot(q):
	return axis2Rot(quat2Axis(q))

#==============================================================#
# Convert Euler angles to quaternion                           #
#                                                              #
#  Input: Euler angles x,y,z OR a list of Euler angles [x,y,z] #
# Output: quaternion corresponding to Euler angles x,y,z       #
#==============================================================#
def euler2Quat(*args):
	x, y, z = args[0] if len(args) == 1 else args
	return axis2Quat(rot2Axis(euler2Rot(x,y,z)))

#=====================================================#
# Convert quaternion to Euler angles                  #
#                                                     #
#  Input: quaternion q                                #
# Output: np array of Euler angles corresponding to q #
#=====================================================#
def quat2Euler(q):
	return rot2Euler(axis2Rot(quat2Axis(q)))

#========================================================================#
# Convert Euler angles to axis-angle                                     #
#                                                                        #
#  Input: Euler angles x,y,z OR a list of Euler angles [x,y,z]           #
# Output: np array of axis and angle corresponding to Euler angles x,y,z #
#========================================================================#
def euler2Axis(*args):
	x, y, z = args[0] if len(args) == 1 else args
	return rot2Axis(euler2Rot(x,y,z))

#==============================================================#
# Convert axis-angle to Euler angles                           #
#                                                              #
#  Input: axis and angle, OR a list of both                    #
# Output: np array of Euler angles corresponding to axis-angle #
#==============================================================#
def axis2Euler(*args):
	axis, angle = args[0] if len(args) == 1 else args
	return rot2Euler(axis2Rot(axis, angle))
