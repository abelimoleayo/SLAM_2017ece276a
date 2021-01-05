#========================================================#
#     Author: Imoleayo Abel                              #
#     Course: Sensing and Estimation Robotics (ECE 276A) #
#    Quarter: Fall 2017                                  #
# Instructor: Nikolay Atanasov                           #
#    Project: 03 - SLAM and Texture Mapping              #
#       File: SLAM.py                                    #
#       Date: Dec-15-2017                                #
#========================================================#
import p3_utils as p3u
import load_data as load
import numpy as np
import matplotlib.pyplot as plt
import os
from myquaternion import euler2Rot, rot2Euler

#========================================================#
# Invert rigid transformation                            #
#                                                        #
#  Input: rigid transformation g in SE(3)                #
# Output: inverse transformation of g                    #
#========================================================#
def invertTransformation(g):
    g_inv = np.eye(4)
    g_inv[:3,3] = -g[:3,:3].T.dot(g[:3,3]) # compute translation
    g_inv[:3,:3] = g[:3,:3].T              # compute rotation
    return g_inv

####################################################################################################

# specify dataset suffix, set to str() if dataset has no numeric suffix
dataset = str(1)

# specify prefix of folder to read file from. E.g. set to "train" for "trainset" folder
folder_prefix = "train"  

####################################################################################################
#                                           LOAD DATA                                              #
####################################################################################################

# load lidar and joint data
folder = folder_prefix + "set"
print "\nLoading lidar data..."
lidar_filename = folder + "/lidar/" + folder_prefix + "_lidar" + len(dataset)*dataset
lidars = load.get_lidar(lidar_filename)
print "Done!\n"
print "Loading joint data..."
joint_filename = folder + "/joint/" + folder_prefix + "_joint" + len(dataset)*dataset
joint = load.get_joint(joint_filename)
print "Done!\n"

# load RGB and DEPTH data
print "Attempting to load RGB and DEPTH data..."
rgb_exists = False
RGBs = []
depths = []
try:
    load_depth = True
    RGB_folder = folder + "/cam"
    RGB_filenames = []
    RGB_filename_prefix = "RGB" + len(dataset)*"_" + dataset
    for filename in os.listdir(RGB_folder):
        if RGB_filename_prefix.lower() in filename.lower():
            RGB_filenames.append(filename)
    if len(RGB_filenames) == 0:
        print "   No RGB or DEPTH data."
        load_depth = False
    elif len(RGB_filenames) == 1:
        print "   Loading RGB data..."
        RGBs.extend(load.get_rgb(RGB_folder + "/" + RGB_filename_prefix))
        print "   Done!"
    else:
        print "   Loading " + str(len(RGB_filenames)) + " RGB datafiles..."
        for i in xrange(len(RGB_filenames)):
            print "      Loading RBG data #" + str(i+1)
            RGB_filename = RGB_folder + "/" + RGB_filename_prefix + "_" + str(i+1)
            RGBs.extend(load.get_rgb(RGB_filename))
            print "      Done!"
        print "   Done!"
    if load_depth:
        print "   Loading DEPTH data..."
        depth_filename = folder + "/cam/DEPTH" + len(dataset)*"_" + dataset
        depths = load.get_depth(depth_filename)
        print "   Done!"
        rgb_exists = True
except:
    print "   Error loading RGB or DEPTH data."
print "Done!\n"

####################################################################################################
#                                      INITIALIZE VARIABLES                                        #
####################################################################################################

# particle related variables (particles are in SE3)
n_particles_thresh = 10
n_particles = 100
particle_weights = np.ones(n_particles)/(1.0*n_particles)
particle_corrs = np.zeros(n_particles)
particle_poses = np.zeros((n_particles, 4, 4)) 
particle_poses[:] = np.eye(4)    # initialize poses to identity 
particle_poses[:,2,3] = 0.93     # set z location to center of mass

noise_mean, noise_cov = np.array([0,0,0]), 0.1*np.eye(3)  # noise for motion model prediction

# occupancy grid variables
half_dim = 30.0   # half width (in meters) of occupancy grid
xmin, xmax, ymin, ymax = -half_dim, half_dim, -half_dim, half_dim
map_resolution = 0.05
map_ncols = int(np.ceil((xmax - xmin)/map_resolution + 1))  
map_nrows = int(np.ceil((ymax - ymin)/map_resolution + 1))
map_logodds = np.zeros((map_nrows, map_ncols))   # main log_odds grid
map_logodds_thresh = np.zeros(map_logodds.shape) # thresholded copy of log_odds for map correlation
map_xmeters = np.arange(xmin, xmax + map_resolution, map_resolution)
map_ymeters = np.arange(ymin, ymax + map_resolution, map_resolution)
x_corr_range = np.arange(-0.2,0.2+0.05,0.05)
x_corr_range_cnt = len(x_corr_range)
y_corr_range = np.arange(-0.2,0.2+0.05,0.05)
y_corr_range_cnt = len(y_corr_range)
corr_center_index = len(x_corr_range)*len(y_corr_range)/2
yaw_corr_range = 1*x_corr_range
yaw_corr_range_cnt = len(yaw_corr_range)
log_fill = np.log(8)  # choosing g(1)=8

# camera/depth related variables if RGB/DEPTH data exists
if rgb_exists:
    rgb_index, rgb_cnt = 0, len(RGBs)
    depth_index, depth_cnt = 0, len(depths)

    exir_rgb = load.getExtrinsics_IR_RGB()
    ir_to_RGB = np.eye(4)
    ir_to_RGB[:3,:3], ir_to_RGB[:3,3] = exir_rgb['rgb_R_ir'], exir_rgb['rgb_T_ir']
    ir_calib = load.getIRCalib()
    K_ir = np.eye(3)
    # alpha_c is 0, no need to update index [0,1]
    K_ir[[0,1],[0,1]], K_ir[:2,2] = ir_calib['fc'], ir_calib['cc'] 
    rgb_Calib = load.getRGBCalib()
    K_rgb = np.eye(3)
    # alpha_c is 0, no need to update index [0,1]
    K_rgb[[0,1],[0,1]], K_rgb[:2,2] = rgb_Calib['fc'], rgb_Calib['cc']
    optical_to_regular = np.zeros((3,3))
    optical_to_regular[[0,1,2],[2,0,1]] = np.array([1, -1, -1])

    # kinect sensor pose in head frame
    kinect_to_head = np.eye(4)
    kinect_to_head[2,3] = 0.07

    rgb_nrows, rgb_ncols, _ = RGBs[0]["image"].shape  # we don't need number of channels (always 3)
    ir_nrows, ir_ncols = depths[0]["depth"].shape

    # convert pixel locations in depth image to normalized Cartesian coordinates in the optical
    # coordinate of the depth sensor
    ir_pixels_loc = np.mgrid[:ir_nrows, :ir_ncols].reshape((2,ir_nrows*ir_ncols))
    ir_pixels_loc[[0,1]] = ir_pixels_loc[[1,0]]
    ir_pixels_loc = np.concatenate((ir_pixels_loc, np.ones((1,ir_nrows*ir_ncols))), axis=0)
    ir_pixels_loc = np.linalg.inv(K_ir).dot(ir_pixels_loc)

    # texture map with similar dimension as log_odds map
    map_texture = 255*np.ones((map_nrows, map_ncols, 3), dtype=np.uint8)

# other variables used repeatedly in loop   
joint_t = joint["ts"][0]
joint_t_index, joint_t_cnt = 0, len(joint_t)

# lidar is just 0.15m above head 
lidar_to_head = np.eye(4)
lidar_to_head[2,3] = 0.15

# angular position of lidar beams 
lidar_angles = np.linspace(-0.75*np.pi, 0.75*np.pi, 1081)

# variables for computing odometry motion model
prior_lidar_to_world = np.eye(4)
prior_lidar_to_body = np.eye(4)

# z value (meters) considered as ground level
z_floor = 0.1

dataskip = 100                               # skip 100 steps in data
num_steps = len(lidars)/dataskip             # number of times loop runs
body_position_map = np.zeros((2,num_steps))  # robot (x,y) coordinates in occupancy gris cell units

# for printing progress to command line
progress = 0
last_printed_progess = 0

####################################################################################################
#                                     MAIN PARTICLE FILTER LOOP                                    #
####################################################################################################

print "Progress: 0% complete."
for i in xrange(num_steps):
    # print progress
    progress = np.round((100.0*i)/num_steps)
    if progress > last_printed_progess:
        print "Progress:",str(int(progress))+"% complete."
        last_printed_progess = progress

    # get lidar data 
    lidar_index = i*dataskip
    lidar = lidars[lidar_index]  
    lidar_time = lidar['t'][0][0]

    # compute corresponding joint data based on lidar time
    while (joint_t_index + 1 < joint_t_cnt) and (joint_t[joint_t_index + 1] <= lidar_time):
        joint_t_index += 1

    # compute corresponding RGB/depth data based on lidar time
    if rgb_exists:
        while (rgb_index + 1 < rgb_cnt) and (RGBs[rgb_index + 1]['t'][0][0] <= lidar_time):
            rgb_index += 1
        while (depth_index + 1 < depth_cnt) and (depths[depth_index + 1]['t'][0][0] <= lidar_time):
            depth_index += 1
        
    # get lidar odometry and joint angles
    lidar_SE2pose = lidar["pose"][0]
    neck_angle, head_angle = joint["head_angles"][:, joint_t_index]

    # compute lidar to body (in SE(3))
    head_to_body =  np.eye(4)
    head_to_body[:3,:3], head_to_body[2,3] = euler2Rot(0,head_angle,neck_angle), 0.33
    lidar_to_body = head_to_body.dot(lidar_to_head)

    # lidar to world
    lidar_to_world = np.eye(4)
    lidar_to_world[:3,:3], lidar_to_world[:3,3] = euler2Rot(0, 0, lidar_SE2pose[2]), \
                                                  np.concatenate((lidar_SE2pose[:2],\
                                                                 [lidar_to_body[2,3] + 0.93]))
    # odometry motion model control
    motion_model_control = prior_lidar_to_body.dot(\
                           invertTransformation(prior_lidar_to_world).dot(\
                           lidar_to_world.dot(invertTransformation(lidar_to_body))))

    # cache variables for next iteration
    prior_lidar_to_world = lidar_to_world
    prior_lidar_to_body = lidar_to_body

    # lidar scan data
    lidar_scan = lidar["scan"][0]

    # remove too near and too far values
    valid_indices = np.logical_and((lidar_scan <= 30),(lidar_scan >= 0.1))

    # convert valid lidar scan data to Cartesian coordinates in lidar frame
    valid_cnt = np.sum(valid_indices)
    scan_lidar_frame = np.zeros((4,valid_cnt))
    scan_lidar_frame[0,:] = lidar_scan[valid_indices] * np.cos(lidar_angles[valid_indices])
    scan_lidar_frame[1,:] = lidar_scan[valid_indices] * np.sin(lidar_angles[valid_indices])
    scan_lidar_frame[3,:] = np.ones(valid_cnt)

    if i == 0: 
        # during first loop iteration, no need to predict/update, just create first log_odds map
        # using the first particle since all particles are the same
        best_particle = particle_poses[0]
    else:
        ############################################################################################
        #                            MOTION MODEL PREDICTION WITH NOISE                            #
        ############################################################################################

        for particle_index in xrange(n_particles):
            # noise pose - only x, y, yaw has noise
            noise_x, noise_y, noise_theta = np.random.multivariate_normal(noise_mean, noise_cov)
            noise_pose = np.eye(4)
            noise_pose[:3,:3], noise_pose[:2,3] = euler2Rot(0,0,noise_theta), \
                                                  np.array([noise_x, noise_y])

            # motion model with noise
            particle_poses[particle_index] = particle_poses[particle_index].dot(\
                                                               noise_pose.dot(motion_model_control))

        ############################################################################################
        #                                OBSERVARTION MODEL UPDATE                                 #
        ############################################################################################

        # threshold log-odds map
        map_logodds_thresh[map_logodds > 0] = 1
        map_logodds_thresh[map_logodds <= 0] = 0

        # update model
        for particle_index in xrange(n_particles):
            # retrieve roll, pitch, yaw of particle
            particle_roll, particle_pitch, particle_yaw = rot2Euler(particle_poses[particle_index,\
                                                                                             :3,:3])
            # compute range of yaw values to test correlation for
            corr_yaws = particle_yaw + yaw_corr_range

            # initialize variables for finding the yaw and indices having the best correlation
            best_yaw_index = 0
            best_corr_index = 0
            best_corr = 0

            # iterate over yaw angles
            for yaw_index in xrange(yaw_corr_range_cnt):
                # update rotation matrix portion of particle using current yaw angle
                particle_poses[particle_index,:3,:3] = euler2Rot(particle_roll, particle_pitch, \
                                                                               corr_yaws[yaw_index]) 
                # convert lidar points to world frame using current particle
                scan_world_frame_all = \
                             particle_poses[particle_index].dot(lidar_to_body.dot(scan_lidar_frame))

                # remove points that hit the ground
                valid_indices = scan_world_frame_all[2,:] >= z_floor
                valid_cnt = np.sum(valid_indices)
                scan_world_frame = scan_world_frame_all[:, valid_indices]

                # compute map correlation
                map_corr = p3u.mapCorrelation(map_logodds_thresh.T,map_xmeters,map_ymeters,\
                                               scan_world_frame[:2,:],x_corr_range,y_corr_range)
                map_corr = map_corr.T
                max_corr_index = np.argmax(map_corr)
                max_corr = map_corr[max_corr_index/y_corr_range_cnt, \
                                    max_corr_index%y_corr_range_cnt]

                # update best yaw, and best correlation index
                if yaw_index == 0:
                    best_corr_index = max_corr_index
                    best_corr = max_corr
                else:
                    # only update if better than current best OR
                    # same as current best but closer to original particle position
                    if (max_corr > best_corr) or \
                       ((max_corr == best_corr) and \
                        (np.linalg.norm([max_corr_index - corr_center_index, \
                                         yaw_index - yaw_corr_range_cnt/2]) < \
                         np.linalg.norm([best_corr_index - corr_center_index, \
                                         best_yaw_index - yaw_corr_range_cnt/2]))):
                        best_yaw_index = yaw_index 
                        best_corr_index = max_corr_index
                        best_corr = max_corr

            # move particle to position with best correlation
            particle_poses[particle_index,:3,:3] = euler2Rot(particle_roll, particle_pitch, \
                                                                          corr_yaws[best_yaw_index])
            particle_poses[particle_index,:2,3] += np.array([x_corr_range[best_corr_index % \
                                                                          y_corr_range_cnt],\
                                                             y_corr_range[best_corr_index / \
                                                                          y_corr_range_cnt]])
            # save particle correlation 
            particle_corrs[particle_index] = best_corr

        # update particle weights
        particle_corrs = np.exp(particle_corrs - max(particle_corrs))
        particle_corrs = particle_corrs/np.sum(particle_corrs)
        particle_weights = particle_weights * particle_corrs
        particle_weights = particle_weights/np.sum(particle_weights)

        # select best particle
        best_particle = particle_poses[np.argmax(particle_weights)]

        ############################################################################################
        #                             RESAMPLE PARTICLES IF NECESSARY                              #
        ############################################################################################
        
        # Stratified resampling algorithm (from class notes)
        if (1.0/np.sum(particle_weights * particle_weights)) < n_particles_thresh:
            new_particles = np.zeros(particle_poses.shape)
            j, c = 0, particle_weights[0]
            divisor = 1.0/n_particles
            for particle_index in xrange(n_particles):
                u = np.random.uniform(0.0, divisor)
                beta = u + particle_index*divisor
                while beta > c:
                    j += 1
                    c += particle_weights[j]
                new_particles[particle_index] = particle_poses[j]
            particle_poses = new_particles
            particle_weights = divisor * np.ones(n_particles)

        ############################################################################################
        #                                     TEXTURE MAPPING                                      #
        ############################################################################################
        
        if rgb_exists:
            # retrieve RGB and depth images
            rgb = RGBs[rgb_index]
            depth = depths[depth_index]
            depth['depth'] = 0.001*depth['depth'] # convert mm to m

            # compute head to body transformation using RGB data. This will very likely be the same
            # angles from the joint data, but for specificity and potential offset in timestamp 
            # between joint and RGB data, we do this
            neck_angle, head_angle = rgb['head_angles'][0]
            rgb_head_to_body = np.eye(4)
            rgb_head_to_body[:3,:3], rgb_head_to_body[2,3] = euler2Rot(0,head_angle,neck_angle),\
                                                                                             0.33
            # un-normalize Cartesian coordinates (in depth sensor optical frame) of pixel locations 
            # of depth image
            pt_cloud_XYZ = depth['depth'].reshape(ir_nrows*ir_ncols) * ir_pixels_loc

            # convert from depth sensor optical frame to RGB camera optical frame
            pt_cloud_XYZ = ir_to_RGB[:3,:3].dot(pt_cloud_XYZ) + ir_to_RGB[:3,3].reshape((3,1))

            # copy of RGB camera optical frame coordinates for converting to RGB image pixel
            # location (original copy would be used for transforming to world Cartesian frame)
            ir_pixels_loc_RGB = pt_cloud_XYZ.copy()

            # normalize Cartesian coordinates in RGB optical frane (only normalize points with 
            # non-zero z-values to avoid division by 0)
            valid_indices = ir_pixels_loc_RGB[2,:] > 0
            ir_pixels_loc_RGB[:, valid_indices] = ir_pixels_loc_RGB[:, valid_indices] / \
                                                                 ir_pixels_loc_RGB[2, valid_indices]
            
            # convert normalized RGB optical frame coordinates to RGB pixel locations
            ir_pixels_loc_RGB = K_rgb.dot(ir_pixels_loc_RGB)
            ir_pixels_loc_RGB = ir_pixels_loc_RGB.astype(np.int)

            # convert RGB optical frame coordinates to regular frame coordinates
            pt_cloud_XYZ = optical_to_regular.dot(pt_cloud_XYZ)

            # add a 4th demension for rigid transformation
            pt_cloud_XYZ = np.concatenate((pt_cloud_XYZ, np.ones((1,ir_nrows*ir_ncols))), axis=0)

            # convert RGB regular frame coordinates to world frame using best particle
            pt_cloud_XYZ = \
                         (best_particle.dot(rgb_head_to_body.dot(kinect_to_head))).dot(pt_cloud_XYZ)

            # Since we started in depth pixel coordinates, get indices corresponding to the floor 
            # in world frame, that fit in the boundary of world grid map, and that fit in boundary 
            # of RGB camera pixel range (in case the transformation from depth pixel to RGB pixel 
            # goes off bounds by any chance)
            valid_indices = np.logical_and(\
                              np.logical_and(pt_cloud_XYZ[2,:] <= z_floor, \
                                           np.logical_and(np.logical_and(pt_cloud_XYZ[0,:] >= xmin,\
                                                                        pt_cloud_XYZ[0,:] <= xmax),\
                                                          np.logical_and(pt_cloud_XYZ[1,:] >= ymin,\
                                                                       pt_cloud_XYZ[1,:] <= ymax))),
                              np.logical_and(np.logical_and(ir_pixels_loc_RGB[0,:] >= 0, \
                                                            ir_pixels_loc_RGB[0,:] < rgb_ncols),\
                                             np.logical_and(ir_pixels_loc_RGB[1,:] >= 0, \
                                                            ir_pixels_loc_RGB[1,:] < rgb_nrows)))

            # select only valid world coordinates satisfying criteria above
            ir_pixels_loc_RGB = ir_pixels_loc_RGB[:, valid_indices]
            pt_cloud_XYZ = pt_cloud_XYZ[:, valid_indices]

            # convert valid points to coordinates (in grid cells) of texture map
            texture_cols = np.ceil((pt_cloud_XYZ[0,:] - xmin)/map_resolution).astype(np.int16) - 1
            texture_rows = np.ceil((pt_cloud_XYZ[1,:] - ymin)/map_resolution).astype(np.int16) - 1

            # update texture map
            map_texture[texture_rows, texture_cols] = rgb['image'][ir_pixels_loc_RGB[1,:], \
                                                                   ir_pixels_loc_RGB[0,:]]

    ################################################################################################
    #                                     UPDATE LOG-ODDS MAP                                      #
    ################################################################################################

    # compute and save off (x.y) position (in grid cells) of best particle for plotting
    body_position_map[0,i] = np.ceil((best_particle[0,3] - xmin) / \
                                      map_resolution).astype(np.int16) - 1
    body_position_map[1,i] = np.ceil((best_particle[1,3] - ymin)/ \
                                      map_resolution).astype(np.int16) - 1

    # use best particle to get world coordinate of lidar hit points
    scan_world_frame_all = best_particle.dot(lidar_to_body.dot(scan_lidar_frame))

    # remove ground plane and corresponding data points
    valid_indices = scan_world_frame_all[2,:] >= z_floor
    valid_cnt = np.sum(valid_indices)
    scan_world_frame = scan_world_frame_all[:, valid_indices]   

    # convert world (x,y) coordinates of lidar hit point to grid cells coordinates
    map_hit_cols = np.ceil((scan_world_frame[0,:] - xmin)/map_resolution).astype(np.int16) - 1
    map_hit_rows = np.ceil((scan_world_frame[1,:] - ymin)/map_resolution).astype(np.int16) - 1

    # get grid-cell coordinates of empty cells using bresenham2D
    lidar_pose_corrected = best_particle.dot(lidar_to_body)
    start_x = np.ceil((lidar_pose_corrected[0,3] - xmin)/map_resolution).astype(np.int16) - 1
    start_y = np.ceil((lidar_pose_corrected[1,3] - ymin)/map_resolution).astype(np.int16) - 1
    empty_cells = np.vstack((start_x, start_y))
    for k in xrange(valid_cnt):
        empty_cells = np.concatenate((empty_cells, \
                       p3u.bresenham2D(start_x, start_y, map_hit_cols[k], map_hit_rows[k])), axis=1)
    empty_cells = empty_cells.astype(np.int16)

    # update log odds map (NOTE: because bresenham2D includes the end-point of ray in the list of 
    # free cells, the first line below would ALSO decrease log odds of the hit-points, thus the 
    # second line increases the log-odds twice
    map_logodds[empty_cells[1,:], empty_cells[0,:]] -= log_fill
    map_logodds[map_hit_rows, map_hit_cols] += 2*log_fill

# PLOT
map_image = 127*np.ones((map_logodds.shape[0],map_logodds.shape[1],3), dtype=np.uint8) 
map_image[map_logodds < 0] = np.array([255, 255, 255])
map_image[map_logodds > 0] = np.array([0, 0, 0])
body_position_map = body_position_map.astype(np.int16)

if rgb_exists:
    map_combined = map_image.copy()
    valid_indices = map_image == np.array([255, 255, 255])
    map_combined[valid_indices] = map_texture[valid_indices]

    fig, axes = plt.subplots(2,2)
    axes[0,0].imshow(map_image)
    axes[0,0].plot(body_position_map[0,:],body_position_map[1,:],'b')
    axes[0,1].imshow(map_texture)
    axes[1,0].imshow(map_combined)
    axes[1,0].plot(body_position_map[0,:],body_position_map[1,:],'b')

    axes[0,0].invert_yaxis()
    axes[0,1].invert_yaxis()
    axes[1,0].invert_yaxis()
else:
    fig, ax = plt.subplots()
    ax.imshow(map_image)
    ax.plot(body_position_map[0,:],body_position_map[1,:],'b')
    ax.invert_yaxis()
print "Complete!!! See plot window."
plt.show()
