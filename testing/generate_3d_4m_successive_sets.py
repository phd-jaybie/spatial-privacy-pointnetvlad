import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import math
import pickle
import pandas as pd
import time
import h5py
import bz2

from numpy import linalg as LA
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0,"../spatial-privacy/")
from info3d import *
from nn_matchers import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
baseline_path = '3d_dataset_4m/'

spatial_span = 2.0

skip = 3

t0 = time.time()

for radius in np.arange(0.5,2.1,0.5):
    
    t1 = time.time()
            
    try:
        with bz2.BZ2File('../spatial-privacy/parallel_samples/{}_100_200_20000nn_successive_point_cloud.pickle.bz2'.format(0.5), 'r') as bz2_f:
            successive_point_collection = pickle.load(bz2_f)

        samples = len(successive_point_collection)
        releases = len(successive_point_collection[0][1])

        print(samples,"samples for radius",radius)
        print(releases,"releases each")
        
    except Exception as e1:
        print(e1)
        continue
        
    try:
        with open(str(radius)+'_successive_point_collection_per_release.pickle','rb') as f:
            successive_point_collection_per_release = pickle.load(f)
            
        with open(str(radius)+'_g_successive_point_collection_per_release.pickle','rb') as f:
            g_successive_point_collection_per_release = pickle.load(f)
            
        print("save points exist")
            
    except:
        print("no save points")
        successive_point_collection_per_release = []
        g_successive_point_collection_per_release = []

        for j in range(int(releases/3)+1):
            successive_point_collection_per_release.append([])
            g_successive_point_collection_per_release.append([])
    
    for k, [obj_, growing_point_collection] in enumerate(successive_point_collection[len(successive_point_collection_per_release[0]):]):
        
        t2 = time.time()
        
        growing_point_cloud = np.asarray([])
        growing_p_point_cloud = np.asarray([])
        
        #print(k, obj_)
        
        for i in range(len(growing_point_collection)):
            
            t_pointCloud = growing_point_collection[i][1]
            
            #Regular Accumulation
            if len(growing_point_cloud) == 0:
                growing_point_cloud = t_pointCloud

            else:
                growing_point_cloud = np.concatenate(
                    (growing_point_cloud,t_pointCloud),
                    axis=0
                )
                
            #RANSAC generalizations
            if len(growing_p_point_cloud) == 0:
                gen_planes = getLOCALIZEDRansacPlanes(
                    pointCloud = t_pointCloud,
                    original_vertex = growing_point_collection[i][0][-1],
                    #verbose = True
                )
                #print(obj_meta[-1])
            else:
                gen_planes = updatePlanesWithSubsumption(
                    new_pointCloud=t_pointCloud,
                    existing_pointCloud=growing_p_point_cloud,
                    planes_to_find = max(min(i,50),30),
                    #verbose=True
                )
                
            if len(gen_planes) == 0:
                print("No gen planes after release",release_count,growing_point_cloud.shape)
                continue
                
            try:
                updated_point_cloud, updated_triangles = getGeneralizedPointCloud(
                    planes = gen_planes,
                    triangle_area_threshold = 0.2,#2.0*np.amax(getTriangleAreas(partial_pointCloud, partial_triangles))
                    #verbose = True
                )
                
                growing_p_point_cloud = np.asarray(updated_point_cloud)
                
            except Exception as ex:
                print("Error getting updated point cloud in release",i)
                print(" ",growing_p_point_cloud.shape, growing_p_triangles.shape,partial_pointcloud.shape)
                print(ex)
                print(gen_planes)
                
                continue
                
            if len(growing_p_point_cloud) == 0:
                continue
                            
            if i % skip != 1: # skip 
                continue
                                
            #Regular Processing
            growing_point_cloud = np.unique(growing_point_cloud,axis=0)
            
            successive_point_collection_per_release[int(i/3)].append([
                growing_point_collection[i][0],
                growing_point_cloud,
                i
                #growing_point_collection[i][0],
                #growing_point_collection[i][1],
                #growing_point_collection[i][2]
            ])
            
            g_successive_point_collection_per_release[int(i/3)].append([
                growing_point_collection[i][0],
                growing_p_point_cloud,
                i
                #growing_point_collection[i][0],
                #growing_point_collection[i][1],
                #growing_point_collection[i][2]
            ])
                        
            print("    Done with {} releases ({},{} samples) in {:.3f} seconds.".format(
                i,
                k,
                len(successive_point_collection_per_release[int(i/3)]),
                time.time() - t2
            ),growing_point_cloud.shape,growing_p_point_cloud.shape)
            t2 = time.time()
            
        with open(str(radius)+'_successive_point_collection_per_release.pickle','wb') as f:
            pickle.dump(successive_point_collection_per_release,f)
            
        with open(str(radius)+'_g_successive_point_collection_per_release.pickle','wb') as f:
            pickle.dump(g_successive_point_collection_per_release,f)
                
    successive_path = os.path.join(BASE_DIR,baseline_path,"successive_radius_"+str(radius))
    
    if not os.path.exists(successive_path): os.mkdir(successive_path)
        
    for i in range(int(releases/3)):
        
        per_release_point_collection = successive_point_collection_per_release[i]
        
        if len(per_release_point_collection) == 0: continue

        pointcloud_successive_path = os.path.join(successive_path,"release_"+str(i*3+1))
        if not os.path.exists(pointcloud_successive_path): os.mkdir(pointcloud_successive_path)
            
        csvfile = open(successive_path+"/release_"+str(i*3+1)+"_centroids_4m_0.5.csv",'w',newline = '')

        csv_writer = csv.writer(csvfile, delimiter = ',')
        csv_writer.writerow(['timestamp', 'northing', 'easting', 'alting'])
        
        for point_collection in per_release_point_collection:
            
            if len(point_collection) == 0:
                continue
                
            obj_meta = point_collection[0]
            pointCloud = point_collection[1]
            
            new_partial_pointcloud = []
            new_vX = []
            new_vZ = []

            original_vertex = obj_meta[2]
            object_name = obj_meta[1]

            if object_name == "Reception-Data61-L5.obj":
                new_X = pointCloud[:,0] + 50
                new_Z = pointCloud[:,2] + 0
                new_vX = original_vertex[0] + 50
                new_vZ = original_vertex[2] + 0
            elif object_name == "Driveway.obj":
                new_X = pointCloud[:,0] - 25
                new_Z = pointCloud[:,2] - 50
                new_vX = original_vertex[0] - 25
                new_vZ = original_vertex[2] - 50
            elif object_name == "Apartment.obj":
                new_X = pointCloud[:,0] + 25
                new_Z = pointCloud[:,2] - 50
                new_vX = original_vertex[0] + 25
                new_vZ = original_vertex[2] - 50
            elif object_name == "Workstations-Data61-L4.obj":
                new_X = pointCloud[:,0] - 50
                new_Z = pointCloud[:,2] + 0
                new_vX = original_vertex[0] - 50
                new_vZ = original_vertex[2] + 0
            elif object_name == "Kitchen-Data61-L4.obj":
                new_X = pointCloud[:,0] + 0
                new_Z = pointCloud[:,2] + 0
                new_vX = original_vertex[0] + 0
                new_vZ = original_vertex[2] + 0
            elif object_name == "HallWayToKitchen-Data61-L4.obj":
                new_X = pointCloud[:,0] - 25
                new_Z = pointCloud[:,2] + 50
                new_vX = original_vertex[0] - 25
                new_vZ = original_vertex[2] + 50
            elif object_name == "StairWell-Data61-L4.obj":
                new_X = pointCloud[:,0] + 25
                new_Z = pointCloud[:,2] + 50
                new_vX = original_vertex[0] + 25
                new_vZ = original_vertex[2] + 50
            else:
                print("Error:",obj_meta)

            new_Y = pointCloud[:,1]

            new_partial_pointcloud = np.stack((new_X,new_Z,new_Y)).T

            nbrs = NearestNeighbors(n_neighbors=min(2000,len(new_partial_pointcloud)), algorithm='kd_tree').fit(new_partial_pointcloud)

            # Get submap "centroids" by quantizing by 0.25m, i.e. round then unique
            if radius > 0.5:
                round_new_partial_pointcloud = min(0.5,radius)*100*np.around((0.01/min(0.5,radius))*new_partial_pointcloud,decimals=2)
                raw_partial_centroids = np.unique(round_new_partial_pointcloud[:,:2],axis = 0)
            else:
                raw_partial_centroids = [[new_vX, new_vZ]]

            """
            # Get submap "centers" by quantizing by 0.5m, i.e. round then unique
            round_ransac_pointcloud = 50*np.around(0.02*ransac_combined_pointcloud,decimals=2)
            unq_round_ransac_pointcloud = np.unique(round_ransac_pointcloud[:,:2],axis = 0)

            ransac_centroids_05 = unq_round_ransac_pointcloud
            """

            for northing, easting in raw_partial_centroids:

                for y_slice in np.arange(-2,2.1,0.5):

                    # Getting the points around our centroid defined by [northing, easting]
                    distances, indices = nbrs.kneighbors([[northing, easting, y_slice]])
                    submap_pointcloud = new_partial_pointcloud[indices[0,np.where(distances[0,:]<=spatial_span)[0]]]

                    if len(submap_pointcloud) == 0:
                        continue

                    # Centering and rescaling
                    submap_pointcloud = (submap_pointcloud - np.mean(submap_pointcloud, axis = 0))/spatial_span
                    #partial_true_lengths.append([radius,northing, easting, y_slice, len(submap_pointcloud)])

                    if len(submap_pointcloud) > 1024:
                        submap_pointcloud = submap_pointcloud[np.random.choice(len(submap_pointcloud),1024)]
                    elif len(submap_pointcloud) < 1024 and len(submap_pointcloud) >= 512 :
                        #print(i,submap_pointcloud.shape)
                        additional_pointcloud = submap_pointcloud[np.random.choice(len(submap_pointcloud),1024-len(submap_pointcloud))]
                        additional_pointcloud = additional_pointcloud + np.random.normal(0,0.05,additional_pointcloud.shape)
                        submap_pointcloud = np.concatenate((submap_pointcloud,additional_pointcloud),axis = 0)
                    elif len(submap_pointcloud) < 512 :
                        #print(i,submap_pointcloud.shape)
                        additional_pointcloud = submap_pointcloud[np.random.choice(len(submap_pointcloud),1024-len(submap_pointcloud), True)]
                        additional_pointcloud = additional_pointcloud + np.random.normal(0,0.05,additional_pointcloud.shape)
                        submap_pointcloud = np.concatenate((submap_pointcloud,additional_pointcloud),axis = 0)


                    timestamp = int(10**16*(time.time()))

                    csv_writer.writerow([timestamp,northing,easting,y_slice])

                    scipy.io.savemat(pointcloud_successive_path+'/{}.mat'.format(timestamp), mdict={'sub_pointcloud': submap_pointcloud.T})

                    """
                    with open("pnvlad/pointcloud_4m_0.25overlap/raw_{:05d}.bin".format(i),'wb') as byteFile:
                        byteFile.write(bytes(npf.asarray(submap_pointcloud.T,dtype=np.double)))

                    """
                    
        if i % 33 == 1:
            print("   Done with submap generation for iteration {} in {:.3f} seconds".format(i,time.time()-t1))
            t1 = time.time()

        csvfile.close()
        
    print(" Done with submap generation for radius {} in {:.3f} seconds".format(radius,time.time()-t0))
    t0 = time.time()
    
    g_successive_path = os.path.join(baseline_path,"g_successive_radius_"+str(radius))
    
    if not os.path.exists(g_successive_path): os.mkdir(g_successive_path)
        
    for i in range(int(releases/3)):
        
        per_release_point_collection = g_successive_point_collection_per_release[i]
        
        if len(per_release_point_collection) == 0: continue
        
        pointcloud_successive_path = os.path.join(g_successive_path,"release_"+str(i*3+1))
        if not os.path.exists(pointcloud_successive_path): os.mkdir(pointcloud_successive_path)
            
        csvfile = open(successive_path+"/release_"+str(i*3+1)+"_centroids_4m_0.5.csv",'w',newline = '')

        csv_writer = csv.writer(csvfile, delimiter = ',')
        csv_writer.writerow(['timestamp', 'northing', 'easting', 'alting'])
        
        for point_collection in per_release_point_collection:
            
            if len(point_collection) == 0:
                continue
                
            obj_meta = point_collection[0]
            pointCloud = point_collection[1]
            
            new_partial_pointcloud = []
            new_vX = []
            new_vZ = []

            original_vertex = obj_meta[2]
            object_name = obj_meta[1]

            if object_name == "Reception-Data61-L5.obj":
                new_X = pointCloud[:,0] + 50
                new_Z = pointCloud[:,2] + 0
                new_vX = original_vertex[0] + 50
                new_vZ = original_vertex[2] + 0
            elif object_name == "Driveway.obj":
                new_X = pointCloud[:,0] - 25
                new_Z = pointCloud[:,2] - 50
                new_vX = original_vertex[0] - 25
                new_vZ = original_vertex[2] - 50
            elif object_name == "Apartment.obj":
                new_X = pointCloud[:,0] + 25
                new_Z = pointCloud[:,2] - 50
                new_vX = original_vertex[0] + 25
                new_vZ = original_vertex[2] - 50
            elif object_name == "Workstations-Data61-L4.obj":
                new_X = pointCloud[:,0] - 50
                new_Z = pointCloud[:,2] + 0
                new_vX = original_vertex[0] - 50
                new_vZ = original_vertex[2] + 0
            elif object_name == "Kitchen-Data61-L4.obj":
                new_X = pointCloud[:,0] + 0
                new_Z = pointCloud[:,2] + 0
                new_vX = original_vertex[0] + 0
                new_vZ = original_vertex[2] + 0
            elif object_name == "HallWayToKitchen-Data61-L4.obj":
                new_X = pointCloud[:,0] - 25
                new_Z = pointCloud[:,2] + 50
                new_vX = original_vertex[0] - 25
                new_vZ = original_vertex[2] + 50
            elif object_name == "StairWell-Data61-L4.obj":
                new_X = pointCloud[:,0] + 25
                new_Z = pointCloud[:,2] + 50
                new_vX = original_vertex[0] + 25
                new_vZ = original_vertex[2] + 50
            else:
                print("Error:",obj_meta)

            new_Y = pointCloud[:,1]

            new_partial_pointcloud = np.stack((new_X,new_Z,new_Y)).T

            nbrs = NearestNeighbors(n_neighbors=min(2000,len(new_partial_pointcloud)), algorithm='kd_tree').fit(new_partial_pointcloud)

            # Get submap "centroids" by quantizing by 0.25m, i.e. round then unique
            if radius > 0.5:
                round_new_partial_pointcloud = min(0.5,radius)*100*np.around((0.01/min(0.5,radius))*new_partial_pointcloud,decimals=2)
                raw_partial_centroids = np.unique(round_new_partial_pointcloud[:,:2],axis = 0)
            else:
                raw_partial_centroids = [[new_vX, new_vZ]]

            """
            # Get submap "centers" by quantizing by 0.5m, i.e. round then unique
            round_ransac_pointcloud = 50*np.around(0.02*ransac_combined_pointcloud,decimals=2)
            unq_round_ransac_pointcloud = np.unique(round_ransac_pointcloud[:,:2],axis = 0)

            ransac_centroids_05 = unq_round_ransac_pointcloud
            """

            for northing, easting in raw_partial_centroids:

                for y_slice in np.arange(-2,2.1,0.5):

                    # Getting the points around our centroid defined by [northing, easting]
                    distances, indices = nbrs.kneighbors([[northing, easting, y_slice]])
                    submap_pointcloud = new_partial_pointcloud[indices[0,np.where(distances[0,:]<=spatial_span)[0]]]

                    if len(submap_pointcloud) == 0:
                        continue

                    # Centering and rescaling
                    submap_pointcloud = (submap_pointcloud - np.mean(submap_pointcloud, axis = 0))/spatial_span
                    #partial_true_lengths.append([radius,northing, easting, y_slice, len(submap_pointcloud)])

                    if len(submap_pointcloud) > 1024:
                        submap_pointcloud = submap_pointcloud[np.random.choice(len(submap_pointcloud),1024)]
                    elif len(submap_pointcloud) < 1024 and len(submap_pointcloud) >= 512 :
                        #print(i,submap_pointcloud.shape)
                        additional_pointcloud = submap_pointcloud[np.random.choice(len(submap_pointcloud),1024-len(submap_pointcloud))]
                        additional_pointcloud = additional_pointcloud + np.random.normal(0,0.05,additional_pointcloud.shape)
                        submap_pointcloud = np.concatenate((submap_pointcloud,additional_pointcloud),axis = 0)
                    elif len(submap_pointcloud) < 512 :
                        #print(i,submap_pointcloud.shape)
                        additional_pointcloud = submap_pointcloud[np.random.choice(len(submap_pointcloud),1024-len(submap_pointcloud), True)]
                        additional_pointcloud = additional_pointcloud + np.random.normal(0,0.05,additional_pointcloud.shape)
                        submap_pointcloud = np.concatenate((submap_pointcloud,additional_pointcloud),axis = 0)


                    timestamp = int(10**16*(time.time()))

                    csv_writer.writerow([timestamp,northing,easting,y_slice])

                    scipy.io.savemat(pointcloud_successive_path+'/{}.mat'.format(timestamp), mdict={'sub_pointcloud': submap_pointcloud.T})

                    """
                    with open("pnvlad/pointcloud_4m_0.25overlap/raw_{:05d}.bin".format(i),'wb') as byteFile:
                        byteFile.write(bytes(npf.asarray(submap_pointcloud.T,dtype=np.double)))

                    """
                    
        if i % 33 == 1:
            print("   Done with submap generation for iteration {} in {:.3f} seconds".format(i,time.time()-t1))
            t1 = time.time()

        csvfile.close()
        
    print(" Done with generalized submap generation for radius {} in {:.3f} seconds".format(radius,time.time()-t0))
    t0 = time.time()
    