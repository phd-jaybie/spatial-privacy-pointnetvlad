import pandas as pd
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import random

###Building database and query files for evaluation
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path= "../jittered_dataset_4096/"#"../partial_dataset/"

query_path = '../partial_radius_4096/'

all_folders=sorted(os.listdir(os.path.join(BASE_DIR,base_path)))
print(all_folders)

def check_in_test_set(northing, easting, points, x_width, y_width):
	in_test_set=False
	for point in points:
		if(point[0]-x_width<northing and northing< point[0]+x_width and point[1]-y_width<easting and easting<point[1]+y_width):
			in_test_set=True
			break
	return in_test_set
##########################################

def output_to_file(output, filename):
	with open(filename, 'wb') as handle:
	    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print("Done ", filename)
    
def get_sets_dict(filename):
	#[key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}},key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}}, ...}
	with open(filename, 'rb') as handle:
		trajectories = pickle.load(handle)
		print("Database Trajectories Loaded.")
		return trajectories
    
database_sets = get_sets_dict('3d_jittered_spaces_evaluation_database.pickle')

database_trees=[]
for folder in all_folders:
    print("Training tree for:",folder)
    df_database= pd.DataFrame(columns=['file','northing','easting','alting'])

    df_locations= pd.read_csv(os.path.join(base_path,folder,"pointcloud_centroids_4m_0.5.csv"),sep=',')
    #df_locations['timestamp']=folder+pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
    #df_locations=df_locations.rename(columns={'timestamp':'file'})
    for index, row in df_locations.iterrows():
        #df_test=df_test.append(row, ignore_index=True)
        df_database=df_database.append(row, ignore_index=True)

    database_tree = KDTree(df_database[['northing','easting','alting']])
    database_trees.append(database_tree)

def construct_query_sets(partial_path, pointcloud_fols, filename):#, partial_name):#, p, output_name):
	test_trees=[]
    
	#for folder in folders:
	#	print(folder)
	df_test= pd.DataFrame(columns=['file','northing','easting','alting'])
        
	df_locations= pd.read_csv(os.path.join(BASE_DIR,query_path,partial_path,filename),sep=',')
	#df_locations['timestamp']=folder+pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
	#df_locations=df_locations.rename(columns={'timestamp':'file'})
	for index, row in df_locations.iterrows():
		df_test=df_test.append(row, ignore_index=True)
		#elif(check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
		#df_test=df_test.append(row, ignore_index=True)
	test_tree = KDTree(df_test[['northing','easting','alting']])
	test_trees.append(test_tree)

	test_sets=[]
	#for folder in folders:
	test={} 
	df_locations['timestamp']=partial_path+pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
	df_locations=df_locations.rename(columns={'timestamp':'file'})
	for index,row in df_locations.iterrows():				
		#entire business district is in the test set
		test[len(test.keys())]={'query':row['file'],'northing':row['northing'],'easting':row['easting'],'alting':row['alting']}
		#elif(check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
		#test[len(test.keys())]={'query':row['file'],'northing':row['northing'],'easting':row['easting']}
	test_sets.append(test)
            
	print("Database (Tree) sets:",len(database_sets),"; Test (Tree) sets:",len(test_sets))    

	for i in range(len(database_sets)):
		tree=database_trees[i]
		for j in range(len(test_sets)):
			#if(i==j):
			#	continue
			for key in range(len(test_sets[j].keys())):
				coor=np.array([[test_sets[j][key]["northing"],test_sets[j][key]["easting"],test_sets[j][key]["alting"]]])
				index = tree.query_radius(coor, r=20) #r=4
				#indices of the positive matches in database i of each query (key) in test set j
				test_sets[j][key][i]=index[0].tolist()

    #'partial_spaces/'+partial_name+'_evaluation_database.pickle')
	output_to_file(test_sets, '3d_jittered_{}_evaluation_query.pickle'.format(partial_path))#'partial_spaces/'+partial_name+'_evaluation_query.pickle')

#For Oxford
#runs_folder = "oxford/"
#train_folders = all_folders[2::2]

for radius in np.arange(0.25,2.1,0.25):
    
    #partial_path = os.path.join(BASE_DIR,query_path+'partial_radius_'+str(radius)+"_4096")#	#folders.append(all_folders[index])
    partial_path = 'partial_radius_'+str(radius)+"_4096"#	#folders.append(all_folders[index])
    
    print(partial_path)
    construct_query_sets(partial_path, "/pointcloud_4m_bin/", "pointcloud_centroids_4m.csv")#, all_folders[index])
    
    

for radius in np.arange(0.25,2.1,0.25):
    
    partial_path = 'ransac_partial_radius_'+str(radius)+"_4096"#
    
    print(partial_path)
    construct_query_sets(partial_path, "/pointcloud_4m_bin/", "pointcloud_centroids_4m.csv")#, all_folders[index])

#print(all_folders)
#print("training:",train_folders)
#	
