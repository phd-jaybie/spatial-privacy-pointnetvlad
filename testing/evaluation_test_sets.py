import pandas as pd
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import random

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


def construct_query_and_database_sets(base_path, folders, pointcloud_fols, filename):#, partial_name):#, p, output_name):
	database_trees=[]
	test_trees=[]
	for folder in folders:
		print(folder)
		df_database= pd.DataFrame(columns=['file','northing','easting'])
		df_test= pd.DataFrame(columns=['file','northing','easting'])
		
		df_locations= pd.read_csv(os.path.join(base_path,folder,filename),sep=',')
		df_locations['timestamp']=folder+pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
		df_locations=df_locations.rename(columns={'timestamp':'file'})
		for index, row in df_locations.iterrows():
			#entire business district is in the test set
			if folder in test_folders:
				#print("test",folder)
				df_test=df_test.append(row, ignore_index=True)
			#elif(check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
			#df_test=df_test.append(row, ignore_index=True)
			df_database=df_database.append(row, ignore_index=True)

		database_tree = KDTree(df_database[['northing','easting']])
		database_trees.append(database_tree)
		if folder in test_folders:
			test_tree = KDTree(df_test[['northing','easting']])
			test_trees.append(test_tree)

	test_sets=[]
	database_sets=[]
	for folder in folders:
		database={}
		test={} 
		df_locations= pd.read_csv(os.path.join(base_path,folder,filename),sep=',')
		df_locations['timestamp']=folder+pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
		df_locations=df_locations.rename(columns={'timestamp':'file'})
		for index,row in df_locations.iterrows():				
			#entire business district is in the test set
			if folder in test_folders:
				#print("test",folder)
				test[len(test.keys())]={'query':row['file'],'northing':row['northing'],'easting':row['easting']}
			#elif(check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
			#test[len(test.keys())]={'query':row['file'],'northing':row['northing'],'easting':row['easting']}
			database[len(database.keys())]={'query':row['file'],'northing':row['northing'],'easting':row['easting']}
		database_sets.append(database)
		if folder in test_folders:
			test_sets.append(test)

	for i in range(len(database_sets)):
		tree=database_trees[i]
		for j in range(len(test_sets)):
			if(i==j):
				continue
			for key in range(len(test_sets[j].keys())):
				coor=np.array([[test_sets[j][key]["northing"],test_sets[j][key]["easting"]]])
				index = tree.query_radius(coor, r=4)
				#indices of the positive matches in database i of each query (key) in test set j
				test_sets[j][key][i]=index[0].tolist()

	output_to_file(database_sets, 'vertical_spaces_evaluation_database.pickle')#'partial_spaces/'+partial_name+'_evaluation_database.pickle')
	output_to_file(test_sets, 'vertical_spaces_evaluation_query.pickle')#'partial_spaces/'+partial_name+'_evaluation_query.pickle')

###Building database and query files for evaluation
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path= "../vertical_dataset/"#"../partial_dataset/"

#For Oxford
#runs_folder = "oxford/"
all_folders=sorted(os.listdir(os.path.join(BASE_DIR,base_path)))

test_folders = all_folders[1::2]

#for folder in all_folders[4:]:
#	#folders.append(all_folders[index])
print(all_folders)
print(test_folders)
#	
construct_query_and_database_sets(base_path, all_folders, "/pointcloud_4m_0.5_bin/", "pointcloud_centroids_4m_0.5.csv")#, all_folders[index])
