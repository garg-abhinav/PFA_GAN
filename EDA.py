import scipy.io
import pandas as pd
import random
import pprint

import matplotlib.image as mpimg
import pickle


data_dict = scipy.io.loadmat('data/celebrity2000_meta.mat')
celebrity_data = data_dict['celebrityImageData']

celebrity_data_name = data_dict['celebrityData']



actor_age = []
actor_image_url = []
actor_id = []
actor_name = []

for i in celebrity_data[0][0][0]:
    actor_age.append(i.tolist()[0])

for i in celebrity_data[0][0][7]:
    actor_image_url.append(i.tolist()[0][0])

for i in celebrity_data_name[0][0][0]:
    actor_name.append(i.tolist()[0][0])

for i in celebrity_data[0][0][1]:
    actor_id.append(i.tolist()[0])

print(len(actor_age))
print(len(actor_image_url))
print(len(actor_name))

print(actor_name[0])
print(len(actor_id))

actor_data_list = {'id':actor_id,'age':actor_age, 'image':actor_image_url}

actor_image_df=pd.DataFrame(actor_data_list)

#print(actor_image_df.head(20))

def getname(x):

    y=''.join(filter(str.isalpha,x))[:-3]

    return(y)
#
# print(actor_image_df.columns)
# print(actor_image_df['image'])
actor_image_df['actor_name']=actor_image_df['image'].apply(lambda x: getname(x))

#print(actor_image_df.head(20))

actor_df=actor_image_df[['id','age','actor_name']].drop_duplicates().reset_index()
#print(actor_df.head(20))

#print(len(actor_df))

actor_list=actor_image_df['actor_name'].to_list()

actor_list_filtered = random.sample(actor_list, 800)

#print(actor_list_filtered[0])



keys=actor_image_df['actor_name'].to_list()
values=actor_image_df['image'].to_list()


#actor_dict = dict(zip(keys, values))



#filtered_dict = {k:v for k,v in actor_dict.items() if k in actor_list_filtered}

#pprint.pprint(actor_dict['RobinWilliams'])
#pprint.pprint(filtered_dict)



final_image=actor_image_df[actor_image_df['actor_name'].isin(actor_list_filtered)]
print(final_image)
print(final_image.head())
print(final_image.shape)

final_actor_image_list=final_image.index.to_list()

with open('data/image_urls.pkl','rb') as f:
    urls = pickle.load(f)
filtered_urls=[urls[i] for i in final_actor_image_list]
print(len(filtered_urls))

with open('data/image_ages.pkl','rb') as f:
    ages = pickle.load(f)
filtered_ages=[ages[i] for i in final_actor_image_list]
print(len(filtered_ages))

with open('data/image_urls_v2.pkl', 'wb') as f:
    pickle.dump(filtered_urls, f)

with open('data/image_ages_v2.pkl', 'wb') as f:
    pickle.dump(filtered_ages, f)