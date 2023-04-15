import streamlit as st
import pandas as pd
import pickle

############################################################################################

#def app():
st.title("Collaborative Filtering")
Review = pd.read_csv("data_full_info_12_4_2023.csv", index_col=0)
Product = pd.read_csv('data_full_info_12_4_2023.csv', encoding="utf8", index_col=0)
pd.set_option('display.max_colwidth', None) # need this option to make sure cell content is displayed in full
#Product['short_name'] = Product['name'].str.split('-').str[0]
Product['short_name'] = Product['product_name'].str.split('-').str[0]

############################################################################################

# Define functions
##### TAKE URL OF AN IMAGE #####
def fetch_image(idx):
    selected_product = Product['image'].iloc[[idx]].reset_index(drop=True)
    url = selected_product[0]
    return url

##### CHECK CUSTOMER SIMILARITIES BY SURPRISE MODEL WITH BASELINE ONLY AND RETURN NAMES & IMAGES OF TOP PRODUCTS WITH HIGHEST ESTIMATED RATING #####
def sursim_check(customer_id,model,n):
    # Get estimate score for list of product ids
    df = Review[['product_id']]
    df['rating'] = df['product_id'].apply(lambda x: model.predict(customer_id, x).est)
    df = df.sort_values(by=['rating'], ascending=False)
    # Drop duplicates, if any
    df = df.drop_duplicates()
    output = df.merge(Product,left_on='product_id', right_on='item_id')
    recommended_names = output['short_name'].values.tolist()
    recommended_images = output['image'].values.tolist()
    return recommended_names, recommended_images

############################################################################################

# Input customer id
number = st.number_input("Input customer id:", min_value=0)
st.write("Your customer id: ", number)

# Choose maximum number of products that system will recommend
n = st.slider(
    'Select maximum number of products similar to the above that you want system to recommend (from 1 to 6)',
    1, 6, 3)
st.write('Maximum number of products to recommend:', n)

# 'Recommend' button
if st.button('Recommend'):
    Sur_model = pickle.load(open('Sur_model.pkl','rb'))
    customer_id = number
    model = Sur_model
    names, images = sursim_check(customer_id,model,n)
    names = names[:n]
    images = images[:n]
    cols = st.columns(n)
    for c in range(n):
        with cols[c]:
            st.image(images[c], caption = names[c])
