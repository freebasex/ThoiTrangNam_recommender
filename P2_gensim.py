import streamlit as st
import pandas as pd
from gensim import corpora, models, similarities
from underthesea import word_tokenize, pos_tag, sent_tokenize
import re
import regex
import emoji

##### TEXT PROCESSING #####
def process_text(document):
    # Change to lower text
    document = document.lower()
    # Remove line break
    document = document.replace(r'[\r\n]+', ' ')
    # Change / by white space
    document = document.replace('/', ' ') 
    # Change , by white space
    document = document.replace(',', ' ') 
    # Remove punctuations
    document = document.replace('[^\w\s]', '')
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for char in punctuation:
        document = document.replace(char, '')
    # Remove numbers, keep words
    document = document.replace('[\w]*\d+[\w]*', '')
    document = document.replace('[0-9]+', '')   
    # Replace mutiple spaces by single space
    document = document.replace('[\s]{2,}', ' ')
    # Word_tokenize
    document = word_tokenize(document, format="text")   
    # Pos_tag
    document = pos_tag(document)    
    # Remove stopwords
    STOP_WORD_FILE = 'vietnamese-stopwords.txt'   
    with open(STOP_WORD_FILE, 'r', encoding='utf-8') as file:
        stop_words = file.read()
    stop_words = stop_words.split()  
    document = [[word[0] for word in document if not word[0] in stop_words]] 
    return document

def clean_text(text):
    text_clean = str(text).lower()
    #Loại bỏ thông tin liên quan đến chi tiết như Xuất xứ, danh mục, kho hàng
    if 'danh mục\n' in text_clean: 
        text_clean = text_clean[text_clean.index("danh mục\n"):]
    elif 'thông tin sản phẩm\n' in text_clean: 
        text_clean = text_clean[text_clean.index("thông tin sản phẩm\n"):]
    elif "mô tả sản phẩm\n" in text_clean: 
        text_clean = text_clean[text_clean.index("mô tả sản phẩm\n"):]
    elif 'danh mục\n' in text_clean:
        text_clean = text_clean[text_clean.index("danh mục\n"):]
    elif 'shopee\n' in text_clean:
        text_clean = text_clean[text_clean.index("shopee\n"):]
    elif 'xuất xứ\n' in text_clean:
        text_clean = text_clean[text_clean.index("xuất xứ\n"):]
    elif 'kho hàng\n' in text_clean:
        text_clean = text_clean[text_clean.index("kho hàng\n"):]
    elif 'thời trang nam\n' in text_clean:
        text_clean = text_clean[text_clean.index("thời trang nam\n"):]
    elif "\n\n" in text_clean: 
        text_clean = text_clean[text_clean.index("\n\n") + 4:]
    elif "\ngửi từ\n" in text_clean: 
        text_clean = text_clean[text_clean.index("\ngửi từ\n") + len("\ngửi từ\n"):]
    elif "\ngửi từ\n" in text_clean: 
        text_clean = text_clean[text_clean.index("\ngửi từ\n") + len("\ngửi từ\n"):]
    #Loại bỏ phần size
    text_clean = re.sub(r"\nsize[^\n]*","",text_clean)

    #Loại bỏ các hastag
    text_clean = re.sub(r"#[^#]*","", text_clean)
    #Loại bỏ các ký tự không hợp lệ
    text_clean = re.sub(r"\n", " ", text_clean) 
    text_clean = emoji.replace_emoji(text_clean)
    text_clean = re.sub('[\.\:\,\-\-\-\+\d\!\%\...\.\"\*\>\<\^\&\/\[\]\(\)\=\~]',' ', text_clean)
    #Loại bỏ các từ không cần thiết
    text_clean = re.sub('\ss\s|\sm\s|\sl\s|\sxl|xxl|xxxl|xxxxl|2x1|3x1|4xl|size|\smm\s|\scm\s|\sm\s|\sg\s|\skg\s',' ', text_clean)
    #Loại bỏ khoảng trắng thừa
    text_clean = re.sub('\s+',' ', text_clean)
    
    text_clean = re.sub('[\s]{2,}', ' ', text_clean)
    text_clean = re.sub('[\s]{3,}', ' ', text_clean)
    text_clean = re.sub('^[\s]{1,}', '', text_clean)
    text_clean = re.sub('[\s]{1,}$', '', text_clean)
    text_clean = re.sub('[^\w\s]', '', text_clean)    
    text_clean = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', '',text_clean)
    text_clean = re.sub('danh mục shopee thời trang nam', ' ', text_clean)
    return text_clean
############################################################################################
#def app():
st.title("Content Based Filtering")
Product = pd.read_csv('data_full_info_12_4_2023.csv', encoding="utf8", index_col=0)
pd.set_option('display.max_colwidth', None) # need this option to make sure cell content is displayed in full
Product['short_name'] = Product['product_name'].apply(clean_text)

product_map = Product.iloc[:,[0,-1]]
product_list = product_map['short_name'].values


############################################################################################

# Define functions to use for both methods

##### TAKE URL OF AN IMAGE #####
def fetch_image(idx):
    selected_product = Product['image'].iloc[[idx]].reset_index(drop=True)
    url = selected_product[0]
    #url = 'https://cf.shopee.vn/file/c7ea4c6574dc79be6b266b9d69b49abc_tn'
    return url

##### CHECK PRODUCT SIMILARITIES BY GENSIM MODEL AND RETURN NAMES & IMAGES OF TOP PRODUCTS WITH HIGHEST SIMILARITY INDEX #####
def gensim_check(document, dictionary, tfidf, index, n):
    # Convert document to lower text
    document = document.lower().split()
    # Convert document to dictionary according to reference
    vector = dictionary.doc2bow(document)
    # Index similarities of document versus reference
    sim = index[tfidf[vector]]
    # Print output
    list_id = []
    list_score = []
    for i in range(len(sim)):
        list_id.append(i)
        list_score.append(sim[i])
    # Create dataframe to store output
    result = pd.DataFrame({'id':list_id, 'score':list_score})
    # Extract number of products according to users' input (as we choose product from list, extract n+1 items, including the 1st chosen one):
    n_highest_score = result.sort_values(by='score', ascending=False).head(n + 1)
    # Extract product_id of above request
    id_tolist = list(n_highest_score['id'])
    recommended_names = []
    recommended_images = []
    for i in id_tolist:
        # Fetch the product names
        #product_name = Product['short_name'].iloc[[i]]
        product_name = Product.iloc[i,2]
        #recommended_names.append(product_name.to_string(index=False))
        recommended_names.append(product_name)
        # Fetch the product images

        recommended_images.append(fetch_image(i))
    return recommended_names, recommended_images


############################################################################################

# Define separate page to demo each method
##### CONTENT_BASED FILTERING BY FIXED LIST #####
def filter_list():
    # Markdown name of Content_based method
    st.markdown("### By Fixed List")

    # Select product from list
    selected_idx = st.selectbox("Select product to view: ", range(len(product_list)), format_func=lambda x: product_list[x])

    # Fetch image of selected product
    idx = selected_idx
    
    st.image(fetch_image(idx))

    # 'Recommend' button
    if st.button('Recommend'):
        selection = Product.iloc[[idx]]
        selection_str = selection['product_name_wt'].to_string(index=False)
        document = selection_str
        dictionary = corpora.dictionary.Dictionary.load("dictionary.dictionary")
        tfidf = models.tfidfmodel.TfidfModel.load("tfidf.tfidfmodel")
        index = similarities.docsim.SparseMatrixSimilarity.load("index.docsim")
        names, images = gensim_check(document, dictionary, tfidf, index, 6)
        names = names[1:-1]
        images = images[1:-1]
        cols = st.columns(5)
        for c in range(5):
            with cols[c]:
                st.image(images[c], caption = names[c])
    
##### CONTENT_BASED FILTERING BY INPUTING DESCRIPTION #####
def input_description():
    # Markdown name of Content_based method
    st.markdown("### By Inputing Description")

    # input product description
    text_input = st.text_input(
        "Input product description to search: "
    )

    if text_input:
        st.write("Your product description: ", text_input)

    # Choose maximum number of products that system will recommend
    #n = st.slider(
    #'Select maximum number of products similar to the above that you want system to recommend (from 1 to 6)',
    #1, 6, 3)
    #st.write('Maximum number of products to recommend:', n)
    
    #st.write("Your product description: ", text_input)
    #document = '_'.join(map(str,process_text(text_input)))
    document = text_input
    #st.write("Your product description: ", document)

    # 'Recommend' button
    if st.button('Recommend'):
   
        dictionary = corpora.dictionary.Dictionary.load("dictionary.dictionary")
        tfidf = models.tfidfmodel.TfidfModel.load("tfidf.tfidfmodel")
        index = similarities.docsim.SparseMatrixSimilarity.load("index.docsim")
        names, images = gensim_check(document, dictionary, tfidf, index, 6)
        #names = names[:5]
        #images = images[:5]

        names = names[1:-1]
        images = images[1:-1]
        cols = st.columns(5)
        for c in range(5):
            with cols[c]:
                st.image(images[c], caption = names[c])

    ##### CALLING PAGE  #####

page_names_to_funcs = {
    "Filter List": filter_list,
    "Input Description": input_description
    }
selected_page = st.sidebar.selectbox("Select method", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()