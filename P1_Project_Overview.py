import streamlit as st
from PIL import Image

st.header("Project 02: Recommender System")
"""
Shopee là một hệ sinh thái thương mại “all in one”, trong đó có shopee.vn , là một website thương mại điện tử đứng top 1 của Việt Nam và khu vực Đông Nam Á.
"""
"\n"
"""
Trên trang này đã triển khai nhiều tiện ích hỗ trợ nâng cao trải nghiệm người dùng và họ muốn xây dựng nhiều tiện ích hơn nữa.
"""
"""
Hiện tại công ty này chưa triển khai Recommender System và đang có nhu cầu xây dựng Recommendation System cho một hoặc một số nhóm hàng hóa trên shopee.vn giúp đề xuất và gợi ý cho người dùng/ khách hàng
"""
image = Image.open('shopee.png')

st.image(image, caption='shopee.vn')
