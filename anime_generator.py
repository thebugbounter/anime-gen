import streamlit as st
import tensorflow as tf 

def generate(num, confidence):
    gen = tf.keras.models.load_model("anime_gen.h5")
    dis = tf.keras.models.load_model("anime_dis.h5")
    cnt = num
    while(cnt > 0):
        image = gen(tf.random.normal((1, 100)), training=False)
        res = dis(image, training=False)
        if res > confidence:
            image = tf.keras.preprocessing.image.array_to_img(image[0, :, :, :])
            st.image(image)
            cnt -= 1

def homepage():
    st.write(
        """
        ## Welcome to the Anime Face Generator!
        """
    )

    cols = st.columns(9)
    for i in range(9):
        cols[i].image(f'samples/sample{i+1}.jpg')

    st.write(
        """
        This application allows you to generate random anime faces using a Generative Adversarial Network (GAN). 
        The GAN model has been trained on the [this dataste](https://www.kaggle.com/datasets/soumikrakshit/anime-faces), which contains 64x64 colored images of random anime faces.


        You can use this tool to generate new random anime faces. There are some errors in the results but the images are still pretty good and realistic.

        """
    )  

    st.divider()
    st.write(""" 
        ### The Generator:

        You can use the generator by simply opening the sidebar and selecting the "Generator" option. After that you just have to set the confidence threshold.
        The confidence threshold is threshold the likelihood of generated image to be similar to the original dataset. 
        After that, you have to specify the number of images you want to generate and click on generate button.

    """)


def generator():
    gen = tf.keras.models.load_model("anime_gen.h5")
    dis = tf.keras.models.load_model("anime_dis.h5")
    num = st.number_input("Number of images to generate:", min_value=1, value=1, step=1)
    confidence = st.number_input("Set Confidence Threshold:", min_value=0.0, max_value=0.75, value=0.5, step=0.05)
    cnt = num
    if st.button("Generate"):
        generate(cnt, confidence)



st.title("Anime Face Generator")
st.divider()
bar = st.sidebar
options = ('Home',"Generator")
option = bar.selectbox("Select Page", options)

if option == "Home":
    homepage()
    bar.text("Links:")
    bar.info("[Visit my GitHub account](https://github.com/Lackyjian)")
    bar.info("[Visit my LinkedIn account](https://www.linkedin.com/in/lakshay-jain-ab1895281/)")
    bar.info("[Visit my Kaggle account](https://www.kaggle.com/lakshayjain611)")

if option == "Generator":
    generator()
    bar.text("Links:")
    bar.info("[Visit my GitHub account](https://github.com/Lackyjian)")
    bar.info("[Visit my LinkedIn account](https://www.linkedin.com/in/lakshay-jain-ab1895281/)")
    bar.info("[Visit my Kaggle account](https://www.kaggle.com/lakshayjain611)")