import streamlit as st
import skimage.io as io
from face_2D_detect import run
import blur
import os

st.title("Guardians of the Street")

# Create necessary directories if they don't exist
os.makedirs('API_DATA/brand_box', exist_ok=True)
os.makedirs('API_DATA/brand_blur', exist_ok=True)
os.makedirs('API_DATA/brand_draw', exist_ok=True)

def process_image(option, file):
    if file is not None:
        img = io.imread(file)
        # Save the uploaded image
        save_path = f'API_DATA/{option}/test.png'
        io.imsave(save_path, img)

        if option == 'brand_box':
            result = run(source='API_DATA/brand_box', nosave=True, box=True, device='cpu')
            st.json(result)
        elif option == 'brand_blur':
            result = run(source='API_DATA/brand_blur', nosave=True, box=True, device='cpu')
            blur.rectblur_NEW('API_DATA/brand_blur/test.png', result)
            result_image_path = 'C:/dataset/object_detection-main/API_DATA/bluroutput/test.png'
            result_image = io.imread(result_image_path)
            st.image(result_image, caption="Result", use_column_width=True)
        elif option == 'brand_draw':
            result_path = run(source='API_DATA/brand_draw', nosave=True, img=True, device='cpu')
            result_image = io.imread(result_path)
            st.image(result_image, caption="Result with Bounding Boxes", use_column_width=True)


def main():
    st.sidebar.header("Choose an option")
    option = st.sidebar.selectbox("", ["Select an option", "Brand Box", "Brand Blur", "Brand Draw"])

    file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if st.button("Process"):
        if option == "Select an option":
            st.warning("Please select an option.")
        else:
            process_image(option.lower().replace(" ", "_"), file)

if __name__ == "__main__":
    main()
