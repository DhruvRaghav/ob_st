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
os.makedirs('API_DATA/object_box', exist_ok=True)
os.makedirs('API_DATA/object_blur', exist_ok=True)
os.makedirs('API_DATA/object_draw', exist_ok=True)

def process_brand_image(option, files):
    if files is not None:
        for file in files:
            img = io.imread(file)
            # Save the uploaded image
            save_path = f'API_DATA/{option}/{file.name}'
            io.imsave(save_path, img)

            if option in ['brand_box']:

                result = run(source=f'API_DATA/{option}', nosave=True, box=True, device='cpu')
                st.json(result)
            elif option in ['brand_blur']:
                result = run(source=f'API_DATA/{option}', nosave=True, box=True, device='cpu')
                blur.rectblur_NEW(f'API_DATA/{option}/{file.name}', result, option)
                result_image_path = f'C:/dataset/object_detection-main/API_DATA/{option}/{file.name}'
                result_image = io.imread(result_image_path)
                st.image(result_image, caption="Result", use_column_width=True)
            elif option in ['brand_draw']:
                result_path = run(source=f'API_DATA/{option}', nosave=True, img=True, device='cpu')
                result_image = io.imread(result_path)
                st.image(result_image, caption="Result with Bounding Boxes", use_column_width=True)

def process_object_image(option, files):
    if files is not None:
        for file in files:
            img = io.imread(file)
            # Save the uploaded image
            save_path = f'API_DATA/{option}/{file.name}'
            io.imsave(save_path, img)

            if option in ['object_box']:
                result = run(source=f'API_DATA/{option}', nosave=True, box=True, device='cpu')
                st.json(result)
            elif option in ['object_blur']:
                result = run(source=f'API_DATA/{option}', nosave=True, box=True, device='cpu')
                blur.rectblur_NEW(f'API_DATA/{option}/{file.name}', result, option)
                result_image_path = f'C:/dataset/object_detection-main/API_DATA/{option}/{file.name}'
                result_image = io.imread(result_image_path)
                st.image(result_image, caption="Result", use_column_width=True)
            elif option in ['object_draw']:
                result_path = run(source=f'API_DATA/{option}', nosave=True, img=True, device='cpu')
                result_image = io.imread(result_path)
                st.image(result_image, caption="Result with Bounding Boxes", use_column_width=True)

def main():
    st.sidebar.header("Choose an option")

    # Create a dropdown for brand options
    brand_option = st.sidebar.selectbox("Brand Options", ["Select an option", "Brand Box", "Brand Blur", "Brand Draw"])

    # Create a dropdown for general object detection
    general_option = st.sidebar.selectbox("General Object Detection",
                                          ["Select an option", "Object Box", "Object Blur", "Object Draw"])

    files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if st.button("Process"):
        if brand_option == "Select an option" and general_option == "Select an option":
            st.warning("Please select a brand or general object detection option.")

        elif brand_option != "Select an option":
            process_brand_image(brand_option.lower().replace(" ", "_"), files)

        elif general_option != "Select an option":
            process_object_image(general_option.lower().replace(" ", "_"), files)

if __name__ == "__main__":
    main()
