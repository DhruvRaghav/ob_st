import streamlit as st
import skimage.io as io
from face_2D_detect import *
import blur
import os
import time
# from detect import *
from face_2D_detect_blur import run

from face_2D_detect_1 import run5
from face_2D_detect_blur_1 import run6




st.set_page_config(
    page_title="GUARDIANS OF THE STREET ",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Guardians of the Street")

# Create necessary directories if they don't exist
os.makedirs('API_DATA/brand_box', exist_ok=True)
os.makedirs('API_DATA/brand_blur', exist_ok=True)
os.makedirs('API_DATA/brand_draw', exist_ok=True)
os.makedirs('API_DATA/object_box', exist_ok=True)
os.makedirs('API_DATA/object_blur', exist_ok=True)
os.makedirs('API_DATA/object_draw', exist_ok=True)
os.makedirs('API_DATA/brand_label', exist_ok=True)

os.makedirs('API_DATA/Pothole_box', exist_ok=True)
os.makedirs('API_DATA/Pothole_blur', exist_ok=True)
os.makedirs('API_DATA/Pothole_draw', exist_ok=True)

def delete_image(file_path):
    try:
        os.remove(file_path)
        st.success(f"Image {file_path} deleted successfully!")
    except Exception as e:
        st.error(f"Error deleting image: {e}")


def process_brand_image(option, files):
    if files is not None:
        for file in files:
            img = io.imread(file)
            # Save the uploaded image
            save_path = f'API_DATA/{option}/{file.name}'
            io.imsave(save_path, img)

            # Display Image Statistics
            st.subheader("Image Statistics")
            st.write(f"Dimensions: {img.shape[0]} x {img.shape[1]}")
            st.write(f"Channels: {img.shape[2]}")

            # # Display the uploaded image
            # st.subheader("Uploaded Image")
            # st.image(img, caption="Original Image", use_column_width=True)

            # Progress Bar
            progress_bar = st.progress(0)

            # Loading Spinner
            with st.spinner(f"Processing {file.name}..."):
                for i in range(1, 101):
                    time.sleep(0.05)
                    progress_bar.progress(i)

            # Processing Time
            st.subheader("Processing Time")
            st.write(f"Time taken: {time.process_time()} seconds")

            # Display Original vs Processed Images Side-by-Side
            col1, col2 = st.columns(2)
            col1.subheader("Original Image")
            col1.image(img, caption="Original Image", use_column_width=True)

            col2.subheader("Processed Image")
            # Result Container
            result_container = col2.empty()

            if option in ['pothole_box']:
                print("okay hi")
                result = run5(source=f'API_DATA/{option}/{file.name}', nosave=True, box=True, device='cpu')
                st.json(result)
            elif option in ['pothole_blur']:
                result = run6(source=f'API_DATA/{option}', nosave=True, box=True, device='cpu')
                blur.rectblur_NEW(f'API_DATA/{option}/{file.name}', result, option,file.name)
                result_image_path = f'C:/dataset/object_detection-main/API_DATA/{option}/{file.name}'
                result_image = io.imread(result_image_path)
                result_container.image(result_image, caption="Result", use_column_width=True)

                # Delete the processed image
                delete_image(result_image_path)

            elif option in ['pothole_draw']:
                result_path = run5(source=f'API_DATA/{option}/{file.name}', nosave=True, img=True, device='cpu',save_dir=f'C:/dataset/object_detection-main/API_DATA/{option}/{file.name}')
                result_image = io.imread(result_path)
                result_container.image(result_image, caption="Result with Bounding Boxes", use_column_width=True)

                # Delete the processed image
                delete_image(result_path)


            # elif option in ['Pothole_label']:
            #     result_path = run4(source=f'API_DATA/{option}/{file.name}', nosave=True,  device='cpu')
            #     result_image = io.imread(result_path)
            #     result_container.image(result_image, caption="Result with Bounding Boxes", use_column_width=True)





                # Delete the processed image
                # delete_image(result_path)



def process_object_image(option, files):
    if files is not None:
        for file in files:
            img = io.imread(file)
            print("1")
            # Save the uploaded image
            save_path = f'API_DATA/{option}/{file.name}'
            io.imsave(save_path, img)
            print("2")
            # Display Image Statistics
            st.subheader("Image Statistics")
            st.write(f"Dimensions: {img.shape[0]} x {img.shape[1]}")
            st.write(f"Channels: {img.shape[2]}")
            print("3")
            # # Display the uploaded image
            # st.subheader("Uploaded Image")
            # st.image(img, caption="Original Image", use_column_width=True)

            # Progress Bar
            progress_bar = st.progress(0)
            print("4")
            # Loading Spinner
            with st.spinner(f"Processing {file.name}..."):
                for i in range(1, 101):
                    time.sleep(0.05)
                    progress_bar.progress(i)

            # Processing Time
            st.subheader("Processing Time")
            st.write(f"Time taken: {time.process_time()} seconds")

            # Display Original vs Processed Images Side-by-Side
            col1, col2 = st.columns(2)
            col1.subheader("Original Image")
            col1.image(img, caption="Original Image", use_column_width=True)

            col2.subheader("Processed Image")
            # Result Container
            result_container = col2.empty()

            if option in ['object_box']:

                result = run2(source=f'API_DATA/{option}/{file.name}', nosave=True, box=True, device='cpu')
                st.json(result)


            elif option in ['object_blur']:

                result = run(source=f'API_DATA/{option}', nosave=True, box=True, device='cpu')
                blur.rectblur_NEW(f'API_DATA/{option}/{file.name}', result, option, file.name)
                result_image_path = f'C:/dataset/object_detection-main/API_DATA/{option}/{file.name}'
                result_image = io.imread(result_image_path)
                result_container.image(result_image, caption="Result", use_column_width=True)
                print("5")
                # Delete the processed image
                delete_image(result_image_path)

            elif option in ['object_draw']:
                print("6")
                result_path = run2(source=f'API_DATA/{option}/{file.name}', nosave=True, img=True, device='cpu',
                                   save_dir=f'C:/dataset/object_detection-main/API_DATA/{option}/{file.name}')
                result_image = io.imread(result_path)
                result_container.image(result_image, caption="Result with Bounding Boxes", use_column_width=True)

                # Delete the processed image
                delete_image(result_path)







def main():
    st.sidebar.header("Choose an option")

    # Create a dropdown for brand options
    Pothole_option = st.sidebar.selectbox("Pothole Detection Options", ["Select an option", "Pothole Box", "Pothole Blur", "Pothole Draw"])
    # brand_options = ["Select an option", "Brand Box", "Brand Blur", "Brand Draw", "Brand Label"]

    # Create a dropdown for general object detection
    general_option = st.sidebar.selectbox("General Object Detection",
                                          ["Select an option", "Object Box", "Object Blur", "Object Draw"])

    files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if st.button("Process"):
        if Pothole_option == "Select an option" and general_option == "Select an option":
            st.warning("Please select a Pothole or general object detection option.")

        elif Pothole_option != "Select an option":
            process_brand_image(Pothole_option.lower().replace(" ", "_"), files)

        elif general_option != "Select an option":
            print("0")
            process_object_image(general_option.lower().replace(" ", "_"), files)
            print("finished")


if __name__ == "__main__":
    main()
