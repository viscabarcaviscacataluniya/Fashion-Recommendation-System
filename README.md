# Fashion-Recommendation-System

👗 Fashion Recommender System
A deep learning-based fashion recommendation system that suggests visually similar clothing items using a ResNet50 model and K-Nearest Neighbors (KNN) algorithm.

📌 Overview
This project leverages a pre-trained ResNet50 model to extract feature embeddings from clothing images. By employing the K-Nearest Neighbors algorithm, it identifies and recommends fashion items that are visually similar to a given input image.

Dataset : https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

🧠 How It Works
Feature Extraction: Utilizes ResNet50 (excluding the top classification layer) combined with Global Max Pooling to extract 2048-dimensional feature vectors from images.

Embedding Storage: Precomputed embeddings are stored in embeddings.pkl, and corresponding image filenames are stored in filenames.pkl.

Similarity Search: Implements KNN with Euclidean distance to find the top 5 similar items to the input image.

Image Preprocessing: Input images are resized to 224x224 pixels and preprocessed to match the ResNet50 input requirements.

📁 Project Structure
perl
Copy
Edit
fashion-recommender-system/
├── app.py          # Streamlit app for user interaction

├── main.py         # Script to generate embeddings from dataset

├── test.py         # Script to test recommendations for a sample image

├── embeddings.pkl  # Precomputed feature embeddings

├── filenames.pkl   # Corresponding image filenames

├── sample/         # Directory containing sample images

└── README.md       # Project documentation
🚀 Getting Started
Prerequisites
Python 3.6 or higher

Required Python packages:

tensorflow

numpy

opencv-python

scikit-learn

streamlit

Install the dependencies using pip:

bash
Copy
Edit
pip install tensorflow numpy opencv-python scikit-learn streamlit
Running the Application
Generate Embeddings (if not already available):

bash
Copy
Edit
python main.py
This script processes the dataset images, extracts features using ResNet50, and saves the embeddings and filenames.

Test Recommendations for a Sample Image:

bash
Copy
Edit
python test.py
This script loads a sample image, computes its embedding, and displays the top 5 similar images from the dataset.

Launch the Streamlit App:

bash
Copy
Edit
streamlit run app.py
This will open a web interface where you can upload an image and view recommended fashion items.

🧪 Example
Suppose you have a sample image sample/shirt.jpg. Running test.py will output the indices of the top 5 similar images:

bash
Copy
Edit
[10 23 45 67 89]
These indices correspond to the most visually similar items in the dataset.

📸 Sample Output
Note: Include sample images or screenshots here to showcase the recommendations.

🤝 Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

