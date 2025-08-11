# 🛍️ Customer Personality Segmentation

[![Python](https://img.shields.io/badge/python-3.7-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-47A248?logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![AWS](https://img.shields.io/badge/AWS-232F3E?logo=amazon-aws&logoColor=white)](https://aws.amazon.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 📌 Overview  
In today’s competitive retail landscape, understanding customer personalities is key to delivering personalized experiences and targeted marketing strategies. This project harnesses machine learning for **Customer Personality Segmentation** by clustering users based on demographics and purchase behavior, then **predicting their segment dynamically** to serve real-time insights.

Use cases include:  
- 🎯 Personalized marketing campaigns  
- 📈 Enhanced customer retention  
- 🚀 Optimized product recommendations  

---

## 🎯 Problem Statement  
Handling large-scale customer data — including demographics and purchase histories — presents a challenge in effectively segmenting customers for better marketing and service.

Goal:  
> Efficiently cluster customers into meaningful groups and build a predictive system to classify new customers instantly.

Hybrid approach:  
- **Unsupervised learning:** K-Means clustering to identify natural customer groups  
- **Supervised learning:** Logistic Regression for real-time cluster prediction  

---

## 💡 Proposed Solution  
1. Cluster customers with **K-Means** based on their traits and purchase data.  
2. Label each cluster with business-relevant insights.  
3. Train a **Logistic Regression** classifier to predict clusters of new customers.  
4. Deploy the solution as a **FastAPI microservice**, using **MongoDB Atlas** for storage and **AWS** for cloud infrastructure.

---

## 📂 Dataset  
Access the dataset here:  
[Marketing Campaign Dataset](https://github.com/entbappy/Branching-tutorial/blob/master/marketing_campaign.zip)  

---

## 🛠 Tech Stack  
- **Python 3.7**  
- Frameworks: FastAPI, Scikit-learn, Pandas, NumPy  
- Database: MongoDB Atlas  
- Cloud: AWS (S3), Azure  
- CI/CD: GitHub Actions  
- Containerization: Docker  
- Model tuning: GridSearchCV  

---

## 🏗 Infrastructure  
- AWS S3 bucket (for data/model storage)  
- MongoDB Atlas database  
- Azure / AWS EC2 for deployment  
- GitHub Actions for CI/CD pipelines  

---

## 🚀 Quick Start

### Clone the repo  
git clone https://github.com/username/Customer_categorizer.git
cd Customer_categorizer


### Prepare environment  
conda create --prefix venv python=3.10 -y
conda activate venv/
pip install -r requirements.txt


### Set environment variables  
export AWS_ACCESS_KEY_ID=<your_aws_access_key>
export AWS_SECRET_ACCESS_KEY=<your_aws_secret_key>
export AWS_DEFAULT_REGION=<your_aws_region>
export MONGODB_URL=<your_mongodb_connection_string>


### Run the app  
python app.py


### Train model  
Open in browser or send GET request:  
http://localhost:5000/train


### Predict cluster  
http://localhost:5000/predict


---

## 🐳 Docker Instructions

### Build image  
 docker build \
   --build-arg AWS_ACCESS_KEY_ID=<your_aws_access_key> \
   --build-arg AWS_SECRET_ACCESS_KEY=<your_aws_secret_key> \
   --build-arg AWS_DEFAULT_REGION=<your_aws_region> \
   --build-arg MONGODB_URL=<your_mongodb_connection_string> \
   -t customer-segmentation:latest .



### Run container  
docker run -d -p 5000:5000 customer-segmentation:latest


---

## 🗺️ Architecture Visuals

### Project Workflow  
![Project Architecture](https://user-images.githubusercontent.com/71321529/192722336-54016f79-89ef-4c8c-9d71-a6e91ebab03f.jpeg)

### Data Collection Flow  
![Data Collection](https://user-images.githubusercontent.com/71321529/192721926-de265f9b-f301-4943-ac7d-948bff7be9a0.jpeg)

### Deployment Pipeline  
![Deployment](https://user-images.githubusercontent.com/104005791/199660875-c8e63457-432a-44cb-8a95-800870f3da15.png)

---

## 📊 Models Used  
- **K-Means** for customer segmentation  
- **Logistic Regression** for cluster prediction  

**Extras:** Hyperparameter tuning with GridSearchCV

---

## 📁 Codebase Structure  

src/
├── components/ # ML pipeline steps
│ ├── data_ingestion.py
│ ├── data_validation.py
│ ├── data_transformation.py
│ ├── data_clustering.py
│ ├── model_trainer.py
│ ├── model_evaluation.py
│ └── model_pusher.py
├── logger/ # Custom logger
├── exception/ # Custom exceptions
└── utils/ # Helper functions


---

## ✅ Conclusion  
This project empowers companies to leverage machine learning for **dynamic customer segmentation and prediction**, boosting marketing ROI through targeted campaigns and personalized experiences.

👩‍💻 *Integrate with dashboards* like Power BI, Streamlit, or React for actionable business insights.

---

Feel free to reach out for more enhancements or add-ons to make this project even more impactful! 🚀


