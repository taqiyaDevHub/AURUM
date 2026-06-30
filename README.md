# 🚗 AURUM — Smart Car Configurator

AURUM is an AI-powered smart car configurator built using **Flask**, **Python**, and **Machine Learning**. The application provides an interactive platform where users can browse available car models, customize vehicles with different engine types, colors, and modifications, estimate vehicle prices, receive intelligent configuration recommendations, and explore buyer segmentation through an intuitive web interface.

The project combines a responsive Flask-based frontend with Machine Learning models for price prediction, recommendation generation, and customer segmentation, demonstrating the practical application of data-driven intelligence in a modern vehicle configuration system.

---

# ✨ Features

- 🚘 Browse car models across multiple categories
- ⚙️ Configure vehicles with different engine types, colors, and modifications
- 💰 Predict estimated vehicle prices using Machine Learning
- 🤖 Receive intelligent modification recommendations based on historical user configurations
- 📊 Explore buyer segmentation using K-Means Clustering
- 🔄 RESTful API endpoints built with Flask enable communication between the frontend and backend
- 📁 Create, manage, switch, and delete multiple vehicle configurations within a single session
- 🌐 Interactive and responsive web interface

---

# 🧠 Machine Learning Components

### 💰 Price Prediction
Predicts the estimated vehicle price using a **Linear Regression** model based on the selected vehicle configuration.

### 🤖 Recommendation Engine
Generates intelligent modification recommendations using an **Apriori-inspired Association Rule Mining** approach trained on historical user configuration data.

### 📊 User Segmentation
Analyzes customer preferences and groups buyers into meaningful segments using **K-Means Clustering**.

---

# 🛠️ Technologies Used

### Backend
- Python
- Flask
- REST APIs

### Machine Learning
- Scikit-learn
- Pandas
- NumPy

### Frontend
- HTML5
- CSS3
- JavaScript
- Jinja2 Templates

### Data Storage
- CSV Datasets
- Trained Machine Learning Models (.pkl)

---

# 📂 Project Structure

```text
AURUM/
│
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── car_specifications_dataset.csv
│   ├── car_modifications_dataset.csv
│   └── car_user_configurations.csv
│
├── models/
│   ├── price_predictor.py
│   ├── recommendation_engine.py
│   └── user_segmentation.py
│
├── utils/
│   ├── car_configurator.py
│   └── data_loader.py
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── browse.html
│   ├── configure.html
│   ├── dashboard.html
│   └── segments.html
│
└── saved_models/
    ├── price_predictor.pkl
    ├── recommendation_engine.pkl
    └── user_segmentation.pkl
```

---

# 🚀 Getting Started

### 1️⃣ Clone the repository

```bash
git clone https://github.com/taqiyaDevHub/AURUM.git
cd AURUM
```

### 2️⃣ Install the required dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Machine Learning models *(if the trained model files are unavailable)*

```bash
python train_models.py
```

### 4️⃣ Run the application

```bash
python app.py
```

### 5️⃣ Open your browser and visit

```
http://localhost:5000
```

---

# 🌐 Application Pages

| Page | Description |
|------|-------------|
| 🏠 **Home** | Landing page introducing the application |
| 🚘 **Browse Models** | Explore available vehicle models by category |
| ⚙️ **Car Configurator** | Customize vehicles with engines, colors, and modifications |
| 📋 **Dashboard** | View the selected configuration, estimated price, and AI-generated recommendations |
| 📊 **Buyer Insights** | Explore customer segmentation generated using Machine Learning |

---

# 📊 Datasets

The project uses three custom-designed CSV datasets created for academic and demonstration purposes.

- 🚗 **Car Specifications Dataset** — Contains vehicle specifications and technical details.
- 🛠️ **Car Modifications Dataset** — Contains available modifications, pricing, and compatibility information.
- 👤 **User Configurations Dataset** — Contains historical user configuration data used to train the Machine Learning models.

---

# 🎯 Project Highlights

- ✅ Flask-based web application
- ✅ Interactive vehicle configuration system
- ✅ Machine Learning-powered price prediction
- ✅ Intelligent recommendation engine
- ✅ Customer segmentation using K-Means Clustering
- ✅ REST API integration between frontend and backend
- ✅ Session-based management of multiple vehicle configurations

---

# 🤝 Contributors

This project was developed collaboratively by the following team members:

- **Syeda Taqiya Noman**
- **Zainab Siddiq**
- **Bushra Zafar**
- **Jawerya Shafi**

---

# 👩‍💻 Repository Maintainer

**Syeda Taqiya Noman**

BS Artificial Intelligence  
Dawood University of Engineering & Technology

📧 Email: nomantaqiya31@gmail.com

💼 LinkedIn: https://www.linkedin.com/in/syeda-taqiya-noman

🐙 GitHub: https://github.com/taqiyaDevHub

This repository is maintained by Syeda Taqiya Noman and contains the final version of the group project.

---

# 📄 License

This project was developed as part of an academic group project for educational purposes.
