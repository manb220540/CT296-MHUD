from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Khởi tạo FastAPI
app = FastAPI()

# Cấu hình CORS (cho phép client gửi request từ trình duyệt)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load mô hình đã lưu
# Load different models

try:
    with open("svm_best_model.pkl", "rb") as file:
        svm_model = pickle.load(file)
    with open("knn_best_model.pkl", "rb") as file:
        knn_model = pickle.load(file)
    with open("nb_best_model.pkl", "rb") as file:
        nb_model = pickle.load(file)
    with open("dt_best_model.pkl", "rb") as file:
        dt_model = pickle.load(file)
    with open("rf_best_model.pkl", "rb") as file:
        rf_model = pickle.load(file)
except FileNotFoundError:
    raise FileNotFoundError("Mô hình không được tìm thấy! Vui lòng kiểm tra đường dẫn file.")
except Exception as e:
    raise Exception(f"Không thể tải mô hình: {e}")

# Chuyển đổi dữ liệu từ chuỗi sang số
label_encoders = {
    "parents": {"usual": 0, "pretentious": 1, "great_pret": 2},
    "has_nurs": {"proper": 0, "less_proper": 1, "improper": 2, "critical": 3, "very_crit": 4},
    "form": {"complete": 0, "completed": 1, "incomplete": 2, "foster": 3},
    "children": {"1": 0, "2": 1, "3": 2, "more": 3},
    "housing": {"convenient": 0, "less_conv": 1, "critical": 2},
    "finance": {"convenient": 0, "inconv": 1},
    "social": {"nonprob": 0, "slightly_prob": 1, "problematic": 2},
    "health": {"recommended": 0, "priority": 1, "not_recom": 2},
}

# Chuyển đổi kết quả dự đoán từ số sang chuỗi
class_labels = {0: "Not recommend", 1: "Recommend", 2: "very recommend", 3: "Priority", 4: "special priority"}

# Model input schema
class ModelInput(BaseModel):
    parents: str
    has_nurs: str
    form: str
    children: str
    housing: str
    finance: str
    social: str
    health: str
    model: str

@app.post("/predict/")
def predict(data: ModelInput):
    try:
        # Chuyển đổi dữ liệu từ chuỗi sang số
        input_data = [
            label_encoders["parents"][data.parents],
            label_encoders["has_nurs"][data.has_nurs],
            label_encoders["form"][data.form],
            label_encoders["children"][data.children],
            label_encoders["housing"][data.housing],
            label_encoders["finance"][data.finance],
            label_encoders["social"][data.social],
            label_encoders["health"][data.health],
        ]

        # Chuyển thành mảng NumPy để dự đoán
        input_array = np.array([input_data])
        # Choose the model based on user input
        if data.model == "svm":
            model = svm_model
        elif data.model == "knn":
            model = knn_model
        elif data.model == "naive_bayes":
            model = nb_model
        elif data.model == "decision_tree":
            model = dt_model
        elif data.model == "random_forest":
            model = rf_model
        else:
            return {"error": "Invalid model selected"}
        
        # Dự đoán nhãn chính
        predicted_class = model.predict(input_array)[0]
        # Chuyển đổi nhãn số -> chuỗi
        predicted_label = class_labels[predicted_class]
        # Dự đoán xác suất nếu mô hình hỗ trợ
        try:
            probabilities = model.predict_proba(input_array)[0]
            # Tạo dictionary với xác suất của tất cả các lớp
            probabilities_dict = {class_labels[i]: round(probabilities[i] * 100, 2) for i in range(len(class_labels))}
        except AttributeError:
            probabilities_dict = {"message": "Mô hình không hỗ trợ xác suất."}


        return {
            "predicted_label": predicted_label,
            "probabilities": probabilities_dict
        }
    
    
    except Exception as e:
        return {"error": str(e)}


# Run server
# uvicorn server:app --reload
# khoi tao: venv python -m venv venv
# active venv: source venv/bin/activate