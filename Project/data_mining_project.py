# นำเข้าไลบรารีที่จำเป็น
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.cluster import KMeans
import seaborn as sns
import time
from tqdm import tqdm  # ใช้สำหรับ Progress Bar

# กำหนดฟอนต์ภาษาไทย
thai_font = fm.FontProperties(fname="/Library/Fonts/ThaiSarabunNew.ttf")  # ระบุเส้นทางฟอนต์ภาษาไทย

# เพิ่มฟังก์ชันสำหรับแสดงเปอร์เซ็นต์ความคืบหน้า
def show_progress(step, total_steps):
    percent = (step / total_steps) * 100
    print(f"Progress: {percent:.2f}% - ขั้นตอนที่ {step}/{total_steps}")

# 1. เริ่มต้นโปรแกรม
print("เริ่มต้นโปรแกรม... กำลังโหลดไฟล์ข้อมูล")
start_time = time.time()  # เริ่มจับเวลา
show_progress(1, 8)  # ขั้นตอนที่ 1 จากทั้งหมด 8 ขั้นตอน

# 2. โหลดไฟล์ CSV พร้อมกำหนด Encoding
try:
    data = pd.read_csv("/Users/appleclub/Downloads/univ_grd_11_03_2565 - univ_grd_11_03_2565.csv", encoding="utf-8")
    print("ไฟล์ข้อมูลโหลดสำเร็จ! ใช้เวลา: {:.2f} วินาที".format(time.time() - start_time))
    show_progress(2, 8)  # ขั้นตอนที่ 2
except FileNotFoundError:
    print("ไม่พบไฟล์ข้อมูล กรุณาตรวจสอบเส้นทางไฟล์!")
    exit()

# 3. ฟังก์ชันสำหรับ Clean Data และจัดการ Missing Values
def clean_and_prepare_data(df):
    """จัดการ Missing Values และกรองข้อมูลที่ไม่สามารถประมวลผลได้"""
    print("กำลังจัดการ Missing Values และทำความสะอาดข้อมูล...")
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df.fillna("Unknown", inplace=True)
    numeric_df = df.select_dtypes(include=["number"])
    print("จัดการข้อมูลเรียบร้อย!")
    return df, numeric_df

# เรียกใช้ฟังก์ชัน Clean Data
data, numeric_data = clean_and_prepare_data(data)
show_progress(3, 8)  # ขั้นตอนที่ 3

# 4. Heatmap แสดง Correlation เฉพาะตัวเลข
print("กำลังแสดง Heatmap ของข้อมูล...")
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
print("Heatmap แสดงผลเรียบร้อย!")
show_progress(4, 8)  # ขั้นตอนที่ 4

# ตรวจสอบขนาดของข้อมูล
print(f"Shape of numeric_data: {numeric_data.shape}")

# เลือกเฉพาะคอลัมน์ที่สำคัญ
selected_columns = ["AYEAR", "AMOUNT"]  # เพิ่มคอลัมน์ที่ต้องการ
reduced_data = numeric_data[selected_columns]

# สร้าง Heatmap จากข้อมูลที่ลดขนาด
plt.figure(figsize=(10, 6))
sns.heatmap(reduced_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# ตรวจสอบคอลัมน์ที่จำเป็น
required_columns = ["AYEAR", "AMOUNT"]
if not all(col in data.columns for col in required_columns):
    raise KeyError(f"Missing required columns: {required_columns}")

# 5. Regression: Linear Regression และ Decision Tree Regression
try:
    print("กำลังดำเนินการ Regression (Linear และ Decision Tree)...")
    start_time = time.time()  # จับเวลาส่วน Regression
    X = numeric_data[["AYEAR"]]  # ตัวแปรอิสระ (ปีการศึกษา)
    y = numeric_data["AMOUNT"]   # Target Attribute (จำนวนผู้สำเร็จการศึกษา)

    # แบ่งข้อมูลเป็นชุด Train-Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression
    print("กำลังฝึก Linear Regression...")
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)
    reg_predictions = reg_model.predict(X_test)
    mae_linear = mean_absolute_error(y_test, reg_predictions)
    print(f"ผลลัพธ์ Linear Regression: Mean Absolute Error = {mae_linear:.2f}")

    # Decision Tree Regression
    print("กำลังฝึก Decision Tree Regression...")
    for _ in tqdm(range(1), desc="Training Decision Tree Regression", unit="model"):
        dt_reg_model = DecisionTreeRegressor(random_state=42)
        dt_reg_model.fit(X_train, y_train)
    dt_reg_predictions = dt_reg_model.predict(X_test)
    mae_tree = mean_absolute_error(y_test, dt_reg_predictions)
    print(f"ผลลัพธ์ Decision Tree Regression: Mean Absolute Error = {mae_tree:.2f}")

    print("Regression เสร็จสิ้นในเวลา: {:.2f} วินาที".format(time.time() - start_time))
    show_progress(5, 8)  # ขั้นตอนที่ 5
except KeyError:
    print("ข้อผิดพลาด: ไม่พบคอลัมน์ที่จำเป็นสำหรับการ Regression!")

# กราฟเปรียบเทียบ Linear Regression และ Decision Tree Regression
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color="blue", label="ข้อมูลจริง", alpha=0.8)
plt.plot(X_test, reg_predictions, color="red", linewidth=2, label="Linear Regression")
plt.scatter(X_test, dt_reg_predictions, color="green", label="Decision Tree Regression", alpha=0.8)
plt.title("Regression: การพยากรณ์จำนวนผู้สำเร็จการศึกษา", fontproperties=thai_font)
plt.xlabel("ปีการศึกษา", fontproperties=thai_font)
plt.ylabel("จำนวนผู้สำเร็จการศึกษา", fontproperties=thai_font)
plt.legend(prop=thai_font)
plt.grid(True)
plt.show()

# 6. Classification: Random Forest และ Decision Tree Classifier
try:
    print("กำลังดำเนินการ Classification (Random Forest และ Decision Tree)...")
    start_time = time.time()  # จับเวลาส่วน Classification
    data["GRADUATE_STATUS"] = (numeric_data["AMOUNT"] > 100).astype(int)

    X = pd.get_dummies(data[["AYEAR", "FAC_NAME"]], drop_first=True)
    y = data["GRADUATE_STATUS"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Classifier
    print("กำลังฝึก Random Forest Classifier...")
    for _ in tqdm(range(1), desc="Training Random Forest", unit="model"):
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    accuracy_rf = accuracy_score(y_test, rf_predictions)
    print(f"ผลลัพธ์ Random Forest: Accuracy = {accuracy_rf:.2f}")

    # Decision Tree Classifier
    print("กำลังฝึก Decision Tree Classifier...")
    for _ in tqdm(range(1), desc="Training Decision Tree", unit="model"):
        dt_clf_model = DecisionTreeClassifier(random_state=42)
        dt_clf_model.fit(X_train, y_train)
    dt_clf_predictions = dt_clf_model.predict(X_test)
    accuracy_dt = accuracy_score(y_test, dt_clf_predictions)
    print(f"ผลลัพธ์ Decision Tree: Accuracy = {accuracy_dt:.2f}")

    print("Classification เสร็จสิ้นในเวลา: {:.2f} วินาที".format(time.time() - start_time))
    show_progress(6, 8)  # ขั้นตอนที่ 6
except KeyError:
    print("ข้อผิดพลาด: ไม่พบคอลัมน์ที่จำเป็นสำหรับการ Classification!")

# แสดงโครงสร้าง Decision Tree สำหรับ Classification
tree_rules = export_text(dt_clf_model, feature_names=list(X.columns))
print("\nกฎการตัดสินใจใน Decision Tree (Classification):")
print(tree_rules)

# 7. Clustering: การจัดกลุ่มด้วย k-means
try:
    print("กำลังดำเนินการ Clustering ด้วย k-means...")
    start_time = time.time()  # จับเวลาส่วน Clustering
    clustering_features = numeric_data[["AYEAR", "AMOUNT"]]
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    data["Cluster"] = kmeans.fit_predict(clustering_features)

    print("Clustering เสร็จสิ้นในเวลา: {:.2f} วินาที".format(time.time() - start_time))
    show_progress(7, 8)  # ขั้นตอนที่ 7
except KeyError:
    print("ข้อผิดพลาด: ไม่พบคอลัมน์ที่จำเป็นสำหรับการ Clustering!")

# แสดงผลการจัดกลุ่ม
print("\nการจัดกลุ่ม (Clustering):")
print(data[["AYEAR", "AMOUNT", "Cluster"]].head())

# Heatmap แสดง Correlation
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap", fontproperties=thai_font)
plt.show()

# แสดงกราฟผลลัพธ์การจัดกลุ่ม
plt.figure(figsize=(10, 6))
plt.scatter(data["AYEAR"], data["AMOUNT"], c=data["Cluster"], cmap="viridis", s=50)
plt.title("Clustering: การจัดกลุ่มตามปีและจำนวนผู้สำเร็จการศึกษา", fontproperties=thai_font)
plt.xlabel("ปีการศึกษา", fontproperties=thai_font)
plt.ylabel("จำนวนผู้สำเร็จการศึกษา", fontproperties=thai_font)
plt.colorbar(label="Cluster")
plt.grid(True)
plt.show()

# 8. เสร็จสิ้นโปรแกรม
print("โปรแกรมเสร็จสิ้น!")
show_progress(8, 8)  # ขั้นตอนที่ 8
