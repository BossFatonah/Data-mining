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

# 1. โหลดไฟล์ CSV พร้อมกำหนด Encoding
data = pd.read_csv("/Users/appleclub/Downloads/univ_grd_11_03_2565 - univ_grd_11_03_2565.csv", encoding="utf-8")

# ตรวจสอบข้อมูลเบื้องต้น
print("ตัวอย่างข้อมูล:")
print(data.head())
print("\nข้อมูลเบื้องต้น:")
print(data.info())
print("\nMissing Values:")
print(data.isnull().sum())

# 2. ตั้งค่าฟอนต์สำหรับภาษาไทยใน Matplotlib
font_path = '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'  # ใช้ฟอนต์ที่รองรับภาษาไทย
thai_font = fm.FontProperties(fname=font_path)

# 3. Regression: ทำนายจำนวนผู้จบการศึกษาในอนาคต (Linear Regression และ Decision Tree Regression)
X = data[["AYEAR"]]  # ตัวแปรอิสระ (ปีการศึกษา)
y = data["AMOUNT"]   # Target Attribute (จำนวนผู้สำเร็จการศึกษา)

# แบ่งข้อมูลเป็นชุด Train-Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
reg_predictions = reg_model.predict(X_test)
mae_linear = mean_absolute_error(y_test, reg_predictions)
print(f"\nMean Absolute Error (Linear Regression): {mae_linear:.2f}")

# Decision Tree Regression
dt_reg_model = DecisionTreeRegressor(random_state=42)
dt_reg_model.fit(X_train, y_train)
dt_reg_predictions = dt_reg_model.predict(X_test)
mae_tree = mean_absolute_error(y_test, dt_reg_predictions)
print(f"\nMean Absolute Error (Decision Tree Regression): {mae_tree:.2f}")

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

# 4. Classification: การจำแนกสถานะการจบ (GRADUATE_STATUS) ด้วย Random Forest และ Decision Tree
data["GRADUATE_STATUS"] = (data["AMOUNT"] > 100).astype(int)  # สถานะจบ (1) หรือไม่จบ (0)

X = pd.get_dummies(data[["AYEAR", "FAC_NAME"]])  # แปลงข้อมูล Text เป็นตัวเลข
y = data["GRADUATE_STATUS"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, rf_predictions)
print(f"\nAccuracy (Random Forest - Classification): {accuracy_rf:.2f}")

# Decision Tree Classifier
dt_clf_model = DecisionTreeClassifier(random_state=42)
dt_clf_model.fit(X_train, y_train)
dt_clf_predictions = dt_clf_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, dt_clf_predictions)
print(f"Accuracy (Decision Tree - Classification): {accuracy_dt:.2f}")

# แสดงโครงสร้าง Decision Tree สำหรับ Classification
tree_rules = export_text(dt_clf_model, feature_names=list(X.columns))
print("\nกฎการตัดสินใจใน Decision Tree (Classification):")
print(tree_rules)

# 5. Clustering: การจัดกลุ่มข้อมูล (k-means)
clustering_features = data[["AYEAR", "AMOUNT"]]

kmeans = KMeans(n_clusters=3, random_state=42)
data["Cluster"] = kmeans.fit_predict(clustering_features)

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
