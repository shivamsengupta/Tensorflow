#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Pima Indians Diabetes


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler


# In[2]:


# 1. Load Dataset


df = pd.read_csv("diabetes.csv")   # Update path if needed

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())


# In[3]:


# 2. Feature Ranges


print("\nFeature Ranges:")
for col in df.columns:
    print(f"{col}: Min={df[col].min()}, Max={df[col].max()}")


# In[4]:


# 3. Histograms (All in One Figure)


df.hist(figsize=(15,10), bins=20)
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.show()


# In[5]:


# 4. Correlation Analysis


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# In[6]:


# 5. Handling Zero Values (Optional but Recommended)
# Some features should not realistically contain zeros


columns_with_zero_issue = ["Glucose", "BloodPressure", 
                           "SkinThickness", "Insulin", "BMI"]

for col in columns_with_zero_issue:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].median(), inplace=True)

print("\nAfter Handling Zero Values:")
print(df.describe())


# In[7]:


# 6. Prepare X and y


X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# In[8]:


# 7. Standardization


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFinal X shape:", X_scaled.shape)
print("Final y shape:", y.shape)

print("\nNo dummy variables required (all features are numeric).")


# In[ ]:





# In[ ]:





# In[ ]:




