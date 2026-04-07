import joblib as jb

model = jb.load("MDL.pkl")
vectorizer = jb.load("VTR.pkl")

with open("news1.txt", "r", encoding="utf-8" ) as f:
    news = f.read()

X = vectorizer.transform([news])
 
prediction = model.predict(X)

if prediction[0] == 1:
    print("\n🖥: True news 📰")
else:
    print("\n🖥: False News ❎")

proba = model.predict_proba(X)[0]

print(f"Fake confidence: {proba[0]*100:.2f}%")
print(f"Real confidence: {proba[1]*100:.2f}%")
