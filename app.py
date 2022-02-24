#!/usr/bin/env python
# coding: utf-8

# In[46]:


from flask import Flask


# In[47]:


app = Flask(__name__)


# In[48]:


from flask import request, render_template
from keras.models import load_model
import joblib

@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        Income = request.form.get("Income")
        Age = request.form.get("Age")
        Loan = request.form.get("Loan")
        print(Income,Age, Loan)
        
        model = joblib.load("MLPClassifier_CreditDefault")
        pred = model.predict([[float(Income), float(Age), float(Loan)]])
        pred = pred[0]
        s = "The predicted default risk is: " + str(pred)
        return(render_template("index.html", result=s))
    else:
        return(render_template("index.html", result="2"))


# In[ ]:


# only if its your program, then run
if __name__ == "__main__":
    app.run()

