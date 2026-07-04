from utils import *
from preprocess import *
from train import *

# *** Predictive System ***

# Prediction
input_data = pd.DataFrame(
    [[41,0,1,130,204,0,0,172,0,1.4,2,0,2]],
)

prediction = model.predict(input_data)

print("Predicted Gold Price:", prediction[0])
