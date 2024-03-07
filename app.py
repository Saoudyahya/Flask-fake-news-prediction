from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import re
import nltk
from nltk.corpus import stopwords
from flask import Flask, render_template, request, jsonify
from nltk.stem import WordNetLemmatizer
from tensorflow import keras

app = Flask(__name__, static_url_path='/static', static_folder='static')

# preprocess functions

nltk.download("stopwords")
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

def pre_process(txt):
  lemmatized_text = []
  for i in txt.split():
    if i in stop_words:
      continue
    else:
      tmp = lemmatizer.lemmatize(i)
      tmp = lemmatizer.lemmatize(tmp, pos='v')
      tmp = lemmatizer.lemmatize(tmp, pos='r')
      tmp = lemmatizer.lemmatize(tmp, pos='s')
      lemmatized_text.append(tmp)
  return lemmatized_text

# ###
def clean_txt(txt):
  text = re.sub('u.s.', 'united state', txt)
  text = re.sub('U.S.', 'united state', txt)
  text = re.sub('U.K.', 'united kingdom', text)
  text = re.sub('u.k.', 'united kingdom', text)
  text = re.sub('US', 'united state', text)
  text = re.sub('UK', 'united kingdom', text)
  text = re.sub(r'\n', ' ', text)
  text = re.sub(r'>(.*?)<', '', text)
  text = re.sub(r'<.*?>', '', text)
  text = re.sub(r'@\w+', '', text)
  text = re.sub(r'[.*?]', '', text)
  text = re.sub(r'watch:', '', text)
  text = re.sub(r'\([^)]*\)', '', text)
  text = re.sub(r'https?://[^\s]+', '', text)
  text = re.sub(r'[^\w\s]', '', text)
  text = re.sub(r'\d+', '', text)
  text = re.sub(r' +', ' ', text)
  text = text.lower()
  return text

# loading the model
model = keras.models.load_model("RNN_BIBI_LSTM128_LSTM64_len_sequ5000_w2vec_200dim_wind10_TFv2.16.0rc0.h5")

# main function to predict
def fake_or_real(new):
  new = clean_txt(new)
  new = pre_process(new)
  new_prd = [new]
  tok = Tokenizer()
  tok.fit_on_texts(new_prd)
  new_prd = tok.texts_to_sequences(new_prd)
  new_prd = pad_sequences(new_prd, maxlen=5000, padding="post")
  pred = model.predict(new_prd)
  if pred >= 0.5:
    return "The new is real : {:.2f} %".format(pred[0][0] * 100)
  else:
    return "The new is fake : {:.2f} %".format((1 - pred[0][0])*100)

@app.route('/')
def index():
    # ... (initial setup) ...
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Initialize an empty string for the user's input
    user_input = ""
    prediction = ""

    # Check if there's a "prompt" parameter in the request
    if request.method == 'POST':
        user_input = request.form['prompt']
        prediction = fake_or_real(user_input)

    return jsonify({"user_input": user_input, "prediction": prediction, "status": "OK"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)