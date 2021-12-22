conda activate credit_risk



git rm -r --cached 'data\raw\SouthGermanCredit_data.csv'
git commit -m "stop tracking data\raw\SouthGermanCredit_data.csv"


dvc repro

dvc remove split_data --outs


  205  python src/get_data.py
  206  python src/get_data.py
  207  python Cassandra_Python_Connectivity/connect_database.py
  208  python src/get_data.py
  209  python src/load_data.py
       python src/data_preprocessing.py
  210  python src/split_data.py
  211  dvc repro
  212  dvc repro
  213  git add . && git commit -m "stage 2 complete" && git push origin main
  214  touch src/train_and_evaluate.py

pip install -e .

pytest -v

tox -r

python setup.py sdist bdist_wheel

mkdir -p prediction_services/models

makedir webapp

touch app.py

touch prediction_services/__init__.py
  197  touch prediction_services/prediction.py
  198  mkdir -p webapp/css
  199  mkdir -p webapp/static/css
  200  mkdir -p webapp/static/script
  201  touch webapp/static/css/main.css
  202  touch webapp/static/script/index.js